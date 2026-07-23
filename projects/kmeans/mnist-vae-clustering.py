import time
import jax
import os, sys
import optax
import logging
import random
import string
from functools import partial
from jax import Array
import jax.numpy as jnp
from typing import Dict, Any, Iterable, Tuple
import numpy as np

# Update import paths for the new location
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from projects.kmeans.visualize import (
    create_probability_animation, 
    plot_training_progress, 
    plot_final_probabilities,
    create_clustering_visualization,
    create_clustering_visualization_generative,
    create_probability_animation_with_reconstructions,
    sample_cluster_examples,
    plot_cluster_samples_grid
)

def generate_uid() -> str:
    """Generate a 4-character UID using alphanumeric characters"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))

# Generate UID for this run
run_uid = generate_uid()
print(f"Run UID: {run_uid}")

# Set up logging (simple like mnist-base.py)
logging.basicConfig(level=logging.INFO)

# Global constants
K = 15  # Number of clusters
USE_CLUSTER_BASED_LOSS = True  # True: cluster-based sum, False: direct mean
input_dim = 784

def init() -> Tuple[Dict[str, Any], Dict[str, Array], Any]:
    """Initialize VAE parameters for K clusters using tensors"""

    config = {
        "dataset_name": "mnist",
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "random_seed": 2,
        "img_dims": (28, 28),  # Image dimensions
        "K": K,  # Number of clusters
        "n_samples": 60000,
        "latent_dim": 16,  # VAE latent dimension
        "n_hidden": 128,  # Hidden layer dimension for encoder/decoder
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    # For MNIST, we know the flattened input dimension is 28*28 = 784
    n_hidden = config["n_hidden"]
    latent_dim = config["latent_dim"]
    
    # Initialize VAE parameters as tensors with K as first dimension
    params = {
        # Encoder parameters (K, input_dim, n_hidden) etc.
        'enc_w1': jax.random.normal(next(key_gen).get(), (K, input_dim, n_hidden)) * jnp.sqrt(2./input_dim),
        'enc_b1': jnp.zeros((K, n_hidden)),
        'enc_w_mean': jax.random.normal(next(key_gen).get(), (K, n_hidden, latent_dim)) * jnp.sqrt(2./n_hidden),
        'enc_b_mean': jnp.zeros((K, latent_dim)),
        'enc_w_logvar': jax.random.normal(next(key_gen).get(), (K, n_hidden, latent_dim)) * jnp.sqrt(2./n_hidden),
        'enc_b_logvar': jnp.zeros((K, latent_dim)),
        
        # Decoder parameters
        'dec_w1': jax.random.normal(next(key_gen).get(), (K, latent_dim, n_hidden)) * jnp.sqrt(2./latent_dim),
        'dec_b1': jnp.zeros((K, n_hidden)),
        'dec_w2': jax.random.normal(next(key_gen).get(), (K, n_hidden, input_dim)) * jnp.sqrt(2./n_hidden),
        'dec_b2': jnp.zeros((K, input_dim))
    }
    
    # Debug: print parameter shapes
    logging.info(f"VAE tensorized parameter shapes:")
    logging.info(f"  enc_w1: {params['enc_w1'].shape}")
    logging.info(f"  enc_w_mean: {params['enc_w_mean'].shape}")
    logging.info(f"  dec_w1: {params['dec_w1'].shape}")
    logging.info(f"  dec_w2: {params['dec_w2'].shape}")
    logging.info(f"  Total clusters (K): {K}")
    
    return config, params, key_gen

def vae_encode(params: Dict[str, Array], x: Array, cluster_id: int) -> Tuple[Array, Array]:
    """VAE encoder for a specific cluster using tensor indexing"""
    # Encoder forward pass using tensor indexing
    h1 = jnp.dot(x, params['enc_w1'][cluster_id]) + params['enc_b1'][cluster_id]
    h1 = jax.nn.relu(h1)
    
    # Mean and log variance
    z_mean = jnp.dot(h1, params['enc_w_mean'][cluster_id]) + params['enc_b_mean'][cluster_id]
    z_logvar = jnp.dot(h1, params['enc_w_logvar'][cluster_id]) + params['enc_b_logvar'][cluster_id]
    
    return z_mean, z_logvar

def vae_decode(params: Dict[str, Array], z: Array, cluster_id: int) -> Array:
    """VAE decoder for a specific cluster using tensor indexing"""
    # Decoder forward pass using tensor indexing
    h1 = jnp.dot(z, params['dec_w1'][cluster_id]) + params['dec_b1'][cluster_id]
    h1 = jax.nn.relu(h1)
    
    # Reconstruction (using sigmoid to get [0,1] range)
    x_recon = jnp.dot(h1, params['dec_w2'][cluster_id]) + params['dec_b2'][cluster_id]
    x_recon = jax.nn.sigmoid(x_recon)
    
    return x_recon

def reparameterize(z_mean: Array, z_logvar: Array, key: Array) -> Array:
    """Reparameterization trick for VAE"""
    eps = jax.random.normal(key, z_mean.shape)
    return z_mean + jnp.exp(0.5 * z_logvar) * eps

def vae_reconstruction_loss(params: Dict[str, Array], x: Array, cluster_id: int, key: Array) -> Array:
    """Compute reconstruction loss for a single sample using a specific VAE cluster"""
    # Encode
    z_mean, z_logvar = vae_encode(params, x, cluster_id)
    
    # Reparameterize
    z = reparameterize(z_mean, z_logvar, key)
    
    # Decode
    x_recon = vae_decode(params, z, cluster_id)
    
    # Reconstruction loss (binary cross-entropy)
    recon_loss = -jnp.sum(x * jnp.log(x_recon + 1e-8) + (1 - x) * jnp.log(1 - x_recon + 1e-8))
    
    return recon_loss

def get_reconstruction_losses(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get reconstruction losses for a single sample across all K clusters"""
    losses = []
    
    for k in range(K):
        loss = vae_reconstruction_loss(params, x, k, key)
        losses.append(loss)
    
    return jnp.array(losses)

def get_K(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get cluster assignment by sampling from probability distribution"""
    # probabilities = get_probabilities(params, x, key)
    # return jax.random.categorical(key, jnp.log(probabilities + 1e-8))
    return jnp.argmin(get_reconstruction_losses(params, x, key))

def get_K_batch(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get cluster assignments for a batch of samples"""
    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(get_K, in_axes=(None, 0, 0))(params, x, keys)

def get_probabilities(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get probability distribution based on reconstruction losses (softmax of negative losses)"""
    losses = get_reconstruction_losses(params, x, key)
    # Convert losses to probabilities (lower loss = higher probability)
    # Apply logit normalization: subtract mean and divide by std
    logits = -losses
    logits = (logits - jnp.mean(logits)) / (jnp.std(logits) + 1e-8)
    probabilities = jax.nn.softmax(logits)
    return probabilities

def get_probabilities_batch(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get probability distributions for a batch of samples"""
    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(get_probabilities, in_axes=(None, 0, 0))(params, x, keys)

def get_reconstruction(params: Dict[str, Array], x: Array, cluster_id: int, key: Array) -> Array:
    """Get reconstruction of input using specific cluster's VAE"""
    # Encode
    z_mean, z_logvar = vae_encode(params, x, cluster_id)
    
    # Reparameterize
    z = reparameterize(z_mean, z_logvar, key)
    
    # Decode
    x_recon = vae_decode(params, z, cluster_id)
    
    return x_recon

def get_reconstruction_batch(params: Dict[str, Array], x: Array, cluster_assignments: Array, key: Array) -> Array:
    """Get reconstructions for a batch of samples using their assigned clusters"""
    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(get_reconstruction, in_axes=(None, 0, 0, 0))(params, x, cluster_assignments, keys)

def vae_loss_with_kl(params: Dict[str, Array], x: Array, cluster_id: int, key: Array) -> Array:
    """Full VAE loss including KL divergence for a single sample"""
    # Encode
    z_mean, z_logvar = vae_encode(params, x, cluster_id)
    
    # Reparameterize
    z = reparameterize(z_mean, z_logvar, key)
    
    # Decode
    x_recon = vae_decode(params, z, cluster_id)
    
    # Reconstruction loss (binary cross-entropy)
    recon_loss = -jnp.sum(x * jnp.log(x_recon + 1e-8) + (1 - x) * jnp.log(1 - x_recon + 1e-8))
    
    # KL divergence loss
    kl_loss = -0.5 * jnp.sum(1 + z_logvar - jnp.square(z_mean) - jnp.exp(z_logvar))
    
    return recon_loss + kl_loss

def loss_batch_vae(params: Dict[str, Array], xs: Array, selected_k: Array, key: Array) -> Array:
    """VAE batch loss function - supports cluster-based sum or direct mean based on global flag"""
    batch_size = xs.shape[0]
    keys = jax.random.split(key, batch_size)
    
    # Vectorized loss computation
    def single_sample_loss(x, cluster_id, sample_key):
        return vae_loss_with_kl(params, x, cluster_id, sample_key)
    
    # Use vmap to compute losses for all samples
    losses = jax.vmap(single_sample_loss)(xs, selected_k, keys)
    
    if USE_CLUSTER_BASED_LOSS:
        # Cluster-based sum: compute mean loss for each cluster then sum
        # Create one-hot encoding for cluster assignments: (batch_size, K)
        cluster_masks = jax.nn.one_hot(selected_k, K)
        
        # For each cluster, compute sum of losses and count of samples
        cluster_loss_sums = jnp.dot(cluster_masks.T, losses)  # (K,)
        cluster_counts = jnp.sum(cluster_masks, axis=0)  # (K,)
        
        # Compute mean loss for each cluster (avoid division by zero)
        cluster_mean_losses = jnp.where(
            cluster_counts > 0,
            cluster_loss_sums / cluster_counts,
            0.0
        )
        
        # Sum mean losses across all clusters
        return jnp.sum(cluster_mean_losses)
    else:
        # Direct mean: simply return mean of all sample losses
        return jnp.mean(losses)

@jax.jit
def train_step(params: Dict[str, Array], optimizer_state: Any, x: Array, selected_k: Array, key: Array) -> Tuple[Dict[str, Array], Any, Array, Any]:
    """Training step using VAE loss"""
    loss, grads = jax.value_and_grad(loss_batch_vae)(params, x, selected_k, key)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, grads

if __name__ == "__main__":
    # Initialize model and config
    config, params, key_gen = init()
    
    # Load dataset
    logging.info("Loading MNIST dataset...")
    data = load_supervised_image(config["dataset_name"])
    
    # Normalize pixel values to [0, 1] and flatten
    X_train = data.X.reshape(data.n_samples, -1) / 255.0
    X_test = data.X_test.reshape(data.n_test_samples, -1) / 255.0
    y_train = data.y
    y_test = data.y_test
    
    logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Limit samples for faster training
    num_samples = config["n_samples"]
    X_train = X_train[:num_samples]
    y_train = y_train[:num_samples]
    
    logging.info(f"Using {num_samples} samples for training")
    
    # Select 12 examples for visualization
    num_examples = 12
    example_indices = np.random.choice(num_samples, num_examples, replace=False)
    X_examples = X_train[example_indices]
    y_examples = y_train[example_indices]
    
    logging.info(f"Selected {num_examples} examples for visualization: {example_indices}")
    
    # Initialize optimizer
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    
    # Training loop
    logging.info("Starting VAE clustering training...")
    start_time = time.perf_counter()
    
    train_losses = []
    train_accuracies = []
    animation_frames = []
    reconstruction_frames = []  # Store reconstruction data for animation
    
    num_batches = num_samples // config["batch_size"]
    frames_per_epoch = min(200, num_batches)
    
    # Safety check: ensure we have at least one batch
    if num_batches == 0:
        logging.warning(f"num_samples ({num_samples}) is smaller than batch_size ({config['batch_size']}). Setting num_batches to 1.")
        num_batches = 1
        frames_per_epoch = 1
    
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for batch in range(num_batches):
        start_idx = batch * config["batch_size"]
        end_idx = min(start_idx + config["batch_size"], num_samples)
        
        batch_x = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        # Get cluster assignments for this batch based on current VAE reconstruction losses
        batch_key = next(key_gen).get()
        k = get_K_batch(params, batch_x, batch_key)
        
        # Training step with VAE loss
        train_key = next(key_gen).get()
        params, opt_state, loss_value, grads = train_step(params, opt_state, batch_x, k, train_key)
        
        # Compute accuracy for this batch
        batch_accuracy = jnp.mean(k == batch_y)
        
        epoch_loss += loss_value
        epoch_accuracy += batch_accuracy
        
        train_losses.append(loss_value)
        train_accuracies.append(batch_accuracy)
        
        # Collect animation frames at regular intervals
        frame_interval = max(1, num_batches // frames_per_epoch)
        if batch % frame_interval == 0 or batch == num_batches - 1:
            # Get probability distributions for the selected examples
            example_key = next(key_gen).get()
            example_probabilities = get_probabilities_batch(params, X_examples, example_key)
            
            # Calculate cluster distribution for ALL training samples
            all_key = next(key_gen).get()
            all_training_probabilities = get_probabilities_batch(params, X_train, all_key)
            all_cluster_assignments = np.argmax(all_training_probabilities, axis=1)
            all_cluster_counts = np.bincount(all_cluster_assignments, minlength=K)
            
            # Generate reconstructions for animation examples using current parameters
            example_cluster_assignments = np.argmax(example_probabilities, axis=1)
            recon_key = next(key_gen).get()
            example_reconstructions = get_reconstruction_batch(params, X_examples, example_cluster_assignments, recon_key)
            
            # Create animation frame
            animation_frame = {
                'probabilities': np.array(example_probabilities),
                'cluster_counts': all_cluster_counts,
                'epoch': 0,
                'batch': batch,
                'total_batches': num_batches
            }
            animation_frames.append(animation_frame)
            reconstruction_frames.append(np.array(example_reconstructions))
            
            logging.info(f"Animation frame collected - Batch {batch}/{num_batches}, Loss: {loss_value:.4f}")
        
        if batch % 100 == 0:
            # Calculate current cluster distribution for all training data processed so far
            current_key = next(key_gen).get()
            current_probabilities = get_probabilities_batch(params, X_train[:end_idx], current_key)
            current_assignments = np.argmax(current_probabilities, axis=1)
            current_cluster_counts = np.bincount(current_assignments, minlength=K)
            non_empty = np.sum(current_cluster_counts > 0)
            
            logging.info(f"Batch {batch}/{num_batches}, Loss: {loss_value:.4f}, Accuracy: {batch_accuracy:.4f}")
            logging.info(f"  Cluster distribution: {current_cluster_counts}")
            logging.info(f"  Non-empty clusters: {non_empty}/{K} ({non_empty/K*100:.1f}%)")
    
    # Safety check for division by zero
    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
    else:
        avg_loss = epoch_loss
        avg_accuracy = epoch_accuracy
    
    logging.info(f"VAE clustering training completed - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    
    training_time = time.perf_counter() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Show final probability distributions
    final_key = next(key_gen).get()
    final_probabilities = get_probabilities_batch(params, X_examples, final_key)
    
    # Calculate cluster distribution for ALL training samples
    logging.info("Calculating cluster distribution for all training samples...")
    final_all_key = next(key_gen).get()
    all_training_probabilities = get_probabilities_batch(params, X_train, final_all_key)
    all_cluster_assignments = np.argmax(all_training_probabilities, axis=1)
    all_cluster_counts = np.bincount(all_cluster_assignments, minlength=K)
    
    logging.info(f"Final cluster distribution: {all_cluster_counts}")
    logging.info(f"Total samples distributed: {np.sum(all_cluster_counts)}")
    
    # Show cluster usage percentage
    cluster_percentages = all_cluster_counts / np.sum(all_cluster_counts) * 100
    logging.info(f"Cluster usage percentages: {cluster_percentages}")
    
    # Count non-empty clusters
    non_empty_clusters = np.sum(all_cluster_counts > 0)
    logging.info(f"Number of non-empty clusters: {non_empty_clusters}/{K}")
    
    # Get final cluster assignments for examples
    final_example_key = next(key_gen).get()
    final_example_cluster_assignments = get_K_batch(params, X_examples, final_example_key)
    
    # Generate reconstructions using assigned clusters
    logging.info("Generating VAE reconstructions for visualization examples...")
    recon_key = next(key_gen).get()
    X_example_reconstructions = get_reconstruction_batch(params, X_examples, final_example_cluster_assignments, recon_key)
    
    # Create generative visualizations with reconstructions (includes all features from basic visualization)
    logging.info("Creating generative clustering visualizations with reconstructions...")
    all_saved_files = create_clustering_visualization_generative(
        animation_frames=animation_frames,
        final_probabilities=final_probabilities,
        X_examples=X_examples,
        y_examples=y_examples,
        X_reconstructions=X_example_reconstructions,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        run_uid=run_uid,
        all_cluster_counts=all_cluster_counts,
        cluster_assignments=final_example_cluster_assignments,
        reconstruction_frames=reconstruction_frames
        # output_dir defaults to outputs/yy-mm-dd/
    )
    
    # Create cluster sample grid visualization
    logging.info("Creating cluster sample grid visualization...")
    sampled_images, sampled_labels = sample_cluster_examples(
        X_train=X_train,
        y_train=y_train,
        cluster_assignments=all_cluster_assignments,
        n_samples_per_cluster=5,
        K=K
    )
    
    plot_cluster_samples_grid(
        sampled_images=sampled_images,
        sampled_labels=sampled_labels,
        run_uid=run_uid,
        K=K,
        n_samples_per_cluster=5
    )
    
    logging.info(f"All visualizations created and saved to: {all_saved_files}")
    
    logging.info("VAE clustering training completed! Check the saved plots and animation for results.")