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
from typing import Dict, Any, Iterable, Tuple, NamedTuple
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
K = 16  # Number of clusters
input_dim = 784

class PoolState(NamedTuple):
    """JAX-compatible sample pool state - stores indices only, losses computed on-demand"""
    sample_indices: Array  # (K, max_size) - indices into the original dataset
    counts: Array          # (K,) - number of samples in each pool

def init_pool_state(K: int, max_size: int, input_dim: int) -> PoolState:
    """Initialize empty pool state"""
    return PoolState(
        sample_indices=jnp.full((K, max_size), -1, dtype=jnp.int32),  # -1 indicates empty slot
        counts=jnp.zeros(K, dtype=jnp.int32)
    )

def update_pool_vectorized(pool_state: PoolState, batch_indices: Array, new_losses: Array, cluster_assignments: Array, 
                          params: Dict[str, Array], X_train: Array, key: Array) -> PoolState:
    """Vectorized pool update using padding and masks"""
    K, max_size = pool_state.sample_indices.shape
    
    # Precompute all pool losses in one vectorized operation
    all_pool_samples = X_train[pool_state.sample_indices]  # Shape: (K, max_size, input_dim)
    valid_mask = pool_state.sample_indices >= 0  # Shape: (K, max_size)
    
    # Create cluster IDs for each pool position
    cluster_ids = jnp.arange(K)[:, None]  # Shape: (K, 1)
    cluster_ids = jnp.broadcast_to(cluster_ids, (K, max_size))  # Shape: (K, max_size)
    
    # Generate keys for all pool positions
    pool_keys = jax.random.split(key, K * max_size).reshape(K, max_size, -1)
    
    # Vectorized loss computation: vmap over K, then over max_size
    all_pool_losses = jax.vmap(jax.vmap(vae_loss_with_kl, in_axes=(None, 0, 0, 0)), in_axes=(None, 0, 0, 0))(
        params, all_pool_samples, cluster_ids, pool_keys
    )
    # Mask invalid entries with high loss
    all_pool_losses = jnp.where(valid_mask, all_pool_losses, jnp.inf)
    
    new_sample_indices = pool_state.sample_indices
    new_counts = pool_state.counts
    
    # Process each cluster individually
    for k in range(K):
        # Get new samples for this cluster
        mask = cluster_assignments == k
        n_cluster_samples = jnp.sum(mask)
        
        if n_cluster_samples > 0:
            # Get indices using jnp.where
            indices_in_batch = jnp.where(mask, size=batch_indices.shape[0], fill_value=-1)[0]
            cluster_indices = batch_indices[indices_in_batch[:n_cluster_samples]]
            cluster_losses = new_losses[indices_in_batch[:n_cluster_samples]]
            
            # Get current pool indices and precomputed losses
            current_count = new_counts[k]
            current_indices = new_sample_indices[k, :current_count]
            current_losses = all_pool_losses[k, :current_count] if current_count > 0 else jnp.array([])
            
            # Combine all samples and losses
            all_indices = jnp.concatenate([current_indices, cluster_indices])
            all_losses = jnp.concatenate([current_losses, cluster_losses])
            
            # Sort by loss and take top max_size
            sort_order = jnp.argsort(all_losses)
            final_size = min(len(all_indices), max_size)
            best_indices = all_indices[sort_order[:final_size]]
            
            # Pad to maintain shape
            padded = jnp.pad(best_indices, (0, max_size - final_size), constant_values=-1)
            
            # Update arrays
            new_sample_indices = new_sample_indices.at[k].set(padded)
            new_counts = new_counts.at[k].set(final_size)
    
    return pool_state._replace(
        sample_indices=new_sample_indices,
        counts=new_counts
    )


def get_full_pool_training_batch(pool_state: PoolState, params: Dict[str, Array], X_train: Array, y_train: Array, key: Array, max_batch_size: int = 1024) -> Tuple[Array, Array, Array, Array]:
    """Get training batch using ALL samples from all pools with fixed size padding and masking"""
    # Get valid indices from all pools
    valid_mask = pool_state.sample_indices >= 0
    all_indices = pool_state.sample_indices[valid_mask]
    
    # Create cluster assignments for each index
    cluster_ids = jnp.repeat(jnp.arange(pool_state.counts.shape[0]), pool_state.counts)
    
    if len(all_indices) > 0:
        # Look up data and shuffle
        batch_samples = X_train[all_indices]
        batch_labels = y_train[all_indices]
        
        shuffle_idx = jax.random.permutation(key, len(all_indices))
        batch_samples = batch_samples[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        cluster_ids = cluster_ids[shuffle_idx]
        
        # Limit to max_batch_size if needed
        actual_size = min(len(all_indices), max_batch_size)
        batch_samples = batch_samples[:actual_size]
        batch_labels = batch_labels[:actual_size]
        cluster_ids = cluster_ids[:actual_size]
        
        # Pad to fixed size with masking
        input_dim = X_train.shape[1]
        padded_samples = jnp.zeros((max_batch_size, input_dim))
        padded_labels = jnp.zeros(max_batch_size, dtype=jnp.int32)
        padded_cluster_ids = jnp.zeros(max_batch_size, dtype=jnp.int32)
        
        # Create mask: True for real data, False for padding
        mask = jnp.zeros(max_batch_size, dtype=jnp.bool_)
        
        # Fill in the real data
        padded_samples = padded_samples.at[:actual_size].set(batch_samples)
        padded_labels = padded_labels.at[:actual_size].set(batch_labels)
        padded_cluster_ids = padded_cluster_ids.at[:actual_size].set(cluster_ids)
        mask = mask.at[:actual_size].set(True)
        
        logging.info(f"Training batch from pools: {actual_size}/{max_batch_size} samples (padded)")
        return padded_samples, padded_labels, padded_cluster_ids, mask
    else:
        # Return empty padded arrays
        input_dim = X_train.shape[1]
        return (jnp.zeros((max_batch_size, input_dim)), 
                jnp.zeros(max_batch_size, dtype=jnp.int32), 
                jnp.zeros(max_batch_size, dtype=jnp.int32),
                jnp.zeros(max_batch_size, dtype=jnp.bool_))

def init() -> Tuple[Dict[str, Any], Dict[str, Array], Any]:
    """Initialize VAE parameters for K clusters using tensors"""

    config = {
        "dataset_name": "mnist",
        "batch_size": 512,
        "learning_rate": 1e-3,
        "random_seed": 2,
        "img_dims": (28, 28),  # Image dimensions
        "K": K,  # Number of clusters
        "n_samples": 60000,
        "latent_dim": 32,  # VAE latent dimension
        "n_hidden": 128,  # Hidden layer dimension for encoder/decoder
        "pool_size": 512,  # Size of priority pool per cluster 
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
    logging.info(f"  Pool size per cluster: {config['pool_size']}")
    
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

@jax.jit
def get_reconstruction_losses(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get reconstruction losses for a single sample across all K clusters - vectorized"""
    keys = jax.random.split(key, K)
    losses = jax.vmap(lambda k, k_key: vae_reconstruction_loss(params, x, k, k_key))(
        jnp.arange(K), keys)
    return losses

@jax.jit
def get_reconstruction_losses_batch_all_clusters(params: Dict[str, Array], x_batch: Array, key: Array) -> Array:
    """Get reconstruction losses for batch of samples across all clusters - fully vectorized
    Returns: (batch_size, K) array of losses
    """
    batch_size = x_batch.shape[0]
    keys = jax.random.split(key, batch_size)
    
    # Vectorize over batch dimension
    losses_batch = jax.vmap(get_reconstruction_losses, in_axes=(None, 0, 0))(
        params, x_batch, keys)
    
    return losses_batch  # Shape: (batch_size, K)


def get_K_batch(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get cluster assignments for a batch of samples"""
    keys = jax.random.split(key, x.shape[0])
    # Vectorized version of argmin(get_reconstruction_losses())
    return jax.vmap(lambda x, k: jnp.argmin(get_reconstruction_losses(params, x, k)), in_axes=(0, 0))(x, keys)

def get_probabilities_batch(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get probability distributions for a batch of samples"""
    keys = jax.random.split(key, x.shape[0])
    
    def single_prob(x, k):
        losses = get_reconstruction_losses(params, x, k)
        # Convert losses to probabilities (lower loss = higher probability)
        logits = -losses
        logits = (logits - jnp.mean(logits)) / (jnp.std(logits) + 1e-8)
        return jax.nn.softmax(logits)
    
    return jax.vmap(single_prob, in_axes=(0, 0))(x, keys)

def get_reconstruction_batch(params: Dict[str, Array], x: Array, cluster_assignments: Array, key: Array) -> Array:
    """Get reconstructions for a batch of samples using their assigned clusters"""
    keys = jax.random.split(key, x.shape[0])
    
    def single_reconstruction(x, cluster_id, k):
        # Encode
        z_mean, z_logvar = vae_encode(params, x, cluster_id)
        # Reparameterize  
        z = reparameterize(z_mean, z_logvar, k)
        # Decode
        return vae_decode(params, z, cluster_id)
    
    return jax.vmap(single_reconstruction, in_axes=(0, 0, 0))(x, cluster_assignments, keys)

def sample_from_vae(params: Dict[str, Array], cluster_id: int, key: Array, latent_dim: int = 16) -> Array:
    """Sample a new image from the VAE decoder for a specific cluster"""
    # Sample random latent vector from standard normal distribution
    z = jax.random.normal(key, (latent_dim,))
    
    # Decode using the specified cluster's decoder
    x_generated = vae_decode(params, z, cluster_id)
    
    return x_generated

def sample_vae_cluster_examples(params: Dict[str, Array], n_samples_per_cluster: int, K: int, key: Array, latent_dim: int = 16) -> Array:
    """Sample generated images from each cluster's VAE decoder"""
    all_generated_samples = []
    
    for k in range(K):
        cluster_samples = []
        cluster_key = jax.random.split(key, 1)[0]
        sample_keys = jax.random.split(cluster_key, n_samples_per_cluster)
        
        for i in range(n_samples_per_cluster):
            generated_sample = sample_from_vae(params, k, sample_keys[i], latent_dim)
            cluster_samples.append(generated_sample)
        
        cluster_samples = jnp.array(cluster_samples)
        all_generated_samples.append(cluster_samples)
    
    return jnp.array(all_generated_samples)  # Shape: (K, n_samples_per_cluster, input_dim)

def plot_vae_generated_samples_grid(generated_samples: Array, run_uid: str, K: int, n_samples_per_cluster: int, img_dims: tuple = (28, 28)):
    """Plot a grid of VAE-generated samples for each cluster"""
    import matplotlib.pyplot as plt
    from shared_lib.media import save_matplotlib_figure
    
    # Reshape samples for visualization
    generated_samples = generated_samples.reshape(K, n_samples_per_cluster, img_dims[0], img_dims[1])
    
    fig, axes = plt.subplots(K, n_samples_per_cluster, figsize=(n_samples_per_cluster * 2, K * 2))
    fig.suptitle(f'VAE Generated Samples by Cluster (Run: {run_uid})', fontsize=14)
    
    # Ensure axes is 2D even if K=1 or n_samples_per_cluster=1
    if K == 1:
        axes = axes.reshape(1, -1)
    elif n_samples_per_cluster == 1:
        axes = axes.reshape(-1, 1)
    
    for k in range(K):
        for i in range(n_samples_per_cluster):
            ax = axes[k, i]
            
            # Display the generated image
            img = generated_samples[k, i]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Add cluster label on the first sample of each row
            if i == 0:
                ax.set_ylabel(f'Cluster {k}', rotation=90, fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    filename = f"vae_generated_samples_grid_{run_uid}.png"
    save_matplotlib_figure(filename, fig)
    plt.close(fig)
    
    return filename

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

def loss_batch_vae(params: Dict[str, Array], xs: Array, selected_k: Array, key: Array, mask: Array = None) -> Array:
    """VAE batch loss function with optional masking for padded data"""
    batch_size = xs.shape[0]
    keys = jax.random.split(key, batch_size)
    
    # Vectorized loss computation
    def single_sample_loss(x, cluster_id, sample_key):
        return vae_loss_with_kl(params, x, cluster_id, sample_key)
    
    # Use vmap to compute losses for all samples
    losses = jax.vmap(single_sample_loss)(xs, selected_k, keys)
    
    if mask is not None:
        # Apply mask: zero out losses for padded samples
        masked_losses = jnp.where(mask, losses, 0.0)
        # Compute mean only over valid (non-padded) samples
        valid_count = jnp.sum(mask)
        return jnp.where(valid_count > 0, jnp.sum(masked_losses) / valid_count, 0.0)
    else:
        # Original behavior: direct mean of all sample losses
        return jnp.mean(losses)

@jax.jit
def train_step(params: Dict[str, Array], optimizer_state: Any, x: Array, selected_k: Array, key: Array, mask: Array = None) -> Tuple[Dict[str, Array], Any, Array, Any]:
    """Training step using VAE loss with optional masking"""
    loss, grads = jax.value_and_grad(lambda p, xs, sk, k: loss_batch_vae(p, xs, sk, k, mask))(params, x, selected_k, key)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, grads

@jax.jit 
def compute_cluster_assignments_and_losses(params: Dict[str, Array], samples: Array, key: Array) -> Tuple[Array, Array]:
    """Compute cluster assignments and losses for a batch of samples - fully vectorized"""
    # Get losses for all samples across all clusters: (batch_size, K)
    all_losses = get_reconstruction_losses_batch_all_clusters(params, samples, key)
    
    # Get cluster assignments (argmin of losses)
    cluster_assignments = jnp.argmin(all_losses, axis=1)  # (batch_size,)
    
    # Get the loss for each sample with its assigned cluster
    batch_indices = jnp.arange(samples.shape[0])
    assigned_losses = all_losses[batch_indices, cluster_assignments]  # (batch_size,)
    
    return cluster_assignments, assigned_losses

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
    
    # Initialize JAX-based sample pool state (index-based)
    pool_state = init_pool_state(K, config["pool_size"], input_dim)
    
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
    logging.info("Starting VAE clustering training with replay pools...")
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
    
    # Start training immediately with alternating pool updates and replay training
    logging.info("Starting alternating pool updates and replay training...")
    
    pool_update_frequency = 3  # Update pools every 3 iterations (2:1 replay ratio)
    training_iterations = num_batches * 2  # More training iterations since we're doing full pool training
    
    for iteration in range(training_iterations):
        if iteration % pool_update_frequency == 0:
            # POOL UPDATE PHASE: Process new samples for cluster assignment
            batch_idx = iteration // pool_update_frequency
            start_idx = batch_idx * config["batch_size"]
            end_idx = min(start_idx + config["batch_size"], num_samples)
            
            if start_idx < num_samples:
                # Get new batch for processing
                new_batch_x = X_train[start_idx:end_idx]
                new_batch_y = y_train[start_idx:end_idx]
                
                # Compute cluster assignments and losses for new batch
                pool_update_key = next(key_gen).get()
                cluster_assignments, losses = compute_cluster_assignments_and_losses(params, new_batch_x, pool_update_key)
                new_batch_indices = jnp.arange(start_idx, end_idx, dtype=jnp.int32)
                
                # Update pools (this automatically computes current pool losses on-demand)
                pool_state = update_pool_vectorized(pool_state, new_batch_indices, losses, cluster_assignments, params, X_train, pool_update_key)
                
                logging.info(f"Updated pools with batch {start_idx}-{end_idx}")
                logging.info(f"Pool sizes: {pool_state.counts}")
        
        # TRAINING PHASE: Train on full pools (now with fixed-size batches)
        batch_key = next(key_gen).get()
        max_train_batch = 1024
        batch_x, batch_y, k, mask = get_full_pool_training_batch(pool_state, params, X_train, y_train, batch_key, max_train_batch)
        
        # Check if we have any valid samples
        if not jnp.any(mask):
            logging.warning(f"No samples available for training at iteration {iteration}")
            continue
        
        # Training step with VAE loss and masking
        train_key = next(key_gen).get()
        params, opt_state, loss_value, grads = train_step(params, opt_state, batch_x, k, train_key, mask)
        
        # Compute accuracy for this batch (only for valid samples)
        correct_predictions = jnp.where(mask, k == batch_y, False)
        valid_count = jnp.sum(mask)
        batch_accuracy = jnp.where(valid_count > 0, jnp.sum(correct_predictions) / valid_count, 0.0)
        
        epoch_loss += loss_value
        epoch_accuracy += batch_accuracy
        
        train_losses.append(loss_value)
        train_accuracies.append(batch_accuracy)
        
        # Collect animation frames at regular intervals
        frame_interval = max(1, training_iterations // frames_per_epoch)
        if iteration % frame_interval == 0 or iteration == training_iterations - 1:
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
                'batch': iteration,
                'total_batches': training_iterations
            }
            animation_frames.append(animation_frame)
            reconstruction_frames.append(np.array(example_reconstructions))
            
            logging.info(f"Animation frame collected - Iteration {iteration}/{training_iterations}, Loss: {loss_value:.4f}")
        
        if iteration % 50 == 0:
            # Calculate current cluster distribution for all training data
            current_key = next(key_gen).get()
            current_probabilities = get_probabilities_batch(params, X_train, current_key)
            current_assignments = np.argmax(current_probabilities, axis=1)
            current_cluster_counts = np.bincount(current_assignments, minlength=K)
            non_empty = np.sum(current_cluster_counts > 0)
            
            # Show pool sizes
            pool_sizes = pool_state.counts
            
            logging.info(f"Iteration {iteration}/{training_iterations}, Loss: {loss_value:.4f}, Accuracy: {batch_accuracy:.4f}")
            logging.info(f"  Cluster distribution: {current_cluster_counts}")
            logging.info(f"  Non-empty clusters: {non_empty}/{K} ({non_empty/K*100:.1f}%)")
            logging.info(f"  Pool sizes: {pool_sizes}")
            logging.info(f"  Training batch size: {jnp.sum(mask)}/{batch_x.shape[0]} (valid/total)")
    
    # Safety check for division by zero
    if training_iterations > 0:
        avg_loss = epoch_loss / training_iterations
        avg_accuracy = epoch_accuracy / training_iterations
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
    
    # Show final pool sizes
    final_pool_sizes = pool_state.counts
    logging.info(f"Final pool sizes: {final_pool_sizes}")
    
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
    
    # Generate and visualize VAE samples
    logging.info("Generating VAE samples from each cluster...")
    vae_sample_key = next(key_gen).get()
    generated_samples = sample_vae_cluster_examples(
        params=params,
        n_samples_per_cluster=5,
        K=K,
        key=vae_sample_key,
        latent_dim=config["latent_dim"]
    )
    
    vae_grid_file = plot_vae_generated_samples_grid(
        generated_samples=generated_samples,
        run_uid=run_uid,
        K=K,
        n_samples_per_cluster=5,
        img_dims=config["img_dims"]
    )
    
    logging.info(f"VAE generated samples grid saved: {vae_grid_file}")
    logging.info(f"All visualizations created and saved to: {all_saved_files}")
    
    logging.info("VAE clustering training with replay completed! Check the saved plots and animation for results.")