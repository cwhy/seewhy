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
    create_clustering_visualization
)

def generate_uid() -> str:
    """Generate a 4-character UID using alphanumeric characters"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))

# Generate UID for this run
run_uid = generate_uid()
print(f"Run UID: {run_uid}")

# Set up logging (simple like mnist-base.py)
logging.basicConfig(level=logging.INFO)

def init() -> Tuple[Dict[str, Any], Dict[str, Array], Any]:
    """Initialize all model parameters in a single flat dictionary"""

    config = {
        "dataset_name": "mnist",
        "batch_size": 256,
        "learning_rate": 1e-3,
        "random_seed": 2,
        "img_dims": (28, 28),  # Image dimensions
        "K": 10,  # Number of clusters
        "n_samples": 20000,
        "n_hidden": 128,  # Hidden layer dimension
        "entropy_threshold": 0.5,  # Minimum entropy threshold - clusters below this get reinitialized
        "entropy_check_interval": 50,  # Check entropy every N batches
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    # For MNIST, we know the flattened input dimension is 28*28 = 784
    input_dim = 784
    K = config["K"]
    n_hidden = config["n_hidden"]
    
    # Initialize parameters using Kaiming He initialization - KEEPING YOUR WORKING LOGIC
    params = {
        'l1_w': jax.random.normal(next(key_gen).get(), (K, input_dim, n_hidden)) * jnp.sqrt(2./input_dim),
        'l1_b': jnp.zeros((K, n_hidden)),
        'l2_w': jax.random.normal(next(key_gen).get(), (K, n_hidden, 2)) * jnp.sqrt(2./n_hidden),
        'l2_b': jnp.zeros((K, 2))
    }
    
    # Debug: print parameter shapes
    logging.info(f"Parameter shapes:")
    logging.info(f"  l1_w: {params['l1_w'].shape}")
    logging.info(f"  l1_b: {params['l1_b'].shape}")
    logging.info(f"  l2_w: {params['l2_w'].shape}")
    logging.info(f"  l2_b: {params['l2_b'].shape}")
    logging.info(f"  input_dim: {input_dim}")
    
    return config, params, key_gen

# KEEPING YOUR WORKING MODEL FUNCTIONS
def mlp_single_label(params: Dict[str, Array], x: Array) -> Array:
    """Simple MLP forward pass for a single sample using a flat dictionary for parameters."""
    h1 = jnp.dot(x, params['l1_w']) + params['l1_b']
    h1 = jax.nn.relu(h1)
    logits = jnp.dot(h1, params['l2_w']) + params['l2_b']
    return logits

def mlp(params: Dict[str, Array], x: Array) -> Array:
    """Vectorized MLP forward pass for multiple samples."""
    return jax.vmap(mlp_single_label, in_axes=(0, None))(params, x)

def mlp_batch(params: Dict[str, Array], x: Array) -> Array:
    """Vectorized MLP forward pass for batches."""
    return jax.vmap(mlp, in_axes=(None, 0))(params, x)

def get_K(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get cluster assignment for a single sample."""
    y_hat_logits = mlp(params, x)[:, 1]
    # return jax.random.categorical(key, y_hat_logits)
    return jnp.argmax(y_hat_logits)

def get_K_batch(params: Dict[str, Array], x: Array, key: Array) -> Array:
    """Get cluster assignments for a batch of samples."""
    return jax.jit(jax.vmap(get_K, in_axes=(None, 0, None)))(params, x, key)

def get_probabilities(params: Dict[str, Array], x: Array) -> Array:
    """Get probability distribution across all K labels for a sample."""
    y_hat_logits = mlp(params, x)[:, 1]
    probabilities = jax.nn.softmax(y_hat_logits)
    return probabilities

def get_probabilities_batch(params: Dict[str, Array], x: Array) -> Array:
    """Get probability distributions across all K labels for a batch of samples."""
    return jax.jit(jax.vmap(get_probabilities, in_axes=(None, 0)))(params, x)


# ENTROPY MEASUREMENT AND CLUSTER REINITIALIZATION FUNCTIONS
def compute_entropy(logits: Array) -> Array:
    """Compute entropy of softmax distribution."""
    probs = jax.nn.softmax(logits)
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    entropy = -jnp.sum(probs * jnp.log(probs + epsilon))
    return entropy

def measure_cluster_entropies(params: Dict[str, Array], X: Array, key: Array) -> Array:
    """Measure average entropy for each cluster across a sample of data."""
    K = params['l1_w'].shape[0]
    cluster_entropies = jnp.zeros(K)
    
    # Get cluster assignments and probabilities for all samples
    cluster_assignments = get_K_batch(params, X, key)
    all_logits = mlp_batch(params, X)
    
    # Calculate average entropy for each cluster
    for k in range(K):
        # Find samples assigned to this cluster
        cluster_mask = cluster_assignments == k
        if jnp.sum(cluster_mask) > 0:
            # Get logits for samples assigned to this cluster
            cluster_logits = all_logits[cluster_mask, k, :]
            # Compute entropy for each sample in this cluster
            entropies = jax.vmap(compute_entropy)(cluster_logits)
            # Average entropy for this cluster
            cluster_entropies = cluster_entropies.at[k].set(jnp.mean(entropies))
        else:
            # No samples assigned to this cluster, set entropy to 0 (will trigger reinitialization)
            cluster_entropies = cluster_entropies.at[k].set(0.0)
    
    return cluster_entropies

def reinitialize_cluster(params: Dict[str, Array], cluster_idx: int, key_gen: Any, input_dim: int, n_hidden: int) -> Dict[str, Array]:
    """Reinitialize weights for a specific cluster (MLP network)."""
    new_params = params.copy()
    
    # Reinitialize weights for the specified cluster using Kaiming He initialization
    new_params['l1_w'] = new_params['l1_w'].at[cluster_idx].set(
        jax.random.normal(next(key_gen).get(), (input_dim, n_hidden)) * jnp.sqrt(2./input_dim)
    )
    new_params['l1_b'] = new_params['l1_b'].at[cluster_idx].set(jnp.zeros(n_hidden))
    new_params['l2_w'] = new_params['l2_w'].at[cluster_idx].set(
        jax.random.normal(next(key_gen).get(), (n_hidden, 2)) * jnp.sqrt(2./n_hidden)
    )
    new_params['l2_b'] = new_params['l2_b'].at[cluster_idx].set(jnp.zeros(2))
    
    return new_params


# STANDARD LOSS FUNCTIONS (WITHOUT ENTROPY REGULARIZATION)
def loss_fn_symbol(params: Dict[str, Array], x: Array, d: Array) -> Array:
    """Cross-entropy loss function for a single sample."""
    logits = mlp_single_label(params, x)
    loss = -jnp.mean(d * jax.nn.log_softmax(logits))
    return loss

def loss_fn(params: Dict[str, Array], x: Array, ds: Array) -> Array:
    """Loss function for multiple samples."""
    return jnp.mean(jax.vmap(loss_fn_symbol, in_axes=(0, None, 0))(params, x, ds))

def loss_batch(params: Dict[str, Array], xs: Array, ys: Array) -> Array:
    """Batch loss function."""
    y_one_hot = jax.nn.one_hot(ys, num_classes=10)
    dss = jnp.stack([1 - y_one_hot, y_one_hot], axis=-1)
    return jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0))(params, xs, dss))

@jax.jit
def train_step(params: Dict[str, Array], optimizer_state: Any, x: Array, y: Array) -> Tuple[Dict[str, Array], Any, Array, Any]:
    """Performs a single training step."""
    loss, grads = jax.value_and_grad(loss_batch)(params, x, y)
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
    logging.info("Starting training...")
    start_time = time.perf_counter()
    
    train_losses = []
    train_accuracies = []
    animation_frames = []
    cluster_reinit_history = []  # Track which clusters were reinitialized when
    
    num_batches = num_samples // config["batch_size"]
    frames_per_epoch = min(200, num_batches)  # Collect more frames for longer animation
    
    # Safety check: ensure we have at least one batch
    if num_batches == 0:
        logging.warning(f"num_samples ({num_samples}) is smaller than batch_size ({config['batch_size']}). Setting num_batches to 1.")
        num_batches = 1
        frames_per_epoch = 1
    
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    # For entropy measurement, sample a subset of training data
    entropy_sample_size = min(1000, num_samples)
    entropy_sample_indices = np.random.choice(num_samples, entropy_sample_size, replace=False)
    X_entropy_sample = X_train[entropy_sample_indices]
    
    for batch in range(num_batches):
        start_idx = batch * config["batch_size"]
        end_idx = min(start_idx + config["batch_size"], num_samples)
        
        batch_x = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        # Get cluster assignments for this batch
        batch_key = next(key_gen).get()
        k = get_K_batch(params, batch_x, batch_key)
        
        # Training step (standard, no entropy regularization)
        params, opt_state, loss_value, grads = train_step(params, opt_state, batch_x, k)
        
        # Compute accuracy for this batch
        batch_accuracy = jnp.mean(k == batch_y)
        
        epoch_loss += loss_value
        epoch_accuracy += batch_accuracy
        
        train_losses.append(loss_value)
        train_accuracies.append(batch_accuracy)
        
        # Check entropy and reinitialize biased clusters periodically
        if batch % config["entropy_check_interval"] == 0 and batch > 0:
            entropy_key = next(key_gen).get()
            cluster_entropies = measure_cluster_entropies(params, X_entropy_sample, entropy_key)
            
            # Find clusters with entropy below threshold
            low_entropy_clusters = jnp.where(cluster_entropies < config["entropy_threshold"])[0]
            
            if len(low_entropy_clusters) > 0:
                logging.info(f"Batch {batch}: Found {len(low_entropy_clusters)} clusters with low entropy: {low_entropy_clusters}")
                logging.info(f"Cluster entropies: {cluster_entropies}")
                
                # Reinitialize low-entropy clusters
                for cluster_idx in low_entropy_clusters:
                    params = reinitialize_cluster(params, int(cluster_idx), key_gen, 784, config["n_hidden"])
                    logging.info(f"  Reinitialized cluster {cluster_idx}")
                
                # Reinitialize optimizer state for the reinitialized parameters
                opt_state = optimizer.init(params)
                
                # Record reinitialization event
                cluster_reinit_history.append({
                    'batch': batch,
                    'reinitialized_clusters': [int(c) for c in low_entropy_clusters],
                    'entropies': np.array(cluster_entropies)
                })
            else:
                logging.info(f"Batch {batch}: All clusters have healthy entropy. Min: {jnp.min(cluster_entropies):.4f}")
        
        # Collect animation frames at regular intervals
        frame_interval = max(1, num_batches // frames_per_epoch)
        if batch % frame_interval == 0 or batch == num_batches - 1:
            # Get probability distributions for the selected examples
            example_probabilities = get_probabilities_batch(params, X_examples)
            
            # Calculate cluster distribution for ALL training samples
            all_training_probabilities = get_probabilities_batch(params, X_train)
            all_cluster_assignments = np.argmax(all_training_probabilities, axis=1)
            all_cluster_counts = np.bincount(all_cluster_assignments, minlength=10)
            
            # Measure current cluster entropies for the frame
            frame_key = next(key_gen).get()
            current_entropies = measure_cluster_entropies(params, X_entropy_sample, frame_key)
            
            # Create animation frame
            animation_frame = {
                'probabilities': np.array(example_probabilities),
                'cluster_counts': all_cluster_counts,
                'cluster_entropies': np.array(current_entropies),
                'epoch': 0,
                'batch': batch,
                'total_batches': num_batches
            }
            animation_frames.append(animation_frame)
            
            logging.info(f"Animation frame collected - Batch {batch}/{num_batches}, Loss: {loss_value:.4f}")
        
        if batch % 100 == 0:
            logging.info(f"Batch {batch}/{num_batches}, Loss: {loss_value:.4f}, Accuracy: {batch_accuracy:.4f}")
    
    # Safety check for division by zero
    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
    else:
        avg_loss = epoch_loss
        avg_accuracy = epoch_accuracy
    
    logging.info(f"Training completed - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    logging.info(f"Total cluster reinitializations: {len(cluster_reinit_history)}")
    
    training_time = time.perf_counter() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Show final probability distributions
    final_probabilities = get_probabilities_batch(params, X_examples)
    
    # Calculate cluster distribution for ALL training samples
    logging.info("Calculating cluster distribution for all training samples...")
    all_training_probabilities = get_probabilities_batch(params, X_train)
    all_cluster_assignments = np.argmax(all_training_probabilities, axis=1)
    all_cluster_counts = np.bincount(all_cluster_assignments, minlength=10)
    
    # Final entropy measurement
    final_key = next(key_gen).get()
    final_entropies = measure_cluster_entropies(params, X_entropy_sample, final_key)
    
    logging.info(f"Cluster distribution: {all_cluster_counts}")
    logging.info(f"Final cluster entropies: {final_entropies}")
    
    # Create all visualizations using the unified function
    logging.info("Creating all clustering visualizations...")
    saved_files = create_clustering_visualization(
        animation_frames=animation_frames,
        final_probabilities=final_probabilities,
        X_examples=X_examples,
        y_examples=y_examples,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        run_uid=run_uid,
        all_cluster_counts=all_cluster_counts,
        cluster_reinit_history=cluster_reinit_history
        # output_dir defaults to outputs/yy-mm-dd/
    )
    
    logging.info(f"All visualizations created and saved to: {saved_files}")
    
    logging.info("Training completed! Check the saved plots and animation for results.")