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

K = 10

def init() -> Tuple[Dict[str, Any], Dict[str, Array], Any]:
    """Initialize all model parameters in a single flat dictionary"""

    config = {
        "dataset_name": "mnist",
        "learning_rate": 1e-3,
        "random_seed": 1,
        "img_dims": (28, 28),  # Image dimensions
        "K": K,  # Number of clusters
        "n_samples": 20000,
        "n_hidden": 128,  # Hidden layer dimension
        "em_iterations": 20,  # Number of EM iterations
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    # For MNIST, we know the flattened input dimension is 28*28 = 784
    input_dim = 784
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

def get_probabilities(params: Dict[str, Array], x: Array) -> Array:
    """Get probability distribution across all K labels for a sample."""
    y_hat_logits = mlp(params, x)[:, 1]
    probabilities = jax.nn.softmax(y_hat_logits)
    return probabilities

def get_probabilities_batch(params: Dict[str, Array], x: Array) -> Array:
    """Get probability distributions across all K labels for a batch of samples."""
    return jax.jit(jax.vmap(get_probabilities, in_axes=(None, 0)))(params, x)

# MODIFIED LOSS FUNCTIONS
def loss_fn_symbol(params: Dict[str, Array], x: Array, d: Array) -> Array:
    """Cross-entropy loss function for a single sample."""
    logits = mlp_single_label(params, x)
    loss = -jnp.mean(d * jax.nn.log_softmax(logits))
    return loss

def loss_fn(params: Dict[str, Array], x: Array, ds: Array) -> Array:
    """Loss function for multiple samples."""
    return jnp.mean(jax.vmap(loss_fn_symbol, in_axes=(0, None, 0))(params, x, ds))

def loss_batch_modified(params: Dict[str, Array], xs: Array, selected_k: Array, K: int) -> Array:
    """Modified batch loss function - uses [0.5, 0.5] for unmatched clusters instead of 0."""
    batch_size = xs.shape[0]
    
    # Create target distributions - default to [0.5, 0.5] for all
    target_distributions = jnp.full((batch_size, K, 2), 0.5)
    
    # For each sample, if cluster x is assigned (selected_k[i] == x), set target to [0, 1]
    # Otherwise keep [0.5, 0.5]
    for i in range(batch_size):
        assigned_cluster = selected_k[i]
        # Set target distribution for the assigned cluster to [0, 1]
        target_distributions = target_distributions.at[i, assigned_cluster].set(jnp.array([0.0, 1.0]))
    
    return jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0))(params, xs, target_distributions))

@jax.jit
def train_step(params: Dict[str, Array], optimizer_state: Any, x: Array, selected_k: Array) -> Tuple[Dict[str, Array], Any, Array, Any]:
    """Modified training step using the new loss function."""
    loss, grads = jax.value_and_grad(loss_batch_modified)(params, x, selected_k, K)
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
    
    # EM Training loop
    logging.info("Starting EM training...")
    start_time = time.perf_counter()
    
    train_losses = []
    train_accuracies = []
    animation_frames = []
    
    em_iterations = config["em_iterations"]
    batch_size = 2048  # Use smaller batches for GPU memory
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    for em_iter in range(em_iterations):
        # E-step: Get cluster assignments for ALL data (batched)
        logging.info(f"E-step: Processing {num_batches} batches of size {batch_size}")
        all_selected_k = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = X_train[start_idx:end_idx]
            
            batch_probabilities = get_probabilities_batch(params, batch_x)
            batch_selected_k = jnp.argmax(batch_probabilities, axis=1)
            all_selected_k.append(batch_selected_k)
            
            if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
                logging.info(f"  E-step batch {batch_idx + 1}/{num_batches} completed")
        
        selected_k = jnp.concatenate(all_selected_k)
        
        # M-step: Train on X_examples only (much smaller for GPU)
        example_probabilities = get_probabilities_batch(params, X_examples)
        example_selected_k = jnp.argmax(example_probabilities, axis=1)
        params, opt_state, loss_value, grads = train_step(params, opt_state, X_examples, example_selected_k)
        
        # Compute accuracy for this iteration (using all data assignments vs true labels)
        accuracy = jnp.mean(selected_k == y_train)
        
        train_losses.append(loss_value)
        train_accuracies.append(accuracy)
        
        # Calculate cluster distribution for ALL training samples
        cluster_counts = np.bincount(selected_k, minlength=K)
        
        # Collect animation frames at regular intervals
        frames_per_em = min(50, em_iterations)  
        frame_interval = max(1, em_iterations // frames_per_em)
        if em_iter % frame_interval == 0 or em_iter == em_iterations - 1:
            # Get probability distributions for the selected examples
            example_probabilities = get_probabilities_batch(params, X_examples)
            
            # Create animation frame
            animation_frame = {
                'probabilities': np.array(example_probabilities),
                'cluster_counts': cluster_counts,
                'epoch': 0,
                'batch': em_iter,
                'total_batches': em_iterations
            }
            animation_frames.append(animation_frame)
            
            logging.info(f"Animation frame collected - EM Iteration {em_iter + 1}/{em_iterations}")
        
        if em_iter % 1 == 0 or em_iter == em_iterations - 1:
            logging.info(f"EM Iter {em_iter + 1}/{em_iterations}, Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}")
            logging.info(f"Cluster distribution: {cluster_counts}")
    
    logging.info(f"EM training completed")
    
    training_time = time.perf_counter() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Show final probability distributions
    final_probabilities = get_probabilities_batch(params, X_examples)
    
    # Calculate final cluster distribution for ALL training samples
    logging.info("Calculating final cluster distribution for all training samples...")
    all_training_probabilities = get_probabilities_batch(params, X_train)
    all_cluster_assignments = np.argmax(all_training_probabilities, axis=1)
    all_cluster_counts = np.bincount(all_cluster_assignments, minlength=10)
    
    logging.info(f"Final cluster distribution: {all_cluster_counts}")
    
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
        all_cluster_counts=all_cluster_counts
        # output_dir defaults to outputs/yy-mm-dd/
    )
    
    logging.info(f"All visualizations created and saved to: {saved_files}")
    
    logging.info("EM training completed! Check the saved plots and animation for results.")