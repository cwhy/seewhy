import time
import jax
import os, sys
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
    create_enhanced_clustering_visualization
)

def generate_uid() -> str:
    """Generate a 4-character UID using alphanumeric characters"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))

# Generate UID for this run
run_uid = generate_uid()
print(f"Run UID: {run_uid}")

# Set up logging
logging.basicConfig(level=logging.INFO)

K = 10

def init() -> Tuple[Dict[str, Any], Array, Any]:
    """Initialize k-means centroids and config"""

    config = {
        "dataset_name": "mnist",
        "random_seed": 1,
        "img_dims": (28, 28),
        "K": K,
        "n_samples": 20000,
        "kmeans_iterations": 20,
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    # For MNIST, we know the flattened input dimension is 28*28 = 784
    input_dim = 784
    
    # Initialize centroids randomly
    centroids = jax.random.normal(next(key_gen).get(), (K, input_dim)) * 0.1
    
    logging.info(f"Centroids shape: {centroids.shape}")
    logging.info(f"Input dimension: {input_dim}")
    
    return config, centroids, key_gen

@jax.jit
def compute_distances(x: Array, centroids: Array) -> Array:
    """Compute squared Euclidean distances from point x to all centroids"""
    # x: (input_dim,), centroids: (K, input_dim)
    # Returns: (K,) array of distances
    return jnp.sum((x[None, :] - centroids) ** 2, axis=1)

@jax.jit
def assign_to_cluster(x: Array, centroids: Array) -> int:
    """Assign point x to closest centroid"""
    distances = compute_distances(x, centroids)
    return jnp.argmin(distances)

def assign_batch_to_clusters(X: Array, centroids: Array) -> Array:
    """Assign batch of points to closest centroids"""
    return jax.vmap(assign_to_cluster, in_axes=(0, None))(X, centroids)

def update_centroids(X: Array, assignments: Array) -> Array:
    """Update centroids based on cluster assignments"""
    new_centroids = []
    
    for k in range(K):
        # Find all points assigned to cluster k
        mask = assignments == k
        cluster_points = X[mask]
        
        # If cluster has no points, keep old centroid
        if jnp.sum(mask) == 0:
            # Use zero vector if no points (will be handled below)
            new_centroid = jnp.zeros(X.shape[1])
        else:
            # Compute mean of assigned points
            new_centroid = jnp.mean(cluster_points, axis=0)
        
        new_centroids.append(new_centroid)
    
    return jnp.stack(new_centroids)

def create_probability_like_distribution(assignments: Array, K: int) -> Array:
    """Convert hard assignments to probability-like distributions for visualization"""
    batch_size = assignments.shape[0]
    prob_distributions = jnp.zeros((batch_size, K))
    
    # Set probability to 1.0 for assigned cluster, 0 for others
    for i in range(batch_size):
        prob_distributions = prob_distributions.at[i, assignments[i]].set(1.0)
    
    return prob_distributions

def compute_inertia(X: Array, centroids: Array, assignments: Array) -> float:
    """Compute within-cluster sum of squares (inertia)"""
    total_inertia = 0.0
    
    for k in range(K):
        mask = assignments == k
        if jnp.sum(mask) > 0:
            cluster_points = X[mask]
            centroid = centroids[k]
            # Sum of squared distances to centroid
            inertia = jnp.sum((cluster_points - centroid[None, :]) ** 2)
            total_inertia += inertia
    
    return float(total_inertia)

@jax.jit
def compute_cluster_distances_vectorized(X: Array, centroids: Array, assignments: Array, k: int) -> Array:
    """JIT-compiled vectorized distance computation for cluster k"""
    mask = assignments == k
    cluster_points = X[mask]
    centroid = centroids[k]
    return jnp.sum((cluster_points - centroid[None, :]) ** 2, axis=1), mask

def get_closest_samples_to_centroids(X: Array, centroids: Array, assignments: Array, n_samples_per_cluster: int = 3) -> Tuple[Array, Array]:
    """
    Sample the closest points to each cluster center (optimized version).
    
    Args:
        X: Data points (n_samples, n_features)
        centroids: Cluster centers (K, n_features)
        assignments: Cluster assignments for each point (n_samples,)
        n_samples_per_cluster: Number of closest samples to return per cluster
        
    Returns:
        Tuple of (closest_samples, closest_indices) where:
        - closest_samples: (K, n_samples_per_cluster, n_features) array of closest samples
        - closest_indices: (K, n_samples_per_cluster) array of indices in original data
    """
    # Pre-allocate output arrays for better performance
    closest_samples = jnp.zeros((K, n_samples_per_cluster, X.shape[1]))
    closest_indices = jnp.full((K, n_samples_per_cluster), -1)
    
    # Convert to numpy for faster indexing operations
    X_np = np.array(X)
    centroids_np = np.array(centroids)
    assignments_np = np.array(assignments)
    
    for k in range(K):
        # Find all points assigned to cluster k using numpy for speed
        mask = assignments_np == k
        n_cluster_points = np.sum(mask)
        
        if n_cluster_points == 0:
            # No points in cluster - already initialized with zeros and -1
            continue
        
        # Get cluster points and indices
        cluster_points = X_np[mask]
        cluster_indices = np.where(mask)[0]
        
        # Compute distances efficiently
        centroid = centroids_np[k]
        distances = np.sum((cluster_points - centroid[None, :]) ** 2, axis=1)
        
        # Find closest points
        n_available = min(n_samples_per_cluster, len(cluster_points))
        closest_local_indices = np.argpartition(distances, n_available-1)[:n_available]
        
        # Sort the selected indices by distance for consistent ordering
        sorted_indices = closest_local_indices[np.argsort(distances[closest_local_indices])]
        
        # Get the closest samples and their original indices
        selected_samples = cluster_points[sorted_indices]
        selected_indices = cluster_indices[sorted_indices]
        
        # Fill the output arrays
        closest_samples = closest_samples.at[k, :n_available].set(selected_samples)
        closest_indices = closest_indices.at[k, :n_available].set(selected_indices)
        
        # Pad with duplicates if we don't have enough samples
        if n_available < n_samples_per_cluster:
            last_sample = selected_samples[-1]
            last_index = selected_indices[-1]
            for i in range(n_available, n_samples_per_cluster):
                closest_samples = closest_samples.at[k, i].set(last_sample)
                closest_indices = closest_indices.at[k, i].set(last_index)
    
    return closest_samples, closest_indices

if __name__ == "__main__":
    # Initialize centroids and config
    config, centroids, key_gen = init()
    
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
    
    # K-means Training loop
    logging.info("Starting K-means training...")
    start_time = time.perf_counter()
    
    train_losses = []  # Will store inertia values
    train_accuracies = []
    animation_frames = []
    closest_samples_frames = []
    closest_indices_frames = []
    
    kmeans_iterations = config["kmeans_iterations"]
    batch_size = 2048
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for kmeans_iter in range(kmeans_iterations):
        # Assignment step: Assign all points to closest centroids (batched)
        logging.info(f"Assignment step: Processing {num_batches} batches of size {batch_size}")
        all_assignments = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = X_train[start_idx:end_idx]
            
            batch_assignments = assign_batch_to_clusters(batch_x, centroids)
            all_assignments.append(batch_assignments)
            
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                logging.info(f"  Assignment batch {batch_idx + 1}/{num_batches} completed")
        
        assignments = jnp.concatenate(all_assignments)
        
        # Update step: Recompute centroids
        logging.info("Update step: Recomputing centroids...")
        centroids = update_centroids(X_train, assignments)
        
        # Compute inertia (within-cluster sum of squares)
        inertia = compute_inertia(X_train, centroids, assignments)
        
        # Compute accuracy (best permutation matching)
        accuracy = jnp.mean(assignments == y_train)  # Simple accuracy (not optimal)
        
        train_losses.append(inertia)
        train_accuracies.append(accuracy)
        
        # Calculate cluster distribution
        cluster_counts = np.bincount(assignments, minlength=K)
        
        # Collect animation frames at regular intervals
        frames_per_kmeans = min(100, kmeans_iterations)
        frame_interval = max(1, kmeans_iterations // frames_per_kmeans)
        if kmeans_iter % frame_interval == 0 or kmeans_iter == kmeans_iterations - 1:
            # Get assignments for examples and convert to probability-like format
            example_assignments = assign_batch_to_clusters(X_examples, centroids)
            example_probabilities = create_probability_like_distribution(example_assignments, K)
            
            # Create animation frame
            animation_frame = {
                'probabilities': np.array(example_probabilities),
                'cluster_counts': cluster_counts,
                'epoch': 0,
                'batch': kmeans_iter,
                'total_batches': kmeans_iterations
            }
            animation_frames.append(animation_frame)
            
            # Collect closest samples to centers for animation
            closest_samples, closest_indices = get_closest_samples_to_centroids(X_train, centroids, assignments, n_samples_per_cluster=3)
            closest_samples_frames.append(np.array(closest_samples))
            closest_indices_frames.append(np.array(closest_indices))
            
            logging.info(f"Animation frame collected - K-means Iteration {kmeans_iter + 1}/{kmeans_iterations}")
        
        if kmeans_iter % 5 == 0 or kmeans_iter == kmeans_iterations - 1:
            logging.info(f"K-means Iter {kmeans_iter + 1}/{kmeans_iterations}, Inertia: {inertia:.2f}, Accuracy: {accuracy:.4f}")
            logging.info(f"Cluster distribution: {cluster_counts}")
    
    logging.info("K-means training completed")
    
    training_time = time.perf_counter() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Final assignments and probabilities for visualization
    final_assignments = assign_batch_to_clusters(X_examples, centroids)
    final_probabilities = create_probability_like_distribution(final_assignments, K)
    
    # Calculate final cluster distribution
    logging.info("Calculating final cluster distribution for all training samples...")
    all_final_assignments = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_x = X_train[start_idx:end_idx]
        batch_assignments = assign_batch_to_clusters(batch_x, centroids)
        all_final_assignments.append(batch_assignments)
    
    all_cluster_assignments = jnp.concatenate(all_final_assignments)
    all_cluster_counts = np.bincount(all_cluster_assignments, minlength=K)
    
    logging.info(f"Final cluster distribution: {all_cluster_counts}")
    
    # Compute final closest samples to cluster centers
    logging.info("Computing final closest samples to cluster centers...")
    final_closest_samples, final_closest_indices = get_closest_samples_to_centroids(X_train, centroids, all_cluster_assignments, n_samples_per_cluster=3)
    
    # Create all visualizations using the unified function
    logging.info("Creating all clustering visualizations...")
    saved_files = create_enhanced_clustering_visualization(
        animation_frames=animation_frames,
        final_probabilities=final_probabilities,
        X_examples=X_examples,
        y_examples=y_examples,
        train_losses=train_losses,  # Using inertia values
        train_accuracies=train_accuracies,
        run_uid=run_uid,
        all_cluster_counts=all_cluster_counts,
        closest_samples_frames=closest_samples_frames,
        closest_indices_frames=closest_indices_frames,
        y_train=y_train,
        final_closest_samples=final_closest_samples,
        final_closest_indices=final_closest_indices
    )
    
    logging.info(f"All visualizations created and saved to: {saved_files}")
    logging.info("K-means clustering completed! Check the saved plots and animation for results.")