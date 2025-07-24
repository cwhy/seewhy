import time
import jax
import os, sys
import optax
import logging
from functools import partial
from jax import Array
import jax.numpy as jnp
from typing import Dict, Any, Iterable, Tuple

from lib.random_utils import infinite_safe_keys_from_key
from lib.datasets import load_supervised_image
from visualize import plot_training_progress, plot_sample_predictions, plot_section_accuracies, create_section_accuracies_animation_mp4
from online_eval import evaluate_model_performance, evaluate_training_sections_with_permutation, get_section_statistics

logging.basicConfig(level=logging.INFO)

def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    """Initialize all model parameters in a single flat dictionary"""

    config = {
        "dataset_name": "mnist",
        "num_epochs": 1,  # Only 1 epoch as requested
        "hidden_dim": 128,  # Hidden layer dimension
        "batch_size": 64,
        "learning_rate": 1e-3,
        "random_seed": 42,
        "img_dims": (28, 28),  # Image dimensions
    }
    key = jax.random.PRNGKey(config["random_seed"])
    hidden_dim = config["hidden_dim"]
    input_dim = 28 * 28  # Flattened MNIST images
    output_dim = 10  # 10 classes for MNIST
    key_gen = infinite_safe_keys_from_key(key)

    params = {
        'w1': jax.random.normal(next(key_gen).get(), (input_dim, hidden_dim)) * 0.01,
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(next(key_gen).get(), (hidden_dim, output_dim)) * 0.01,
        'b2': jnp.zeros(output_dim),
    }
    return config, params, key_gen

def mlp_forward(params: Dict[str, Array], x: Array) -> Array:
    """Forward pass through the 2-layer MLP"""
    # Flatten input if needed
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    
    # First layer
    h = jax.nn.relu(jnp.dot(x, params['w1']) + params['b1'])
    
    # Second layer
    logits = jnp.dot(h, params['w2']) + params['b2']
    
    return logits

def loss_fn(params: Dict[str, Array], x: Array, y: Array) -> Array:
    """Compute cross-entropy loss"""
    logits = mlp_forward(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)

def accuracy_fn(params: Dict[str, Array], x: Array, y: Array) -> Array:
    """Compute accuracy"""
    logits = mlp_forward(params, x)
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == y)

@partial(jax.jit, static_argnums=(0, ))
def train_step(optimizer: optax.GradientTransformation, params: Dict[str, Array], opt_state: Any, batch_x: Array, batch_y: Array) -> Tuple[optax.Params, optax.OptState, Array]:
    """Single training step"""
    loss_value, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value



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
    
    # Initialize optimizer
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    
    # Training loop
    logging.info("Starting training...")
    start_time = time.perf_counter()
    
    train_losses = []
    train_accuracies = []
    epoch_evaluation_results = []
    animation_frames = []
    
    num_batches = data.n_samples // config["batch_size"]
    frames_per_epoch = 30  # 30 frames per epoch as requested
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        # Shuffle data
        key = next(key_gen).get()
        permutation = jax.random.permutation(key, data.n_samples)
        shuffled_X = X_train[permutation]
        shuffled_y = y_train[permutation]
        
        # Store the permutation for this epoch to use in animation frames
        epoch_permutation = permutation
        
        for batch in range(num_batches):
            start_idx = batch * config["batch_size"]
            end_idx = start_idx + config["batch_size"]
            
            batch_x = shuffled_X[start_idx:end_idx]
            batch_y = shuffled_y[start_idx:end_idx]
            
            # Training step
            params, opt_state, loss_value = train_step(optimizer, params, opt_state, batch_x, batch_y)
            
            # Compute accuracy for this batch
            batch_accuracy = accuracy_fn(params, batch_x, batch_y)
            
            epoch_loss += loss_value
            epoch_accuracy += batch_accuracy
            
            train_losses.append(loss_value)
            train_accuracies.append(batch_accuracy)
            
            # Collect animation frames at regular intervals
            if batch % (num_batches // frames_per_epoch) == 0 or batch == num_batches - 1:
                # Evaluate on training sections for animation using the epoch's permutation
                section_accuracies = evaluate_training_sections_with_permutation(
                    params=params,
                    X_train=shuffled_X,  # Use the shuffled data from this epoch
                    y_train=shuffled_y,
                    forward_fn=mlp_forward,
                    num_sections=100,
                    max_sections_to_eval=100  # Evaluate ALL sections to see adaptation/forgetting
                )
                
                # Get statistics for this frame
                section_stats = get_section_statistics(section_accuracies)
                
                # Create animation frame
                animation_frame = {
                    'section_accuracies': section_accuracies,
                    'section_stats': section_stats,
                    'epoch': epoch,
                    'batch': batch,
                    'total_batches': num_batches
                }
                animation_frames.append(animation_frame)
                
                logging.info(f"Animation frame collected - Epoch {epoch}, Batch {batch}/{num_batches}, Mean accuracy: {section_stats['mean']:.4f}")
            
            if batch % 100 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch}/{num_batches}, Loss: {loss_value:.4f}, Accuracy: {batch_accuracy:.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        logging.info(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
        
        # Evaluate on training sections at the end of each epoch
        logging.info(f"Evaluating model on training sections after epoch {epoch}...")
        evaluation_key = next(key_gen).get()
        eval_results = evaluate_model_performance(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            forward_fn=mlp_forward,
            num_sections=100,
            key=evaluation_key
        )
        epoch_evaluation_results.append(eval_results)
        
        # Plot section accuracies for this epoch
        plot_section_accuracies(
            section_accuracies=eval_results['section_accuracies'],
            section_stats=eval_results['section_statistics'],
            save_path=f"mnist_section_accuracies_epoch_{epoch}.png",
            title=f"MNIST Training Section Accuracies - Epoch {epoch}"
        )
        
        logging.info(f"Epoch {epoch} evaluation completed - Test Accuracy: {eval_results['test_accuracy']:.4f}")
    
    training_time = time.perf_counter() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training progress
    plot_training_progress(train_losses, train_accuracies, "mnist_training_progress.png")
    
    # Plot sample predictions
    plot_sample_predictions(params, X_test, y_test, mlp_forward, num_samples=5, save_path="mnist_sample_predictions.png")
    
    # Create animation of training section accuracies
    if animation_frames:
        logging.info(f"Creating animation with {len(animation_frames)} frames...")
        create_section_accuracies_animation_mp4(
            animation_frames=animation_frames,
            save_path="mnist_section_accuracies_animation.mp4",
            title="MNIST Training Section Accuracies Animation",
            fps=3  # 3x slower (10/3 ≈ 3 fps)
        )
        logging.info("Animation created successfully!")
    
    # Print final evaluation summary
    if epoch_evaluation_results:
        final_eval = epoch_evaluation_results[-1]
        logging.info("Final Evaluation Summary:")
        logging.info(f"  Test Accuracy: {final_eval['test_accuracy']:.4f}")
        logging.info(f"  Training Sections - Mean: {final_eval['section_statistics']['mean']:.4f}")
        logging.info(f"  Training Sections - Std: {final_eval['section_statistics']['std']:.4f}")
    
    logging.info("Training completed! Check the saved plots and animation for results.") 