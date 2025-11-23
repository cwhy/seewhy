"""
Softmax Regression on MNIST using JAX (functional style, no optax/flax).

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.

Examples:
- Pure: init_params(), forward(), loss_fn(), accuracy_fn()
- Methods with side effects: loss_monitor.on_epoch_end_() (mutates state, prints)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple, List, TypedDict, Callable, NamedTuple
import numpy as np
import logging
from functools import partial

from shared_lib.datasets import load_supervised_image
from shared_lib.random_utils import infinite_safe_keys
from loss_monitoring import LossMonitor
from sampler import BatchSampler, BatchMetadata
from trainer import Trainer

# Set up logging to output to console
logging.basicConfig(level=logging.INFO, format='%(message)s')


type MNISTData = Dict[str, Array]


def compute_weight_scale(input_dim: int) -> float:
    """Pure function to compute weight initialization scale (Xavier/Glorot)."""
    return float(jnp.sqrt(2.0 / input_dim))


def init_params(key_gen, input_dim: int, num_classes: int) -> Dict[str, Array]:
    """Pure function to initialize softmax regression parameters."""
    key = next(key_gen).get()
    scale = compute_weight_scale(input_dim)
    W = jax.random.normal(key, (input_dim, num_classes)) * scale
    b = jnp.zeros(num_classes)
    return {'W': W, 'b': b}


def forward(params: Dict[str, Array], X: Array) -> Array:
    """Forward pass: compute logits."""
    return jnp.dot(X, params['W']) + params['b']


def loss_fn(params: Dict[str, Array], data: Dict[str, Array]) -> Array:
    """Cross-entropy loss for softmax regression."""
    logits = forward(params, data['X'])
    # Numerically stable softmax cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # One-hot encode labels
    num_classes = logits.shape[-1]
    y_one_hot = jax.nn.one_hot(data['y'], num_classes)
    # Compute cross-entropy
    loss = -jnp.mean(jnp.sum(y_one_hot * log_probs, axis=-1))
    return loss


class ParamUpdate(NamedTuple):
    """Namedtuple with lr and loss_fn, plus train_batch method."""
    lr: float
    loss_fn: Callable[[Dict[str, Array], Dict[str, Array]], Array]
    
    @jax.jit(static_argnames=['self'])
    def train_batch(self, params: Dict[str, Array], batch: Dict[str, Array]) -> Tuple[Dict[str, Array], Array]:
        """Single training step with manual gradient descent."""
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        
        # Manual gradient descent update
        new_params = {
            'W': params['W'] - self.lr * grads['W'],
            'b': params['b'] - self.lr * grads['b']
        }
        
        return new_params, loss


def compute_predictions(params: Dict[str, Array], X: Array) -> Array:
    """Pure function to compute predictions from logits."""
    logits = forward(params, X)
    return jnp.argmax(logits, axis=-1)


def accuracy_fn(params: Dict[str, Array], X: Array, y: Array) -> Array:
    """Pure function to compute accuracy."""
    predictions = compute_predictions(params, X)
    return jnp.mean(predictions == y)


def normalize_pixels(X: Array) -> Array:
    """Pure function to normalize pixel values to [0, 1]."""
    return X.astype(jnp.float32) / 255.0


if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    dataset = load_supervised_image("mnist", n_tr=64)
    print(f"Training samples: {dataset.n_samples}, Test samples: {dataset.n_test_samples}")
    print(f"Input dimension: {dataset.d_x}, Number of classes: {dataset.d_y}")
    
    # Flatten images for softmax regression: (n_samples, n_channels, height, width) -> (n_samples, n_channels * height * width)
    n_samples, n_channels, height, width = dataset.X.shape
    input_dim = n_channels * height * width
    X_train_flat = dataset.X.reshape((n_samples, input_dim))
    
    n_test_samples, _, _, _ = dataset.X_test.shape
    X_test_flat = dataset.X_test.reshape((n_test_samples, input_dim))
    
    # Normalize pixel values to [0, 1]
    X_train = normalize_pixels(X_train_flat)
    X_test = normalize_pixels(X_test_flat)
    y_train = dataset.y
    y_test = dataset.y_test
    
    # Create data structures
    train_data: MNISTData = {'X': X_train, 'y': y_train}
    test_data: MNISTData = {'X': X_test, 'y': y_test}
    
    # Initialize parameters
    print("\nInitializing parameters...")
    key_gen = infinite_safe_keys(42)
    params = init_params(key_gen, input_dim, dataset.d_y)
    
    # Training hyperparameters
    batch_size = 4
    num_epochs = 1000
    learning_rate = 0.01
    
    # Initialize loss monitoring component
    loss_monitor = LossMonitor(
        log_interval=1,
        plot_title="Training Loss - Softmax Regression on MNIST",
        plot_filename="loss_plot.png",
        accuracy_plot_filename="accuracy_plot.png"
    )
    
    # Initialize batch sampler component
    # Assert that train_data has the correct structure (TypedDict is compatible with Dict[str, Array])
    assert isinstance(train_data, dict) and 'X' in train_data and 'y' in train_data
    sampler = BatchSampler(
        data=train_data,  
        n_samples=dataset.n_samples,
        batch_size=batch_size,
        num_epochs=num_epochs,
        key_gen=key_gen,
        shuffle=True
    )
    
    # Define callbacks for trainer
    def on_epoch_start_(epoch: int) -> None:
        """Callback for epoch start. Has side effects (mutates loss_monitor state)."""
        loss_monitor.start_epoch_tracking_(epoch)
    
    def on_epoch_end_(epoch: int, current_params: Dict[str, Array]) -> None:
        """Callback for epoch end. Has side effects (mutates loss_monitor state)."""
        # Compute accuracy
        train_acc = float(accuracy_fn(current_params, train_data['X'], train_data['y']))
        test_acc = float(accuracy_fn(current_params, test_data['X'], test_data['y']))
        loss_monitor.record_epoch_loss_(epoch, train_acc, test_acc, verbose=False)
        loss_monitor.log_loss_acc_(epoch, train_acc, test_acc)
    
    def on_batch_end_(metadata: BatchMetadata, loss: float) -> None:
        """Callback for batch end. Has side effects (mutates loss_monitor state)."""
        loss_monitor.record_batch_loss_(loss, metadata.batch_idx, verbose=False)
    
    
    # Initialize trainer component
    trainer = Trainer(
        sampler=sampler,
        param_update=ParamUpdate(lr=learning_rate, loss_fn=loss_fn),
        on_epoch_start=on_epoch_start_,
        on_epoch_end=on_epoch_end_,
        on_batch_end=on_batch_end_
    )
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    params = trainer.train_(params)
    
    # Finalize and plot
    loss_monitor.finalize_and_plot_()

