"""
Matrix decomposition-style reconstruction on MNIST using JAX.

We model each image using:
- A sample embedding (computed from the input)
- A position embedding (learned per pixel position, 1D index)

The reconstruction is obtained by multiplying sample and position embeddings,
then applying a sigmoid so outputs live in [0, 1] to match MNIST pixels.

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.

Examples:
- Pure: init_params(), forward(), loss_fn()
- Methods with side effects: loss_monitor.finalize_and_plot_() (mutates state, saves plots)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple, List, Callable, Any, NamedTuple, cast
import numpy as np
import logging
from functools import partial
import matplotlib.pyplot as plt
import optax

from shared_lib.datasets import load_supervised_1d
from shared_lib.random_utils import infinite_safe_keys
from shared_lib.media import save_matplotlib_figure
from loss_monitoring import LossMonitor
from sampler import BatchSampler, BatchMetadata
from configs import ParamUpdateResult

# Set up logging to output to console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


type MNISTData = Dict[str, Array]


def compute_weight_scale(input_dim: int) -> float:
    """Pure function to compute weight initialization scale (Xavier/Glorot)."""
    return float(jnp.sqrt(2.0 / input_dim))


def fsq_quantize(x: Array, levels: int = 16, clip_value: float = 3.0) -> Array:
    """
    Finite scalar-style quantization layer with straight-through gradients.

    Args:
        x: Input array to quantize.
        levels: Number of discrete quantization levels per dimension.
        clip_value: Symmetric clipping range for inputs before quantization.

    Returns:
        Quantized array (same shape as x) with straight-through estimator for gradients.
    """
    # Clip to a symmetric range to avoid extreme values dominating quantization
    x_clipped = jnp.clip(x, -clip_value, clip_value)

    # Normalize to [-1, 1]
    x_norm = x_clipped / clip_value

    # Map to [0, levels - 1], round, then map back to [-1, 1]
    scale = (levels - 1) / 2.0
    x_scaled = x_norm * scale + scale
    x_rounded = jnp.round(x_scaled)
    x_q_norm = (x_rounded - scale) / scale

    # De-normalize back to original scale
    x_q = x_q_norm * clip_value

    # Straight-through estimator: forward uses quantized values, gradient is identity
    return x + jax.lax.stop_gradient(x_q - x)


def init_params(key_gen, input_dim: int, embedding_dim: int, bottleneck_dim: int) -> Dict[str, Array]:
    """
    Initialize parameters for a simple matrix-decomposition-style model.

    - sample_hidden_matrix: first linear layer, maps flattened image -> hidden (d_x, d_emb)
    - sample_embedding_matrix: second linear layer, maps quantized hidden -> sample embedding (d_emb, d_emb)
    - position_embeddings: per-position embedding (d_x, d_emb), using 1D pixel index
    """
    scale_input = compute_weight_scale(input_dim)
    scale_bottleneck = compute_weight_scale(bottleneck_dim)
    key_hidden = next(key_gen).get()
    key_sample = next(key_gen).get()
    key_pos = next(key_gen).get()

    sample_hidden_matrix = (
        jax.random.normal(key_hidden, (input_dim, bottleneck_dim)) * scale_input
    )
    sample_embedding_matrix = (
        jax.random.normal(key_sample, (bottleneck_dim, embedding_dim)) * scale_bottleneck
    )
    position_embeddings = (
        jax.random.normal(key_pos, (input_dim, embedding_dim)) * scale_input
    )

    bias = jnp.zeros((input_dim,))

    return {
        "sample_hidden_matrix": sample_hidden_matrix,
        "sample_embedding_matrix": sample_embedding_matrix,
        "position_embeddings": position_embeddings,
        "bias": bias,
    }


def forward(params: Dict[str, Array], X: Array) -> Array:
    """
    Forward pass: reconstruct images from sample and position embeddings.

    Args:
        params: Parameter dict with sample/position embeddings
        X: Flattened, normalized MNIST images, shape (batch, d_x)

    Returns:
        Reconstructed images in [0, 1], shape (batch, d_x)
    """
    # First layer: project input into a hidden representation
    # shape: (batch, d_x) @ (d_x, d_emb) -> (batch, d_emb)
    hidden = jnp.dot(X, params["sample_hidden_matrix"])

    # Quantize the hidden layer using FSQ
    # hidden_q = fsq_quantize(hidden)
    hidden_q = hidden

    # Second layer: map quantized hidden to final sample embedding
    # shape: (batch, d_emb) @ (d_emb, d_emb) -> (batch, d_emb)
    sample_emb = jnp.dot(hidden_q, params["sample_embedding_matrix"])

    # Position embeddings: learned per 1D pixel position
    # shape: (batch, d_emb) @ (d_emb, d_x) -> (batch, d_x)
    logits = jnp.dot(sample_emb, params["position_embeddings"].T)
    logits += params["bias"]

    # Map to [0, 1] to match normalized MNIST pixels
    return jax.nn.sigmoid(logits)


def loss_fn(params: Dict[str, Array], data: Dict[str, Array]) -> Array:
    """
    Reconstruction loss: mean squared error between input and reconstruction.

    Args:
        params: Model parameters
        data: Dict with key 'X' containing flattened, normalized images
    """
    X = data["X"]
    X_hat = forward(params, X)
    # return jnp.mean(jnp.square(X_hat - X))
    return jnp.mean(jnp.square(X_hat - X))


class ParamUpdate(NamedTuple):
    """
    NamedTuple-style parameter update object, using AdamW internally.

    This keeps the original structure (lr + loss_fn + batch_update method),
    but delegates the actual update to an optax.adamw optimizer.
    """
    lr: float
    optimizer: optax.GradientTransformation
    loss_fn: Callable[[Dict[str, Array], Dict[str, Array]], Array]

    @jax.jit(static_argnames=['self'])
    def batch_update(
        self,
        params: Dict[str, Array],
        opt_state: Any,
        batch: Dict[str, Array],
    ) -> Tuple[ParamUpdateResult, Any]:
        """Single training step using AdamW."""
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = cast(Dict[str, Array], new_params)
        result = ParamUpdateResult(params=new_params, outputs={'loss': loss})
        return result, new_opt_state


def normalize_pixels(X: Array) -> Array:
    """Pure function to normalize pixel values to [0, 1]."""
    return X.astype(jnp.float32) / 255.0


def plot_sample_reconstructions(
    params: Dict[str, Array],
    X: Array,
    forward_fn: Callable[[Dict[str, Array], Array], Array],
    num_samples: int = 4,
    img_dims: Tuple[int, int] = (28, 28),
    save_path: str = "mnist_sample_reconstructions.png",
) -> None:
    """
    Plot a few original vs reconstructed MNIST images and save the figure.
    """
    num_samples = int(min(num_samples, X.shape[0]))
    sample_x = X[:num_samples]
    recon_x = forward_fn(params, sample_x)

    # Reshape to image grids
    sample_x_imgs = sample_x.reshape((num_samples, img_dims[0], img_dims[1]))
    recon_x_imgs = recon_x.reshape((num_samples, img_dims[0], img_dims[1]))

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    if num_samples == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(sample_x_imgs[i], cmap="gray")
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis("off")

        # Reconstruction
        axes[1, i].imshow(recon_x_imgs[i], cmap="gray")
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis("off")

    fig.suptitle("MNIST Sample Reconstructions", fontsize=14)
    plt.tight_layout()

    save_matplotlib_figure(save_path, fig, format="png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    # dataset = load_supervised_1d("mnist", n_tr=8)
    dataset = load_supervised_1d("mnist")
    logger.info(f"Original training samples: {dataset.n_samples}, Test samples: {dataset.X_test.shape[0]}")
    logger.info(f"Input dimension: {dataset.d_x}")

    # Remove all digit-1 samples from the training set
    train_mask = (dataset.y != 1) & (dataset.y != 0) & (dataset.y != 9)
    X_train_raw = dataset.X[train_mask]
    n_train = int(X_train_raw.shape[0])
    logger.info(f"Filtered training samples (excluding digit '1'): {n_train}")

    # Dataset is already flattened to (n_samples, d_x); just normalize
    input_dim = dataset.d_x
    X_train = normalize_pixels(X_train_raw)
    X_test = normalize_pixels(dataset.X_test)
    
    # Create data structures
    train_data: MNISTData = {'X': X_train}
    test_data: MNISTData = {'X': X_test}
    
    # Initialize parameters
    logger.info("\nInitializing parameters for matrix decomposition model...")
    key_gen = infinite_safe_keys(42)
    embedding_dim = 512
    bottleneck_dim = 64
    params = init_params(key_gen, input_dim, embedding_dim, bottleneck_dim)
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Training hyperparameters
    batch_size = 512
    num_epochs = 100 
    learning_rate = 0.001
    
    # Initialize loss monitoring component
    loss_monitor = LossMonitor(
        log_interval=1,
        plot_title="Training Loss - MNIST Matrix Decomposition",
        plot_filename="loss_plot.png",
        accuracy_plot_filename="accuracy_plot.png",
        track_accuracy=False,
        track_test_loss=True,
    )
    
    # Initialize batch sampler component
    assert isinstance(train_data, dict) and 'X' in train_data
    sampler = iter(BatchSampler(
        data=train_data,  
        n_samples=n_train,
        batch_size=batch_size,
        num_epochs=num_epochs,
        key_gen=key_gen,
        shuffle=True
    ))
    
    # Initialize AdamW optimizer from optax and wrap in ParamUpdate
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    param_update = ParamUpdate(lr=learning_rate, optimizer=optimizer, loss_fn=loss_fn)
    
    # Training loop
    logger.info(f"\nTraining for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    # for batch, metadata in sampler:
    while True:
        try:
            batch, metadata = next(sampler)
        except StopIteration:
            break
        # Handle logic before batch
        loss_monitor.before_update_(params, metadata)
        
        # Training step
        result, opt_state = param_update.batch_update(params, opt_state, batch)
        params = result.params
        
        # Handle logic after batch
        loss_monitor.after_update_(
            params=params,
            result=result,
            metadata=metadata,
            train_data=train_data,
            test_data=test_data,
            accuracy_fn=None,
            loss_fn=loss_fn,
        )
    
    # Finalize and plot
    loss_monitor.finalize_and_plot_()

    # Plot sample reconstructions from training and test sets
    logger.info("\nPlotting sample reconstructions from training set...")
    plot_sample_reconstructions(
        params=params,
        X=X_train,
        forward_fn=forward,
        num_samples=6,
        img_dims=(28, 28),
        save_path="mnist_train_reconstructions.png",
    )

    logger.info("\nPlotting sample reconstructions from test set...")
    plot_sample_reconstructions(
        params=params,
        X=X_test,
        forward_fn=forward,
        num_samples=6,
        img_dims=(28, 28),
        save_path="mnist_test_reconstructions.png",
    )
