#!/usr/bin/env python3
"""
Quick manual tests for shared_lib.datasets.

Includes:
- Synthetic test dataset generators (test-only helpers)
- Sanity checks for load_supervised_1d / load_supervised_image
"""

import sys
from pathlib import Path

import jax.numpy as jnp

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared_lib.datasets import (  # noqa: E402
    Supervised1D,
    ImageClassification,
    load_supervised_1d,
    load_supervised_image,
)


def make_test_supervised_1d(
    n_samples: int = 8,
    d_x: int = 4,
    n_classes: int = 3,
    n_test_samples: int | None = None,
) -> Supervised1D:
    """Create a small synthetic Supervised1D dataset (test-only)."""
    if n_test_samples is None:
        n_test_samples = max(1, n_samples // 2)

    X = jnp.arange(n_samples * d_x, dtype=jnp.float32).reshape(n_samples, d_x)
    y = jnp.arange(n_samples, dtype=jnp.int32) % n_classes

    X_test = jnp.arange(n_test_samples * d_x, dtype=jnp.float32).reshape(
        n_test_samples, d_x
    )
    y_test = jnp.arange(n_test_samples, dtype=jnp.int32) % n_classes

    return Supervised1D(
        n_samples=n_samples,
        d_x=d_x,
        d_y=n_classes,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
    )


def make_test_image_classification(
    n_samples: int = 10,
    n_test_samples: int | None = None,
    image_size: tuple[int, int] = (8, 8),
    n_channels: int = 1,
    n_classes: int = 3,
) -> ImageClassification:
    """Create a small synthetic ImageClassification dataset (test-only)."""
    if n_test_samples is None:
        n_test_samples = max(1, n_samples // 2)

    height, width = image_size

    X = jnp.arange(
        n_samples * n_channels * height * width, dtype=jnp.float32
    ).reshape(n_samples, n_channels, height, width)
    y = jnp.arange(n_samples, dtype=jnp.int32) % n_classes

    X_test = jnp.arange(
        n_test_samples * n_channels * height * width, dtype=jnp.float32
    ).reshape(n_test_samples, n_channels, height, width)
    y_test = jnp.arange(n_test_samples, dtype=jnp.int32) % n_classes

    return ImageClassification(
        n_samples=n_samples,
        n_test_samples=n_test_samples,
        d_x=image_size,
        d_y=n_classes,
        n_channels=n_channels,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
    )


def test_synthetic_supervised_1d():
    print("=== Testing synthetic make_test_supervised_1d ===")
    ds = make_test_supervised_1d()
    print(f"n_samples: {ds.n_samples}, d_x: {ds.d_x}, d_y: {ds.d_y}")
    print(f"X shape: {ds.X.shape}, y shape: {ds.y.shape}")
    print(f"X_test shape: {ds.X_test.shape}, y_test shape: {ds.y_test.shape}")


def test_synthetic_image_classification():
    print("=== Testing synthetic make_test_image_classification ===")
    ds = make_test_image_classification()
    print(
        f"n_samples: {ds.n_samples}, n_test_samples: {ds.n_test_samples}, "
        f"d_x: {ds.d_x}, d_y: {ds.d_y}, n_channels: {ds.n_channels}"
    )
    print(f"X shape: {ds.X.shape}, y shape: {ds.y.shape}")
    print(f"X_test shape: {ds.X_test.shape}, y_test shape: {ds.y_test.shape}")


def test_real_loaders():
    """Sanity-check the real dataset loaders (requires network/datasets)."""
    print("=== Testing load_supervised_1d('mnist') ===")
    ds_1d = load_supervised_1d("mnist")
    print(
        f"n_samples: {ds_1d.n_samples}, d_x: {ds_1d.d_x}, d_y: {ds_1d.d_y}, "
        f"X shape: {ds_1d.X.shape}, X_test shape: {ds_1d.X_test.shape}"
    )

    print("\n=== Testing load_supervised_image('mnist', n_tr=64, n_tst=16) ===")
    img_ds = load_supervised_image("mnist", n_tr=64, n_tst=16)
    print(
        f"n_samples: {img_ds.n_samples}, n_test_samples: {img_ds.n_test_samples}, "
        f"d_x: {img_ds.d_x}, d_y: {img_ds.d_y}, n_channels: {img_ds.n_channels}"
    )
    print(f"X shape: {img_ds.X.shape}, X_test shape: {img_ds.X_test.shape}")


def main():
    test_synthetic_supervised_1d()
    print()
    test_synthetic_image_classification()
    print()
    test_real_loaders()


if __name__ == "__main__":
    main()
