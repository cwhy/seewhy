from datasets import Dataset, DatasetDict, load_dataset
from typing import Literal, NamedTuple
import jax.numpy as jnp
from jax import Array
import pickle
import os
from pathlib import Path


class Supervised1D(NamedTuple):
    n_samples: int
    d_x: int
    d_y: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array


class ImageClassification(NamedTuple):
    n_samples: int
    n_test_samples: int
    d_x: tuple[int, int]
    d_y: int
    n_channels: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array


cache_dir = str(Path.home() / ".cache/huggingface/datasets")


def _get_cache_path(
    data: str, n_tr: int | None, n_tst: int | None, kind: str | None = None
) -> str:
    """Generate cache file path for the dataset configuration."""
    cache_base = Path(cache_dir).parent / "processed_datasets"
    cache_base.mkdir(exist_ok=True)

    # Create a unique filename based on dataset and parameters
    n_tr_str = f"n_tr_{n_tr}" if n_tr else "n_tr_all"
    n_tst_str = f"n_tst_{n_tst}" if n_tst else "n_tst_all"
    kind_suffix = f"_{kind}" if kind else ""
    filename = f"{data}{kind_suffix}_{n_tr_str}_{n_tst_str}.pkl"
    return str(cache_base / filename)


def _load_from_cache(cache_path: str):
    """Load a cached dataset object if it exists."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError):
        # If cache is corrupted or doesn't exist, return None
        pass
    return None


def _save_to_cache(cache_path: str, dataset) -> None:
    """Save a dataset object to cache."""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)
    except Exception as e:
        # Don't raise if caching fails - keep main functionality working
        print(f"Warning: Failed to cache dataset to {cache_path}: {e}")


def load_supervised_1d(
    data: Literal["mnist", "fashion_mnist"],
    n_tr: int | None = None,
    n_tst: int | None = None,
) -> Supervised1D:

    # Try to load from cache first (use kind to avoid collision with image datasets)
    cache_path = _get_cache_path(data, n_tr, n_tst, kind="1d")
    cached_dataset = _load_from_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset

    if data == "mnist":
        ds = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")
    elif data == "fashion_mnist":
        ds = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")
    else:  # pragma: no cover - guarded by Literal but kept for safety
        raise ValueError(
            "Unsupported 1D dataset "
            f"'{data}'. Supported values are: 'mnist', 'fashion_mnist'."
        )

    assert isinstance(ds, DatasetDict)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Access the data as JAX arrays via datasets' JAX formatting
    X_img: Array = jnp.array(train_ds["image"][:])
    y: Array = jnp.array(train_ds["label"][:])
    X_img_test: Array = jnp.array(test_ds["image"][:])
    y_test: Array = jnp.array(test_ds["label"][:])

    # Apply slicing after accessing the data
    if n_tr:
        X_img = X_img[:n_tr]
        y = y[:n_tr]
    if n_tst:
        X_img_test = X_img_test[:n_tst]
        y_test = y_test[:n_tst]

    n_samples = X_img.shape[0]
    X = X_img.reshape((n_samples, -1))
    n_test_samples = X_img_test.shape[0]
    X_test = X_img_test.reshape((n_test_samples, -1))
    n_samples, d_x = X.shape
    d_y = len(set(y.tolist()))
    dataset = Supervised1D(n_samples, d_x, d_y, X, y, X_test, y_test)
    _save_to_cache(cache_path, dataset)
    return dataset


def load_supervised_image(
    data: Literal["mnist", "fashion_mnist", "cifar10"],
    n_tr: int | None = None,
    n_tst: int | None = None,
) -> ImageClassification:

    # Try to load from cache first
    cache_path = _get_cache_path(data, n_tr, n_tst)
    cached_dataset = _load_from_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset

    # If not in cache, load and process the dataset
    if data == "mnist":
        ds = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")

        assert isinstance(ds, DatasetDict)
        train_ds = ds["train"]
        test_ds = ds["test"]

        # Access the data as JAX arrays
        X_img = jnp.array(train_ds["image"][:])
        y = jnp.array(train_ds["label"][:])
        X_img_test = jnp.array(test_ds["image"][:])
        y_test = jnp.array(test_ds["label"][:])

        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]

        n_samples = X_img.shape[0]
        n_test_samples = X_img_test.shape[0]
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(
            n_samples,
            n_test_samples,
            d_x,
            d_y,
            n_channels,
            X_train,
            y,
            X_test,
            y_test,
        )
        _save_to_cache(cache_path, dataset)
        return dataset
    elif data == "fashion_mnist":
        ds = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")

        assert isinstance(ds, DatasetDict)
        train_ds = ds["train"]
        test_ds = ds["test"]

        # Access the data as JAX arrays
        X_img= jnp.array(train_ds["image"][:])
        y = jnp.array(train_ds["label"][:])
        X_img_test = jnp.array(test_ds["image"][:])
        y_test = jnp.array(test_ds["label"][:])

        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]

        n_samples = X_img.shape[0]
        n_test_samples = X_img_test.shape[0]
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(
            n_samples,
            n_test_samples,
            d_x,
            d_y,
            n_channels,
            X_train,
            y,
            X_test,
            y_test,
        )
        _save_to_cache(cache_path, dataset)
        return dataset
    elif data == "cifar10":
        ds = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir).with_format("jax")

        assert isinstance(ds, DatasetDict)
        train_ds = ds["train"]
        test_ds = ds["test"]

        # Access the data as JAX arrays
        X_img: Array = jnp.array(train_ds["image"][:])
        y: Array = jnp.array(train_ds["label"][:])
        X_img_test: Array = jnp.array(test_ds["image"][:])
        y_test: Array = jnp.array(test_ds["label"][:])

        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]

        n_samples = X_img.shape[0]
        n_test_samples = X_img_test.shape[0]
        X_train = X_img.transpose((0, 3, 1, 2))
        X_test = X_img_test.transpose((0, 3, 1, 2))
        n_channels = 3
        d_x = (32, 32)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(
            n_samples,
            n_test_samples,
            d_x,
            d_y,
            n_channels,
            X_train,
            y,
            X_test,
            y_test,
        )
        _save_to_cache(cache_path, dataset)
        return dataset
    else:
        raise ValueError(
            "Unsupported image dataset "
            f"'{data}'. Supported values are: 'mnist', 'fashion_mnist', 'cifar10'."
        )
