from datasets import load_dataset
from typing import NamedTuple
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

def _get_cache_path(data: str, n_tr: int | None, n_tst: int | None) -> str:
    """Generate cache file path for the dataset configuration."""
    cache_base = Path(cache_dir).parent / "processed_datasets"
    cache_base.mkdir(exist_ok=True)
    
    # Create a unique filename based on dataset and parameters
    n_tr_str = f"n_tr_{n_tr}" if n_tr else "n_tr_all"
    n_tst_str = f"n_tst_{n_tst}" if n_tst else "n_tst_all"
    filename = f"{data}_{n_tr_str}_{n_tst_str}.pkl"
    return str(cache_base / filename)

def _load_from_cache(cache_path: str) -> ImageClassification | None:
    """Load ImageClassification object from cache if it exists."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError):
        # If cache is corrupted or doesn't exist, return None
        pass
    return None

def _save_to_cache(cache_path: str, dataset: ImageClassification) -> None:
    """Save ImageClassification object to cache."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
    except Exception as e:
        # Silently fail if caching fails - don't break the main functionality
        print(f"Warning: Failed to cache dataset to {cache_path}: {e}")

def load_supervised_1d(data: str) -> Supervised1D:
    if data == "mnist":
        mnist = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")
        mnistData = mnist['train']
        # Access the data to get JAX arrays
        X_img = mnistData['image'][:]
        y = mnistData['label'][:]
        X_img_test = mnist["test"]["image"][:]
        y_test = mnist["test"]["label"][:]
        n_samples, _, _ = X_img.shape
        X = X_img.reshape((n_samples, -1))
        n_test_samples = X_img_test.shape[0]
        X_test = X_img_test.reshape((n_test_samples, -1))
        n_samples, d_x = X.shape
        d_y = len(set(y.tolist()))
        return Supervised1D(n_samples, d_x, d_y, X, y, X_test, y_test)
    elif data == "fashion_mnist":
        fashion_mnist = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")
        fashion_mnistData = fashion_mnist['train']
        # Access the data to get JAX arrays
        X_img = fashion_mnistData['image'][:]
        y = fashion_mnistData['label'][:]
        X_img_test = fashion_mnist["test"]["image"][:]
        y_test = fashion_mnist["test"]["label"][:]
        n_samples, _, _ = X_img.shape
        X = X_img.reshape((n_samples, -1))
        n_test_samples = X_img_test.shape[0]
        X_test = X_img_test.reshape((n_test_samples, -1))
        n_samples, d_x = X.shape
        d_y = len(set(y.tolist()))
        return Supervised1D(n_samples, d_x, d_y, X, y, X_test, y_test)

def load_supervised_image(data: str, n_tr: int | None = None, n_tst: int | None = None) -> ImageClassification:
    # Try to load from cache first
    cache_path = _get_cache_path(data, n_tr, n_tst)
    cached_dataset = _load_from_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset
    
    # If not in cache, load and process the dataset
    if data == "mnist":
        mnist = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")
        mnistData = mnist['train']
        # Access the data to get JAX arrays - need to access the full data first
        X_img = mnistData['image'][:]
        y = mnistData['label'][:]
        X_img_test = mnist["test"]["image"][:]
        y_test = mnist["test"]["label"][:]
        
        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]
            
        n_samples = len(X_img)
        n_test_samples = len(X_img_test)
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(n_samples, n_test_samples, d_x, d_y, n_channels, X_train, y, X_test, y_test)
        _save_to_cache(cache_path, dataset)
        return dataset
    elif data == "fashion_mnist":
        fashion_mnist = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")
        fashion_mnistData = fashion_mnist['train']
        # Access the data to get JAX arrays - need to access the full data first
        X_img = fashion_mnistData['image'][:]
        y = fashion_mnistData['label'][:]
        X_img_test = fashion_mnist["test"]["image"][:]
        y_test = fashion_mnist["test"]["label"][:]
        
        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]
            
        n_samples = len(X_img)
        n_test_samples = len(X_img_test)
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(n_samples, n_test_samples, d_x, d_y, n_channels, X_train, y, X_test, y_test)
        _save_to_cache(cache_path, dataset)
        return dataset
    elif data == "cifar10":
        cifar10 = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir).with_format("jax")
        cifar10Data = cifar10['train']
        # Access the data to get JAX arrays - need to access the full data first
        X_img = cifar10Data['image'][:]
        y = cifar10Data['label'][:]
        X_img_test = cifar10["test"]["image"][:]
        y_test = cifar10["test"]["label"][:]
        
        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]
            
        n_samples = len(X_img)
        n_test_samples = len(X_img_test)
        X_train = X_img.transpose((0, 3, 1, 2))
        X_test = X_img_test.transpose((0, 3, 1, 2))
        n_channels = 3
        d_x = (32, 32)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(n_samples, n_test_samples, d_x, d_y, n_channels, X_train, y, X_test, y_test)
        _save_to_cache(cache_path, dataset)
        return dataset