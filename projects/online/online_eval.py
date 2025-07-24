import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Callable
from jax import Array
import logging

def accuracy_fn(params: Dict[str, Array], x: Array, y: Array, forward_fn: Callable) -> Array:
    """Compute accuracy using the provided forward function"""
    logits = forward_fn(params, x)
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == y)

def evaluate_training_sections(
    params: Dict[str, Array],
    X_train: Array,
    y_train: Array,
    forward_fn: Callable,
    num_sections: int = 100,
    key: jax.random.PRNGKey = None
) -> Tuple[List[float], Array]:
    """
    Evaluate model accuracy on training data split into sections after shuffling.
    
    Args:
        params: Model parameters
        X_train: Training data features
        y_train: Training data labels
        forward_fn: Forward function for the model
        num_sections: Number of sections to split the data into
        key: Random key for shuffling (if None, will use a default key)
    
    Returns:
        Tuple of (section_accuracies, shuffled_indices)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Shuffle the data
    permutation = jax.random.permutation(key, len(X_train))
    shuffled_X = X_train[permutation]
    shuffled_y = y_train[permutation]
    
    # Calculate section size
    section_size = len(X_train) // num_sections
    
    section_accuracies = []
    
    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else len(X_train)
        
        # Get section data
        section_X = shuffled_X[start_idx:end_idx]
        section_y = shuffled_y[start_idx:end_idx]
        
        # Compute accuracy for this section
        section_accuracy = accuracy_fn(params, section_X, section_y, forward_fn)
        section_accuracies.append(float(section_accuracy))
        
        if i % 20 == 0:  # Log progress every 20 sections
            logging.info(f"Evaluated section {i+1}/{num_sections}, accuracy: {section_accuracy:.4f}")
    
    return section_accuracies, permutation

def evaluate_training_sections_animation_frame(
    params: Dict[str, Array],
    X_train: Array,
    y_train: Array,
    forward_fn: Callable,
    num_sections: int = 100,
    key: jax.random.PRNGKey = None,
    max_sections_to_eval: int = 20
) -> Tuple[List[float], Array]:
    """
    Evaluate model accuracy on a subset of training sections for animation frames.
    This is faster than evaluating all sections for real-time animation.
    
    Args:
        params: Model parameters
        X_train: Training data features
        y_train: Training data labels
        forward_fn: Forward function for the model
        num_sections: Number of sections to split the data into
        key: Random key for shuffling
        max_sections_to_eval: Maximum number of sections to evaluate for animation
    
    Returns:
        Tuple of (section_accuracies, shuffled_indices)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Shuffle the data
    permutation = jax.random.permutation(key, len(X_train))
    shuffled_X = X_train[permutation]
    shuffled_y = y_train[permutation]
    
    # Calculate section size
    section_size = len(X_train) // num_sections
    
    # For animation, we'll evaluate a subset of sections to make it faster
    sections_to_eval = min(max_sections_to_eval, num_sections)
    section_accuracies = []
    
    for i in range(sections_to_eval):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else len(X_train)
        
        # Get section data
        section_X = shuffled_X[start_idx:end_idx]
        section_y = shuffled_y[start_idx:end_idx]
        
        # Compute accuracy for this section
        section_accuracy = accuracy_fn(params, section_X, section_y, forward_fn)
        section_accuracies.append(float(section_accuracy))
    
    return section_accuracies, permutation

def evaluate_training_sections_with_permutation(
    params: Dict[str, Array],
    X_train: Array,
    y_train: Array,
    forward_fn: Callable,
    num_sections: int = 100,
    max_sections_to_eval: int = 20
) -> List[float]:
    """
    Evaluate model accuracy on training sections using pre-shuffled data.
    This function assumes the data is already shuffled and uses the same permutation
    throughout the epoch for consistent section evaluation.
    
    Args:
        params: Model parameters
        X_train: Pre-shuffled training data features
        y_train: Pre-shuffled training data labels
        forward_fn: Forward function for the model
        num_sections: Number of sections to split the data into
        max_sections_to_eval: Maximum number of sections to evaluate for animation
    
    Returns:
        List of section accuracies
    """
    # Calculate section size
    section_size = len(X_train) // num_sections
    
    # For animation, we'll evaluate a subset of sections to make it faster
    sections_to_eval = min(max_sections_to_eval, num_sections)
    section_accuracies = []
    
    for i in range(sections_to_eval):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else len(X_train)
        
        # Get section data from the pre-shuffled data
        section_X = X_train[start_idx:end_idx]
        section_y = y_train[start_idx:end_idx]
        
        # Compute accuracy for this section
        section_accuracy = accuracy_fn(params, section_X, section_y, forward_fn)
        section_accuracies.append(float(section_accuracy))
    
    return section_accuracies

def get_section_statistics(section_accuracies: List[float]) -> Dict[str, float]:
    """
    Compute statistics for section accuracies.
    
    Args:
        section_accuracies: List of accuracy values for each section
    
    Returns:
        Dictionary containing mean, std, min, max statistics
    """
    import numpy as np
    
    accuracies = np.array(section_accuracies)
    
    return {
        'mean': float(np.mean(accuracies)),
        'std': float(np.std(accuracies)),
        'min': float(np.min(accuracies)),
        'max': float(np.max(accuracies)),
        'median': float(np.median(accuracies))
    }

def evaluate_model_performance(
    params: Dict[str, Array],
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    forward_fn: Callable,
    num_sections: int = 100,
    key: jax.random.PRNGKey = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation including training sections and test set.
    
    Args:
        params: Model parameters
        X_train: Training data features
        y_train: Training data labels
        X_test: Test data features
        y_test: Test data labels
        forward_fn: Forward function for the model
        num_sections: Number of sections to split training data into
        key: Random key for shuffling
    
    Returns:
        Dictionary containing all evaluation results
    """
    logging.info("Starting comprehensive model evaluation...")
    
    # Evaluate training sections
    section_accuracies, shuffled_indices = evaluate_training_sections(
        params, X_train, y_train, forward_fn, num_sections, key
    )
    
    # Get section statistics
    section_stats = get_section_statistics(section_accuracies)
    
    # Evaluate on test set
    test_accuracy = accuracy_fn(params, X_test, y_test, forward_fn)
    
    # Compile results
    results = {
        'section_accuracies': section_accuracies,
        'section_statistics': section_stats,
        'test_accuracy': float(test_accuracy),
        'shuffled_indices': shuffled_indices,
        'num_sections': num_sections
    }
    
    logging.info(f"Evaluation completed:")
    logging.info(f"  Test accuracy: {test_accuracy:.4f}")
    logging.info(f"  Training sections - Mean: {section_stats['mean']:.4f}, Std: {section_stats['std']:.4f}")
    logging.info(f"  Training sections - Min: {section_stats['min']:.4f}, Max: {section_stats['max']:.4f}")
    
    return results 