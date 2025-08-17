#!/usr/bin/env python3
"""
Test script for testing the plot_final_probabilities visualization function.
This script creates synthetic data and tests the visualization functions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the visualization module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from visualize import (
    plot_final_probabilities,
    plot_training_progress,
    create_probability_animation,
    create_mnist_probability_pair,
    create_mnist_probability_grid,
    create_cluster_distribution_axis
)

def generate_synthetic_mnist_data(num_examples=12, img_dims=(28, 28)):
    """
    Generate synthetic MNIST-like data for testing.
    
    Args:
        num_examples: Number of example images to generate
        img_dims: Image dimensions (height, width)
        
    Returns:
        tuple: (X_examples, y_examples) where X_examples is (num_examples, 784)
    """
    print(f"Generating {num_examples} synthetic MNIST images with dimensions {img_dims}")
    
    # Generate random image data (784 = 28*28)
    X_examples = np.random.rand(num_examples, img_dims[0] * img_dims[1])
    
    # Normalize to [0, 1] to simulate MNIST pixel values
    X_examples = (X_examples - X_examples.min()) / (X_examples.max() - X_examples.min())
    
    # Generate random true labels (0-9 for MNIST digits)
    y_examples = np.random.randint(0, 10, num_examples)
    
    print(f"Generated X_examples shape: {X_examples.shape}")
    print(f"Generated y_examples: {y_examples}")
    
    return X_examples, y_examples


def generate_synthetic_probabilities(num_examples=12, num_clusters=10):
    """
    Generate synthetic probability distributions for testing.
    
    Args:
        num_examples: Number of examples
        num_clusters: Number of clusters (K)
        
    Returns:
        np.ndarray: Shape (num_examples, num_clusters) with probability distributions
    """
    print(f"Generating probability distributions for {num_examples} examples with {num_clusters} clusters")
    
    # Generate random probabilities for each example
    probabilities = np.random.rand(num_examples, num_clusters)
    
    # Normalize to make them proper probability distributions (sum to 1)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Make some probabilities more dominant (simulate clustering)
    for i in range(num_examples):
        # Randomly select a dominant cluster
        dominant_cluster = np.random.randint(0, num_clusters)
        probabilities[i, dominant_cluster] *= 3  # Make it more dominant
        # Renormalize
        probabilities[i] = probabilities[i] / probabilities[i].sum()
    
    print(f"Generated probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities for example 0: {probabilities[0]}")
    print(f"Sum of probabilities for example 0: {probabilities[0].sum():.6f}")
    
    return probabilities


def generate_synthetic_cluster_counts(num_clusters=10, total_samples=1000):
    """
    Generate synthetic cluster distribution counts.
    
    Args:
        num_clusters: Number of clusters
        total_samples: Total number of training samples
        
    Returns:
        np.ndarray: Shape (num_clusters,) with cluster counts
    """
    print(f"Generating cluster distribution for {num_clusters} clusters with {total_samples} total samples")
    
    # Generate random cluster counts
    cluster_counts = np.random.randint(50, 200, num_clusters)
    
    # Normalize to match total_samples
    cluster_counts = (cluster_counts / cluster_counts.sum() * total_samples).astype(int)
    
    # Adjust to ensure exact total
    diff = total_samples - cluster_counts.sum()
    if diff > 0:
        cluster_counts[0] += diff
    elif diff < 0:
        cluster_counts[0] = max(0, cluster_counts[0] + diff)
    
    print(f"Generated cluster counts: {cluster_counts}")
    print(f"Total samples: {cluster_counts.sum()}")
    
    return cluster_counts


def test_individual_functions():
    """Test individual visualization functions."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL VISUALIZATION FUNCTIONS")
    print("="*60)
    
    # Test data
    X_examples, y_examples = generate_synthetic_mnist_data(12)
    probabilities = generate_synthetic_probabilities(12, 10)
    cluster_counts = generate_synthetic_cluster_counts(10, 1000)
    
    # Test 1: create_mnist_probability_pair
    print("\n1. Testing create_mnist_probability_pair...")
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.2)
    
    try:
        ax_img, ax_prob, bars = create_mnist_probability_pair(
            fig, gs, 0, 0, X_examples[0], 1, y_examples[0], 
            probabilities[0], is_animation=False
        )
        print("✓ create_mnist_probability_pair: SUCCESS")
        
        # Save test plot
        plt.savefig('test_single_pair.png', dpi=150, bbox_inches='tight')
        print("  Saved test plot as 'test_single_pair.png'")
        
    except Exception as e:
        print(f"✗ create_mnist_probability_pair: FAILED - {e}")
    
    plt.close()
    
    # Test 2: create_cluster_distribution_axis
    print("\n2. Testing create_cluster_distribution_axis...")
    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(1, 1)
    
    try:
        ax_dist = create_cluster_distribution_axis(fig, gs, 'Test Cluster Distribution')
        print("✓ create_cluster_distribution_axis: SUCCESS")
        
        # Add some test bars
        ax_dist.bar(range(10), cluster_counts, alpha=0.7, color='lightgreen', width=0.8)
        plt.savefig('test_cluster_distribution.png', dpi=150, bbox_inches='tight')
        print("  Saved test plot as 'test_cluster_distribution.png'")
        
    except Exception as e:
        print(f"✗ create_cluster_distribution_axis: FAILED - {e}")
    
    plt.close()


def test_plot_final_probabilities():
    """Test the main plot_final_probabilities function."""
    print("\n" + "="*60)
    print("TESTING plot_final_probabilities FUNCTION")
    print("="*60)
    
    # Generate test data
    X_examples, y_examples = generate_synthetic_mnist_data(12)
    final_probabilities = generate_synthetic_probabilities(12, 10)
    all_cluster_counts = generate_synthetic_cluster_counts(10, 1000)
    run_uid = "TEST123"
    
    # Test the main function
    print("\nTesting plot_final_probabilities...")
    try:
        plot_final_probabilities(
            final_probabilities=final_probabilities,
            X_examples=X_examples,
            y_examples=y_examples,
            run_uid=run_uid,
            save_path="test_final_probabilities.png",
            all_cluster_counts=all_cluster_counts
        )
        print("✓ plot_final_probabilities: SUCCESS")
        print("  Saved plot as 'test_final_probabilities.png'")
        
    except Exception as e:
        print(f"✗ plot_final_probabilities: FAILED - {e}")
        import traceback
        traceback.print_exc()


def test_plot_training_progress():
    """Test the plot_training_progress function."""
    print("\n" + "="*60)
    print("TESTING plot_training_progress FUNCTION")
    print("="*60)
    
    # Generate synthetic training data
    num_batches = 100
    train_losses = np.random.exponential(1.0, num_batches) * np.exp(-np.linspace(0, 3, num_batches))
    train_accuracies = 1 - np.random.exponential(0.1, num_batches) * np.exp(-np.linspace(0, 2, num_batches))
    
    print(f"Generated {num_batches} training batches")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Final accuracy: {train_accuracies[-1]:.4f}")
    
    try:
        plot_training_progress(
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            save_path="test_training_progress.png"
        )
        print("✓ plot_training_progress: SUCCESS")
        print("  Saved plot as 'test_training_progress.png'")
        
    except Exception as e:
        print(f"✗ plot_training_progress: FAILED - {e}")
        import traceback
        traceback.print_exc()


def test_animation_function():
    """Test the create_probability_animation function with synthetic data."""
    print("\n" + "="*60)
    print("TESTING create_probability_animation FUNCTION")
    print("="*60)
    
    # Generate test data
    X_examples, y_examples = generate_synthetic_mnist_data(12)
    
    # Generate synthetic animation frames
    num_frames = 5
    animation_frames = []
    
    for frame_idx in range(num_frames):
        # Generate evolving probabilities
        base_probs = np.random.rand(12, 10)
        # Make them evolve over time
        evolution_factor = frame_idx / (num_frames - 1)
        base_probs = base_probs * (1 + evolution_factor)
        
        # Normalize to probabilities
        base_probs = base_probs / base_probs.sum(axis=1, keepdims=True)
        
        # Generate cluster counts
        cluster_counts = np.random.randint(50, 200, 10)
        
        frame = {
            'probabilities': base_probs,
            'cluster_counts': cluster_counts,
            'batch': frame_idx + 1,
            'total_batches': num_frames
        }
        animation_frames.append(frame)
    
    print(f"Generated {num_frames} animation frames")
    
    try:
        create_probability_animation(
            animation_frames=animation_frames,
            save_path="test_animation.gif",
            X_examples=X_examples,
            y_examples=y_examples,
            title="Test MNIST Clustering Animation"
        )
        print("✓ create_probability_animation: SUCCESS")
        print("  Saved animation as 'test_animation.gif'")
        
    except Exception as e:
        print(f"✗ create_probability_animation: FAILED - {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("VISUALIZATION FUNCTION TESTING SCRIPT")
    print("="*60)
    print("This script will test all visualization functions with synthetic data.")
    print("Generated plots will be saved in the current directory.")
    
    # Test individual functions first
    test_individual_functions()
    
    # Test main visualization functions
    test_plot_final_probabilities()
    test_plot_training_progress()
    test_animation_function()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("Check the current directory for generated test files:")
    print("- test_single_pair.png")
    print("- test_cluster_distribution.png")
    print("- test_final_probabilities.png")
    print("- test_training_progress.png")
    print("- test_animation.gif")
    
    # List generated files
    import glob
    test_files = glob.glob("test_*.png") + glob.glob("test_*.gif")
    if test_files:
        print(f"\nGenerated {len(test_files)} test files:")
        for file in sorted(test_files):
            print(f"  - {file}")
    else:
        print("\nNo test files were generated (all tests may have failed)")


if __name__ == "__main__":
    main()
