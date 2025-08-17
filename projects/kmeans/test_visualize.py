#!/usr/bin/env python3
"""
Test script for MNIST clustering visualization functions.
Tests the default output directory functionality.
"""

import numpy as np
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from projects.kmeans.visualize import (
    get_default_output_dir,
    plot_training_progress,
    plot_final_probabilities,
    create_clustering_visualization
)

def test_default_output_directory():
    """Test that the default output directory is created correctly."""
    print("Testing default output directory...")
    
    output_dir = get_default_output_dir()
    print(f"Default output directory: {output_dir}")
    
    # Check if directory exists
    assert os.path.exists(output_dir), f"Output directory {output_dir} was not created"
    print(f"✅ Output directory created successfully: {output_dir}")
    
    return output_dir

def test_plot_training_progress():
    """Test training progress plot with default output directory."""
    print("\nTesting training progress plot...")
    
    # Generate dummy data
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    train_accuracies = [0.6, 0.7, 0.8, 0.85, 0.9]
    
    # Test with default save path
    plot_training_progress(train_losses, train_accuracies)
    print("✅ Training progress plot created with default path")
    
    # Test with custom save path
    custom_path = os.path.join(get_default_output_dir(), "custom_training_progress.png")
    plot_training_progress(train_losses, train_accuracies, custom_path)
    print(f"✅ Training progress plot created with custom path: {custom_path}")

def test_plot_final_probabilities():
    """Test final probabilities plot with default output directory."""
    print("\nTesting final probabilities plot...")
    
    # Generate dummy data
    final_probabilities = np.random.rand(12, 10)
    final_probabilities = final_probabilities / final_probabilities.sum(axis=1, keepdims=True)  # Normalize
    
    X_examples = np.random.rand(12, 784)  # 12 MNIST examples
    y_examples = np.random.randint(0, 10, 12)  # 12 random labels
    run_uid = "TEST"
    all_cluster_counts = np.random.randint(100, 1000, 10)  # Random cluster counts
    
    # Test with default save path
    plot_final_probabilities(
        final_probabilities, X_examples, y_examples, run_uid, 
        all_cluster_counts=all_cluster_counts
    )
    print("✅ Final probabilities plot created with default path")
    
    # Test with custom save path
    custom_path = os.path.join(get_default_output_dir(), "custom_final_probabilities.png")
    plot_final_probabilities(
        final_probabilities, X_examples, y_examples, run_uid, 
        all_cluster_counts, custom_path
    )
    print(f"✅ Final probabilities plot created with custom path: {custom_path}")

def test_create_clustering_visualization():
    """Test the unified clustering visualization function."""
    print("\nTesting unified clustering visualization...")
    
    # Generate dummy data
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    train_accuracies = [0.6, 0.7, 0.8, 0.85, 0.9]
    
    final_probabilities = np.random.rand(12, 10)
    final_probabilities = final_probabilities / final_probabilities.sum(axis=1, keepdims=True)
    
    X_examples = np.random.rand(12, 784)
    y_examples = np.random.randint(0, 10, 12)
    run_uid = "TEST"
    all_cluster_counts = np.random.randint(100, 1000, 10)
    
    # Test with default output directory
    saved_files = create_clustering_visualization(
        animation_frames=[],  # Empty for testing
        final_probabilities=final_probabilities,
        X_examples=X_examples,
        y_examples=y_examples,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        run_uid=run_uid,
        all_cluster_counts=all_cluster_counts
    )
    
    print(f"✅ Unified visualization created successfully")
    print(f"Saved files: {saved_files}")
    
    # Verify files exist
    for file_type, file_path in saved_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {file_type}: {file_path}")
        else:
            print(f"  ❌ {file_type}: {file_path} (not found)")

def main():
    """Run all tests."""
    print("🧪 Testing MNIST clustering visualization functions...")
    print("=" * 60)
    
    try:
        # Test default output directory
        output_dir = test_default_output_directory()
        
        # Test individual functions
        test_plot_training_progress()
        test_plot_final_probabilities()
        test_create_clustering_visualization()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print(f"📁 Output files saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
