# Visualization Testing

This directory contains test scripts for testing the visualization functions in the kmeans project.

## Test Script: `test_visualize.py`

A comprehensive test script that tests all visualization functions with synthetic data.

### Features

- **Synthetic Data Generation**: Creates realistic MNIST-like data for testing
- **Individual Function Testing**: Tests each visualization component separately
- **Full Function Testing**: Tests the complete visualization pipeline
- **Error Handling**: Comprehensive error reporting and debugging
- **File Output**: Generates test plots and animations for visual verification

### What It Tests

1. **`create_mnist_probability_pair`** - Individual MNIST + probability pair creation
2. **`create_cluster_distribution_axis`** - Cluster distribution plot creation
3. **`plot_final_probabilities`** - Main final probabilities visualization
4. **`plot_training_progress`** - Training loss/accuracy plots
5. **`create_probability_animation`** - Animation creation

### Usage

```bash
# Navigate to the test directory
cd projects/kmeans/test

# Run the test script
python test_visualize.py
```

### Output Files

The script generates several test files:

- `test_single_pair.png` - Single MNIST + probability pair test
- `test_cluster_distribution.png` - Cluster distribution test
- `test_final_probabilities.png` - Main visualization test
- `test_training_progress.png` - Training progress test
- `test_animation.gif` - Animation test

### Requirements

- Python 3.6+
- numpy
- matplotlib
- Access to the parent `visualize.py` module

### Troubleshooting

If you encounter import errors:

1. Make sure you're running from the `test/` directory
2. Verify that `../visualize.py` exists and is accessible
3. Check that all required dependencies are installed

### Customization

You can modify the test script to:

- Change the number of examples (default: 12)
- Adjust the number of clusters (default: 10)
- Modify image dimensions (default: 28x28)
- Change the number of animation frames (default: 5)
- Customize the synthetic data generation parameters

### Example Output

```
VISUALIZATION FUNCTION TESTING SCRIPT
============================================================
This script will test all visualization functions with synthetic data.
Generated plots will be saved in the current directory.

============================================================
TESTING INDIVIDUAL VISUALIZATION FUNCTIONS
============================================================

Generating 12 synthetic MNIST images with dimensions (28, 28)
Generated X_examples shape: (12, 784)
Generated y_examples: [3 7 1 8 4 9 2 6 5 0 1 7]

1. Testing create_mnist_probability_pair...
✓ create_mnist_probability_pair: SUCCESS
  Saved test plot as 'test_single_pair.png'

2. Testing create_cluster_distribution_axis...
✓ create_cluster_distribution_axis: SUCCESS
  Saved test plot as 'test_cluster_distribution.png'

============================================================
TESTING plot_final_probabilities FUNCTION
============================================================

Testing plot_final_probabilities...
✓ plot_final_probabilities: SUCCESS
  Saved plot as 'test_final_probabilities.png'

============================================================
TESTING COMPLETE
============================================================
Check the current directory for generated test files:
- test_single_pair.png
- test_cluster_distribution.png
- test_final_probabilities.png
- test_training_progress.png
- test_animation.gif

Generated 5 test files:
  - test_animation.gif
  - test_cluster_distribution.png
  - test_final_probabilities.png
  - test_single_pair.png
  - test_training_progress.png
```
