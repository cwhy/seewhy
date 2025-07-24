#!/usr/bin/env python3
"""
Test script for lib/media.py functionality.
Tests saving media files to R2 with fallback to local storage.
"""

import os
import sys
import io
import tempfile
from pathlib import Path

# Add the project root to the path so we can import lib modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared_lib.media import save_media, save_matplotlib_figure
import matplotlib.pyplot as plt
import numpy as np


def test_save_media_with_file_path():
    """Test saving a file from a file path."""
    print("\n=== Testing save_media with file path ===")
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test file for media upload")
        temp_file_path = f.name
    
    try:
        result = save_media("test_file.txt", temp_file_path)
        print(f"Result: {result}")
        return result
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def test_save_media_with_bytes():
    """Test saving media from bytes data."""
    print("\n=== Testing save_media with bytes ===")
    
    # Create some test bytes data
    test_data = b"This is test binary data for media upload"
    
    result = save_media("test_bytes.txt", test_data, "text/plain")
    print(f"Result: {result}")
    return result


def test_save_media_with_file_like_object():
    """Test saving media from a file-like object."""
    print("\n=== Testing save_media with file-like object ===")
    
    # Create a file-like object
    file_obj = io.BytesIO(b"This is test data from file-like object")
    
    result = save_media("test_filelike.txt", file_obj, "text/plain")
    print(f"Result: {result}")
    return result


def test_save_matplotlib_figure():
    """Test saving a matplotlib figure."""
    print("\n=== Testing save_matplotlib_figure ===")
    
    # Create a simple matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', label='sin(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Test Plot')
    ax.legend()
    ax.grid(True)
    
    result = save_matplotlib_figure("test_plot", fig, format="png", dpi=150)
    print(f"Result: {result}")
    
    # Close the figure to free memory
    plt.close(fig)
    return result


def test_save_matplotlib_figure_different_formats():
    """Test saving matplotlib figures in different formats."""
    print("\n=== Testing matplotlib figures in different formats ===")
    
    # Create a simple figure
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 2*np.pi, 100)
    y = np.cos(x)
    ax.plot(x, y, 'r-', label='cos(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Cosine Function')
    ax.legend()
    ax.grid(True)
    
    formats = ['png', 'jpg', 'svg']
    results = {}
    
    for fmt in formats:
        result = save_matplotlib_figure(f"test_plot_{fmt}", fig, format=fmt, dpi=150)
        results[fmt] = result
        print(f"{fmt.upper()}: {result}")
    
    plt.close(fig)
    return results


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n=== Testing error handling ===")
    
    # Test with non-existent file path
    try:
        result = save_media("nonexistent.txt", "/path/to/nonexistent/file.txt")
        print(f"Unexpected success: {result}")
    except Exception as e:
        print(f"Expected error for non-existent file: {e}")
    
    # Test with invalid media type
    try:
        result = save_media("test.txt", 123)  # Invalid type
        print(f"Unexpected success: {result}")
    except Exception as e:
        print(f"Expected error for invalid media type: {e}")


def check_outputs_directory():
    """Check what files were created in the outputs directory."""
    print("\n=== Checking outputs directory ===")
    
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        print("Files in outputs directory:")
        for file_path in outputs_dir.rglob("*"):
            if file_path.is_file():
                print(f"  {file_path}")
                # Print file size
                size = file_path.stat().st_size
                print(f"    Size: {size} bytes")
    else:
        print("Outputs directory does not exist")


def main():
    """Run all tests."""
    print("Starting media.py tests...")
    print("=" * 50)
    
    # Run tests
    test_save_media_with_file_path()
    test_save_media_with_bytes()
    test_save_media_with_file_like_object()
    test_save_matplotlib_figure()
    test_save_matplotlib_figure_different_formats()
    test_error_handling()
    
    # Check what was created
    check_outputs_directory()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main() 