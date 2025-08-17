"""
Visualization functions for MNIST clustering experiments.
Pure functions that can be imported and used independently.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import subprocess
import io
from typing import List, Dict, Any, Tuple
from jax import Array
from shared_lib.media import save_media

def check_ffmpeg_available():
    """Check if ffmpeg is available for MP4 encoding."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_default_output_dir():
    """Get the default output directory following project conventions."""
    # Create outputs directory if it doesn't exist (use relative path from project root)
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Create date-based subdirectory (yy-mm-dd format)
    from datetime import datetime
    date_str = datetime.now().strftime('%y-%m-%d')
    date_dir = os.path.join(outputs_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)
    
    return date_dir


def create_mnist_image_axis(fig, gs, row, col_start, img_data, example_num, true_label):
    """
    Create an MNIST image subplot.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        row: row index
        col_start: starting column index
        img_data: MNIST image data (784,)
        example_num: example number (1-based)
        true_label: true label for the image
        
    Returns:
        matplotlib axis object
    """
    ax_img = fig.add_subplot(gs[row, col_start:col_start+1])  # Span only 1 column for smaller image
    img = img_data.reshape(28, 28)
    ax_img.imshow(img, cmap='gray', interpolation='nearest', aspect='equal')  # Keep aspect ratio
    ax_img.set_title(f'Example {example_num} (True: {true_label})', fontsize=9)
    ax_img.axis('off')
    return ax_img


def create_probability_axis(fig, gs, row, col_start, remove_borders=True, span_columns=1):
    """
    Create a probability subplot with proper formatting.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        row: row index
        col_start: starting column index
        remove_borders: whether to remove right and top axis borders
        span_columns: number of columns to span (default: 1, use 2 for wider plots)
        
    Returns:
        matplotlib axis object
    """
    if span_columns == 2:
        ax_prob = fig.add_subplot(gs[row, col_start:col_start+2])  # Span 2 columns for wider probability plot
    else:
        ax_prob = fig.add_subplot(gs[row, col_start])
    
    ax_prob.set_xlabel('Cluster Label')
    # Remove y-label to save space
    ax_prob.set_ylim(0, 100)  # Use percentage scale (0-100%)
    ax_prob.set_xlim(-0.5, 9.5)
    ax_prob.set_xticks(range(10))
    ax_prob.set_xticklabels([f'C{j}' for j in range(10)])
    
    if remove_borders:
        # Remove right and top axis borders
        ax_prob.spines['right'].set_visible(False)
        ax_prob.spines['top'].set_visible(False)
    
    return ax_prob


def create_cluster_distribution_axis(fig, gs, title, remove_borders=True):
    """
    Create a cluster distribution subplot spanning all rows.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        title: title for the plot
        remove_borders: whether to remove right and top axis borders
        
    Returns:
        matplotlib axis object
    """
    ax_dist = fig.add_subplot(gs[:, 9:12])  # Span all rows, last 3 columns
    ax_dist.set_title(title, fontsize=12)
    ax_dist.set_xlabel('Cluster Label')
    ax_dist.set_ylabel('Number of Items')
    ax_dist.set_xlim(-0.5, 9.5)
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels([f'C{j}' for j in range(10)])
    
    if remove_borders:
        # Remove right and top axis borders
        ax_dist.spines['right'].set_visible(False)
        ax_dist.spines['top'].set_visible(False)
    
    return ax_dist


def create_image_probability_grid(fig, gs, X_examples, y_examples, create_prob_axis_func):
    """
    Create the grid of MNIST images and probability plots.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        X_examples: Array of example images (N, 784)
        y_examples: Array of true labels for examples
        create_prob_axis_func: Function to create probability axis (for animation vs static)
        
    Returns:
        tuple: (image_axes, prob_axes, prob_axes_refs)
    """
    image_axes = []
    prob_axes = []
    prob_axes_refs = []  # Store references for easier access
    
    num_examples = len(X_examples)
    for i in range(num_examples):
        row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
        col_start = (i % 3) * 3  # 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 3
        
        # Create MNIST image axis
        ax_img = create_mnist_image_axis(fig, gs, row, col_start, X_examples[i], i+1, y_examples[i])
        image_axes.append(ax_img)
        
        # Create probability axis using common function - spans 2 columns for more space
        ax_prob = create_probability_axis(fig, gs, row, col_start + 1, remove_borders=True, span_columns=2)
        prob_axes.append(ax_prob)
        prob_axes_refs.append(ax_prob)
    
    return image_axes, prob_axes, prob_axes_refs


def setup_animation_probability_axis(fig, gs, row, col_start):
    """
    Create a probability axis specifically for animation (with bars).
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        row: row index
        col_start: starting column index
        
    Returns:
        tuple: (axis, bars)
    """
    # Create probability axis using common function - spans 2 columns for more space
    ax_prob = create_probability_axis(fig, gs, row, col_start, remove_borders=True, span_columns=2)
    
    # Initialize bars with wider spacing
    bar_heights = np.zeros(10)
    bars = ax_prob.bar(range(10), bar_heights, alpha=0.7, color='skyblue', width=0.8)
    
    return ax_prob, bars


def setup_static_probability_axis(fig, gs, row, col_start, probabilities):
    """
    Create a probability axis specifically for static plots (with final probabilities).
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        row: row index
        col_start: starting column index
        probabilities: probability array for this example
        
    Returns:
        tuple: (axis, bars)
    """
    # Create probability axis using common function - spans 2 columns for more space
    ax_prob = create_probability_axis(fig, gs, row, col_start, remove_borders=True, span_columns=2)
    
    # Create bars with final probabilities (convert to percentages)
    probabilities_percent = probabilities * 100
    bars = ax_prob.bar(range(10), probabilities_percent, alpha=0.7, color='lightcoral', width=0.8)
    
    # Highlight the maximum probability
    max_idx = np.argmax(probabilities)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.9)
    
    # Add probability values on bars (show as percentages with 1 decimal)
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax_prob.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
    
    return ax_prob, bars


def create_mnist_probability_pair(fig, gs, row, col_start, img_data, example_num, true_label, 
                                 probabilities=None, is_animation=False):
    """
    Create a complete MNIST image + probability pair.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        row: row index
        col_start: starting column index
        img_data: MNIST image data (784,)
        example_num: example number (1-based)
        true_label: true label for the image
        probabilities: probability array (for static plots) or None (for animation)
        is_animation: whether this is for animation (affects bar setup)
        
    Returns:
        tuple: (image_axis, prob_axis, bars) where bars may be None for animation
    """
    # Create MNIST image
    ax_img = create_mnist_image_axis(fig, gs, row, col_start, img_data, example_num, true_label)
    
    # Create probability axis using common function - spans 2 columns for more space
    ax_prob = create_probability_axis(fig, gs, row, col_start + 1, remove_borders=True, span_columns=2)
    
    bars = None
    if not is_animation and probabilities is not None:
        # For static plots, create bars with probabilities (convert to percentages)
        probabilities_percent = probabilities * 100
        bars = ax_prob.bar(range(10), probabilities_percent, alpha=0.7, color='lightcoral', width=0.8)
        
        # Highlight the maximum probability
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(0.9)
        
        # Add probability values on bars (show as percentages with 1 decimal)
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    return ax_img, ax_prob, bars


def create_mnist_probability_grid(fig, gs, X_examples, y_examples, probabilities=None, is_animation=False):
    """
    Create the complete grid of MNIST + probability pairs.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        X_examples: Array of example images (N, 784)
        y_examples: Array of true labels for examples
        probabilities: Array of probability distributions (N, 10) or None for animation
        is_animation: whether this is for animation
        
    Returns:
        tuple: (image_axes, prob_axes, all_bars) where all_bars may be empty for animation
    """
    image_axes = []
    prob_axes = []
    all_bars = []
    
    num_examples = len(X_examples)
    for i in range(num_examples):
        row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
        col_start = (i % 3) * 3  # 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 3
        
        # Get probabilities for this example if available
        example_probs = probabilities[i] if probabilities is not None else None
        
        # Create complete MNIST + probability pair
        ax_img, ax_prob, bars = create_mnist_probability_pair(
            fig, gs, row, col_start, X_examples[i], i+1, y_examples[i], 
            example_probs, is_animation
        )
        
        image_axes.append(ax_img)
        prob_axes.append(ax_prob)
        if bars is not None:
            all_bars.append(bars)
    
    return image_axes, prob_axes, all_bars


def create_probability_animation(
    animation_frames: List[Dict[str, Any]], 
    X_examples: np.ndarray, 
    y_examples: np.ndarray, 
    save_path: str = None, 
    title: str = "MNIST Clustering Probability Evolution"
) -> None:
    """
    Create an animation showing probability evolution for 12 examples with sample images.
    
    Args:
        animation_frames: List of frames with probability data and cluster counts
        save_path: Path to save the animation (defaults to outputs/yy-mm-dd/mnist_clustering_probability_evolution.gif)
        X_examples: Array of example images (12, 784)
        y_examples: Array of true labels for examples
        title: Title for the animation
    """
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, "mnist_clustering_probability_evolution.gif")
    
    fig = plt.figure(figsize=(24, 15))
    fig.suptitle(title, fontsize=16)
    
    # Create a 4x12 grid (4 rows, 12 columns) with tighter spacing
    # Columns 0-8: samples (3 groups of 3 columns each)
    # Columns 9-11: cluster distribution (right side)
    gs = fig.add_gridspec(4, 12, hspace=0.3, wspace=0.2)
    
    # Set up the bar plots with sample images using the new common function
    bars = []
    prob_axes = []  # Store references to probability axes for easier access
    text_objects = []  # Store text objects for easy removal/update
    
    # Create the grid using the new common function
    for i in range(12):  # 12 examples
        row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
        col_start = (i % 3) * 3  # 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 3
        
        # Create complete MNIST + probability pair for animation
        _, ax_prob, _ = create_mnist_probability_pair(
            fig, gs, row, col_start, X_examples[i], i+1, y_examples[i], 
            is_animation=True
        )
        prob_axes.append(ax_prob)
        
        # Initialize bars with wider spacing (use percentage scale)
        bar_heights = np.zeros(10)
        bars.append(ax_prob.bar(range(10), bar_heights, alpha=0.7, color='skyblue', width=0.8))
        
        # Initialize text objects for this axis (10 text objects, one for each bar)
        text_objects.append([ax_prob.text(0, 0, '', ha='center', va='bottom', fontsize=7) for _ in range(10)])
    
    # Add cluster distribution plot on the right side using common function
    ax_dist = create_cluster_distribution_axis(fig, gs, 'Cluster Distribution (All Training Samples)')
    
    # Initialize cluster distribution bars
    dist_bars = ax_dist.bar(range(10), np.zeros(10), alpha=0.7, color='lightgreen', width=0.8)
    
    def animate(frame_idx):
        frame = animation_frames[frame_idx]
        probabilities = frame['probabilities']  # Shape: (12, 10)
        cluster_counts = frame['cluster_counts']  # Shape: (10,)
        
        # Update probability bars
        for i in range(12):
            if i < probabilities.shape[0]:
                # Update bar heights (convert to percentages)
                for j, bar in enumerate(bars[i]):
                    bar.set_height(probabilities[i, j] * 100)  # Convert to percentage
                
                # Highlight the maximum probability (like in static plots)
                max_idx = np.argmax(probabilities[i])
                for j, bar in enumerate(bars[i]):
                    if j == max_idx:
                        bar.set_color('red')
                        bar.set_alpha(0.9)
                    else:
                        bar.set_color('skyblue')
                        bar.set_alpha(0.7)
                
                # Update text objects with probability values (show as percentages with 1 decimal)
                for j, bar in enumerate(bars[i]):
                    height = bar.get_height()
                    # Update existing text object instead of creating new ones
                    text_objects[i][j].set_position((bar.get_x() + bar.get_width()/2., height + 1))
                    text_objects[i][j].set_text(f'{height:.1f}%')
        
        # Update cluster distribution using pre-calculated data
        for j, bar in enumerate(dist_bars):
            bar.set_height(cluster_counts[j])
        
        # Update cluster distribution title with total count
        total_samples = np.sum(cluster_counts)
        ax_dist.set_title(f'Cluster Distribution (Total: {total_samples})', fontsize=12)
        
        # Set y-axis limit based on actual data
        max_count = np.max(cluster_counts)
        ax_dist.set_ylim(0, max_count * 1.1)  # Add 10% padding
        
        # Add frame info
        fig.text(0.02, 0.02, f'Batch {frame["batch"]}/{frame["total_batches"]}', 
                transform=fig.transFigure, fontsize=12, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        return bars + list(dist_bars)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(animation_frames), 
        interval=500, blit=False, repeat=True
    )
    
    # Save animation with upload
    try:
        # Save to temporary file in outputs directory first
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        
        # Determine writer based on file extension
        if save_path.lower().endswith('.mp4'):
            # Use ffmpeg writer for MP4 if available
            if check_ffmpeg_available():
                try:
                    anim.save(temp_path, writer='ffmpeg', fps=2)
                    print(f"MP4 animation saved to temporary file: {temp_path}")
                except Exception as mp4_error:
                    print(f"Failed to save MP4 with ffmpeg: {mp4_error}")
                    # Fall back to GIF if MP4 fails
                    temp_path_gif = temp_path.replace('.mp4', '.gif')
                    anim.save(temp_path_gif, writer='pillow', fps=2)
                    temp_path = temp_path_gif
                    print(f"Fell back to GIF format: {temp_path}")
            else:
                print("ffmpeg not available, falling back to GIF format")
                # Fall back to GIF if ffmpeg is not available
                temp_path_gif = temp_path.replace('.mp4', '.gif')
                anim.save(temp_path_gif, writer='pillow', fps=2)
                temp_path = temp_path_gif
        else:
            # Use pillow writer for GIF
            anim.save(temp_path, writer='pillow', fps=2)
        
        # Upload using media.py
        content_type = 'video/mp4' if save_path.lower().endswith('.mp4') else 'image/gif'
        try:
            uploaded_url = save_media(save_path, temp_path, content_type=content_type)
            print(f"Animation uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Animation saved locally to {temp_path}")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save animation: {e}")
        # Fall back to direct save in outputs directory
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        if save_path.lower().endswith('.mp4'):
            if check_ffmpeg_available():
                try:
                    anim.save(fallback_path, writer='ffmpeg', fps=2)
                    print(f"Animation saved locally to: {fallback_path}")
                except Exception as mp4_error:
                    print(f"MP4 save failed, falling back to GIF: {mp4_error}")
                    gif_path = fallback_path.replace('.mp4', '.gif')
                    anim.save(gif_path, writer='pillow', fps=2)
                    print(f"Animation saved locally as GIF: {gif_path}")
            else:
                print("ffmpeg not available, saving as GIF instead")
                gif_path = fallback_path.replace('.mp4', '.gif')
                anim.save(gif_path, writer='pillow', fps=2)
                print(f"Animation saved locally as GIF: {gif_path}")
        else:
            anim.save(fallback_path, writer='pillow', fps=2)
            print(f"Animation saved locally to: {fallback_path}")
    
    plt.close()


def plot_training_progress(
    train_losses: List[float], 
    train_accuracies: List[float], 
    save_path: str = None
) -> None:
    """
    Plot training loss and accuracy curves.
    
    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        save_path: Path to save the plot (defaults to outputs/yy-mm-dd/mnist_clustering_training_progress.png)
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Use subplots_adjust for simple subplot layouts
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, "mnist_clustering_training_progress.png")
    
    # Save plot with upload
    try:
        # Save to temporary file in outputs directory first
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        
        # Upload using media.py
        try:
            uploaded_url = save_media(save_path, temp_path, content_type='image/png')
            print(f"Training progress plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Training progress plot saved locally to {temp_path}")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save training progress plot: {e}")
        # Fall back to direct save in outputs directory
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved locally to: {fallback_path}")
    
    plt.close()


def plot_final_probabilities(
    final_probabilities: np.ndarray,
    X_examples: np.ndarray,
    y_examples: np.ndarray,
    run_uid: str,
    all_cluster_counts: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot final probability distributions for all examples with sample images.
    
    Args:
        final_probabilities: Array of final probability distributions (12, 10)
        X_examples: Array of example images (12, 784)
        y_examples: Array of true labels for examples
        run_uid: Unique identifier for this run
        save_path: Path to save the plot (defaults to outputs/yy-mm-dd/mnist_clustering_final_probabilities_{run_uid}.png)
        all_cluster_counts: Cluster distribution for all training samples (10,)
    """
    fig = plt.figure(figsize=(24, 15))
    fig.suptitle(f'Final Probability Distributions (Run: {run_uid})', fontsize=16)
    
    # Create a 4x12 grid (4 rows, 12 columns) with tighter spacing
    # Columns 0-8: samples (3 groups of 3 columns each)
    # Columns 9-11: cluster distribution (right side)
    gs = fig.add_gridspec(4, 12, hspace=0.3, wspace=0.2)
    
    num_examples = len(X_examples)
    all_bars = []
    for i in range(num_examples):
        row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
        col_start = (i % 3) * 3  # 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6
        
        # Create complete MNIST + probability pair using common function
        _, _, bars = create_mnist_probability_pair(
            fig, gs, row, col_start, X_examples[i], i+1, y_examples[i], 
            final_probabilities[i], is_animation=False
        )
        all_bars.append(bars)
    
    # Add cluster distribution plot on the right side using common function
    ax_dist = create_cluster_distribution_axis(fig, gs, 'Final Cluster Distribution (All Training Samples)')
    
    # Plot cluster distribution using pre-calculated data
    dist_bars = ax_dist.bar(range(10), all_cluster_counts, alpha=0.7, color='lightgreen', width=0.8)
    total_samples = np.sum(all_cluster_counts)
    ax_dist.set_title(f'Final Cluster Distribution (Total: {total_samples})', fontsize=12)
    
    # Set y-axis limit based on actual data with padding
    max_count = np.max(all_cluster_counts)
    ax_dist.set_ylim(0, max_count * 1.1)  # Add 10% padding
    
    # Add count values on bars
    for j, bar in enumerate(dist_bars):
        height = bar.get_height()
        ax_dist.text(bar.get_x() + bar.get_width()/2., height + max_count * 0.01,
               f'{int(height)}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Use subplots_adjust for complex GridSpec layouts instead of tight_layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, f"mnist_clustering_final_probabilities_{run_uid}.png")
    
    # Save plot with upload
    try:
        # Save to temporary file in outputs directory first
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        
        # Upload using media.py
        try:
            uploaded_url = save_media(save_path, temp_path, content_type='image/png')
            print(f"Final probabilities plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Final probabilities plot saved locally to {temp_path}")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save final probabilities plot: {e}")
        # Fall back to direct save in outputs directory
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
        print(f"Final probabilities plot saved locally to: {fallback_path}")
    plt.close()


def plot_ema_evolution(
    ema_history: List[np.ndarray],
    samples_trained_per_batch: List[int],
    save_path: str = None
) -> None:
    """
    Plot EMA values evolution and samples trained over time.
    
    Args:
        ema_history: List of EMA values arrays over time
        samples_trained_per_batch: List of number of samples trained per batch
        save_path: Path to save the plot (defaults to outputs/yy-mm-dd/mnist_clustering_ema_evolution.png)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Convert to numpy for easier plotting
    ema_array = np.array(ema_history)  # Shape: (num_batches, K)
    
    # Plot EMA evolution for each cluster
    for k in range(ema_array.shape[1]):
        ax1.plot(ema_array[:, k], label=f'Cluster {k}', alpha=0.8)
    
    ax1.set_title('EMA Values Evolution During Training')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('EMA Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot samples trained per batch
    ax2.plot(samples_trained_per_batch, color='red', linewidth=2)
    ax2.set_title('Number of Samples Trained per Batch (Above EMA Threshold)')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Samples Trained')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, "mnist_clustering_ema_evolution.png")
    
    # Save plot with upload
    try:
        # Save to temporary file in outputs directory first
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        
        # Upload using media.py
        try:
            uploaded_url = save_media(save_path, temp_path, content_type='image/png')
            print(f"EMA evolution plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"EMA evolution plot saved locally to {temp_path}")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save EMA evolution plot: {e}")
        # Fall back to direct save in outputs directory
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
        print(f"EMA evolution plot saved locally to: {fallback_path}")
    
    plt.close()


def create_clustering_visualization(
    animation_frames: List[Dict[str, Any]],
    final_probabilities: np.ndarray,
    X_examples: np.ndarray,
    y_examples: np.ndarray,
    train_losses: List[float],
    train_accuracies: List[float],
    run_uid: str,
    all_cluster_counts: np.ndarray,
    ema_history: List[np.ndarray] = None,
    samples_trained_per_batch: List[int] = None,
    output_dir: str = None
) -> Dict[str, str]:
    """
    Create all clustering visualizations in one function.
    
    Args:
        animation_frames: List of animation frames
        final_probabilities: Final probability distributions
        X_examples: Example images
        y_examples: True labels for examples
        train_losses: Training loss history
        train_accuracies: Training accuracy history
        run_uid: Unique identifier for this run
        all_cluster_counts: Cluster distribution for all training samples
        ema_history: List of EMA values over time (optional)
        samples_trained_per_batch: List of samples trained per batch (optional)
        output_dir: Directory to save outputs (defaults to outputs/yy-mm-dd/)
        
    Returns:
        Dictionary with paths to saved files
    """
    import os
    
    # Use default output directory if none specified
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Create probability evolution animation
    if animation_frames:
        anim_path = os.path.join(output_dir, f"mnist_clustering_probability_evolution_{run_uid}.gif")
        create_probability_animation(
            animation_frames=animation_frames,
            X_examples=X_examples,
            y_examples=y_examples,
            save_path=anim_path,
            title="MNIST Clustering: Probability Evolution for 12 Examples"
        )
        saved_files['animation'] = anim_path
    
    # Create training progress plot
    progress_path = os.path.join(output_dir, f"mnist_clustering_training_progress_{run_uid}.png")
    plot_training_progress(train_losses, train_accuracies, progress_path)
    saved_files['training_progress'] = progress_path
    
    # Create final probabilities plot
    final_probs_path = os.path.join(output_dir, f"mnist_clustering_final_probabilities_{run_uid}.png")
    plot_final_probabilities(
        final_probabilities, X_examples, y_examples, run_uid, all_cluster_counts, final_probs_path
    )
    saved_files['final_probabilities'] = final_probs_path
    
    # Create EMA evolution plot if data is provided
    if ema_history is not None and samples_trained_per_batch is not None:
        ema_path = os.path.join(output_dir, f"mnist_clustering_ema_evolution_{run_uid}.png")
        plot_ema_evolution(ema_history, samples_trained_per_batch, ema_path)
        saved_files['ema_evolution'] = ema_path
    
    return saved_files
