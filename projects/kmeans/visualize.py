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


def create_probability_animation_with_reconstructions(
    animation_frames: List[Dict[str, Any]], 
    X_examples: np.ndarray, 
    y_examples: np.ndarray, 
    params: Dict[str, Any],
    key_gen: Any,
    save_path: str = None, 
    title: str = "MNIST Clustering with VAE Reconstructions"
) -> None:
    """
    Create an animation showing probability evolution for 12 examples with original and reconstructed images.
    
    Args:
        animation_frames: List of frames with probability data and cluster counts
        X_examples: Array of example images (12, 784)
        y_examples: Array of true labels for examples
        params: VAE model parameters for generating reconstructions
        key_gen: Key generator for JAX random operations
        save_path: Path to save the animation
        title: Title for the animation
    """
    # Import JAX functions for reconstruction
    import jax
    import jax.numpy as jnp
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, "mnist_clustering_probability_evolution_with_reconstructions.mp4")
    
    fig = plt.figure(figsize=(28, 20))
    fig.suptitle(title, fontsize=16)
    
    # Create a 8x15 grid (8 rows, 15 columns) with tighter spacing
    # Rows 0-3: Original images with probabilities (4 rows for 12 examples)
    # Rows 4-7: Reconstructed images (4 rows for 12 examples)
    # Columns 12-14: cluster distribution (right side)
    gs = fig.add_gridspec(8, 15, hspace=0.3, wspace=0.2)
    
    # Set up the bar plots with sample images
    bars = []
    prob_axes = []
    text_objects = []
    recon_image_objects = []  # For reconstruction images
    
    # Create the grid for original images and probabilities (top half)
    for i in range(12):  # 12 examples
        row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
        col_start = (i % 3) * 4  # 0, 4, 8, 0, 4, 8, 0, 4, 8, 0, 4, 8
        
        # Create original image (1 column)
        ax_orig = fig.add_subplot(gs[row, col_start])
        img = X_examples[i].reshape(28, 28)
        ax_orig.imshow(img, cmap='gray', interpolation='nearest', aspect='equal')
        ax_orig.set_title(f'Ex {i+1} (True: {y_examples[i]})', fontsize=8)
        ax_orig.axis('off')
        
        # Create probability axis (spans 3 columns)
        ax_prob = fig.add_subplot(gs[row, col_start + 1:col_start + 4])
        ax_prob.set_xlabel('Cluster Label')
        ax_prob.set_ylim(0, 100)  # Use percentage scale (0-100%)
        ax_prob.set_xlim(-0.5, 9.5)
        ax_prob.set_xticks(range(10))
        ax_prob.set_xticklabels([f'C{j}' for j in range(10)])
        ax_prob.spines['right'].set_visible(False)
        ax_prob.spines['top'].set_visible(False)
        
        prob_axes.append(ax_prob)
        
        # Initialize bars
        bar_heights = np.zeros(10)
        bars.append(ax_prob.bar(range(10), bar_heights, alpha=0.7, color='skyblue', width=0.8))
        
        # Initialize text objects for this axis
        text_objects.append([ax_prob.text(0, 0, '', ha='center', va='bottom', fontsize=7) for _ in range(10)])
        
        # Create reconstruction image (bottom half, same column structure)
        recon_row = row + 4  # Move to bottom half
        ax_recon = fig.add_subplot(gs[recon_row, col_start])
        # Initialize with black image, will be updated in animation
        initial_recon = np.zeros((28, 28))
        recon_im = ax_recon.imshow(initial_recon, cmap='gray', interpolation='nearest', aspect='equal', vmin=0, vmax=1)
        ax_recon.set_title(f'Recon C?', fontsize=8)
        ax_recon.axis('off')
        
        recon_image_objects.append({
            'image': recon_im,
            'axis': ax_recon
        })
    
    # Add cluster distribution plot on the right side
    ax_dist = fig.add_subplot(gs[:, 12:15])  # Span all rows, last 3 columns
    ax_dist.set_title('Cluster Distribution (All Training Samples)', fontsize=12)
    ax_dist.set_xlabel('Cluster Label')
    ax_dist.set_ylabel('Number of Items')
    ax_dist.set_xlim(-0.5, 9.5)
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels([f'C{j}' for j in range(10)])
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)
    
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
                    bar.set_height(probabilities[i, j] * 100)
                
                # Highlight the maximum probability
                max_idx = np.argmax(probabilities[i])
                for j, bar in enumerate(bars[i]):
                    if j == max_idx:
                        bar.set_color('red')
                        bar.set_alpha(0.9)
                    else:
                        bar.set_color('skyblue')
                        bar.set_alpha(0.7)
                
                # Update text objects with probability values
                for j, bar in enumerate(bars[i]):
                    height = bar.get_height()
                    text_objects[i][j].set_position((bar.get_x() + bar.get_width()/2., height + 1))
                    text_objects[i][j].set_text(f'{height:.1f}%')
                
                # Generate reconstruction using current parameters and assigned cluster
                assigned_cluster = max_idx  # Use the most probable cluster
                
                # Import reconstruction functions from the main script
                # We need to access these from the global scope or pass them as parameters
                try:
                    # Generate reconstruction for this example
                    recon_key = next(key_gen).get()
                    
                    # Use JAX functions directly since we have access to them
                    def vae_encode_local(x, cluster_id):
                        h1 = jnp.dot(x, params['enc_w1'][cluster_id]) + params['enc_b1'][cluster_id]
                        h1 = jax.nn.relu(h1)
                        z_mean = jnp.dot(h1, params['enc_w_mean'][cluster_id]) + params['enc_b_mean'][cluster_id]
                        z_logvar = jnp.dot(h1, params['enc_w_logvar'][cluster_id]) + params['enc_b_logvar'][cluster_id]
                        return z_mean, z_logvar
                    
                    def reparameterize_local(z_mean, z_logvar, key):
                        eps = jax.random.normal(key, z_mean.shape)
                        return z_mean + jnp.exp(0.5 * z_logvar) * eps
                    
                    def vae_decode_local(z, cluster_id):
                        h1 = jnp.dot(z, params['dec_w1'][cluster_id]) + params['dec_b1'][cluster_id]
                        h1 = jax.nn.relu(h1)
                        x_recon = jnp.dot(h1, params['dec_w2'][cluster_id]) + params['dec_b2'][cluster_id]
                        x_recon = jax.nn.sigmoid(x_recon)
                        return x_recon
                    
                    # Generate reconstruction
                    z_mean, z_logvar = vae_encode_local(X_examples[i], assigned_cluster)
                    z = reparameterize_local(z_mean, z_logvar, recon_key)
                    reconstruction = vae_decode_local(z, assigned_cluster)
                    
                    # Update reconstruction image
                    recon_img = reconstruction.reshape(28, 28)
                    recon_image_objects[i]['image'].set_array(recon_img)
                    recon_image_objects[i]['axis'].set_title(f'Recon C{assigned_cluster}', fontsize=8)
                    
                except Exception as e:
                    # If reconstruction fails, show placeholder
                    recon_image_objects[i]['image'].set_array(np.zeros((28, 28)))
                    recon_image_objects[i]['axis'].set_title(f'Recon Error', fontsize=8)
        
        # Update cluster distribution
        for j, bar in enumerate(dist_bars):
            bar.set_height(cluster_counts[j])
        
        # Update cluster distribution title with total count
        total_samples = np.sum(cluster_counts)
        ax_dist.set_title(f'Cluster Distribution (Total: {total_samples})', fontsize=12)
        
        # Set y-axis limit based on actual data
        max_count = np.max(cluster_counts)
        ax_dist.set_ylim(0, max_count * 1.1)
        
        # Add frame info
        fig.text(0.02, 0.02, f'Batch {frame["batch"]}/{frame["total_batches"]}', 
                transform=fig.transFigure, fontsize=12, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        return bars + list(dist_bars) + [obj['image'] for obj in recon_image_objects]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(animation_frames), 
        interval=500, blit=False, repeat=True
    )
    
    # Save animation with upload (same logic as original)
    try:
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        
        if save_path.lower().endswith('.mp4'):
            if check_ffmpeg_available():
                try:
                    anim.save(temp_path, writer='ffmpeg', fps=2)
                    print(f"MP4 animation saved to temporary file: {temp_path}")
                except Exception as mp4_error:
                    print(f"Failed to save MP4 with ffmpeg: {mp4_error}")
                    temp_path_gif = temp_path.replace('.mp4', '.gif')
                    anim.save(temp_path_gif, writer='pillow', fps=2)
                    temp_path = temp_path_gif
                    print(f"Fell back to GIF format: {temp_path}")
            else:
                print("ffmpeg not available, falling back to GIF format")
                temp_path_gif = temp_path.replace('.mp4', '.gif')
                anim.save(temp_path_gif, writer='pillow', fps=2)
                temp_path = temp_path_gif
        else:
            anim.save(temp_path, writer='pillow', fps=2)
        
        # Upload using media.py
        content_type = 'video/mp4' if save_path.lower().endswith('.mp4') else 'image/gif'
        try:
            uploaded_url = save_media(save_path, temp_path, content_type=content_type)
            print(f"Animation with reconstructions uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Animation with reconstructions saved locally to {temp_path}")
        
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save animation with reconstructions: {e}")
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        if save_path.lower().endswith('.mp4'):
            if check_ffmpeg_available():
                try:
                    anim.save(fallback_path, writer='ffmpeg', fps=2)
                    print(f"Animation with reconstructions saved locally to: {fallback_path}")
                except Exception as mp4_error:
                    print(f"MP4 save failed, falling back to GIF: {mp4_error}")
                    gif_path = fallback_path.replace('.mp4', '.gif')
                    anim.save(gif_path, writer='pillow', fps=2)
                    print(f"Animation with reconstructions saved locally as GIF: {gif_path}")
            else:
                print("ffmpeg not available, saving as GIF instead")
                gif_path = fallback_path.replace('.mp4', '.gif')
                anim.save(gif_path, writer='pillow', fps=2)
                print(f"Animation with reconstructions saved locally as GIF: {gif_path}")
        else:
            anim.save(fallback_path, writer='pillow', fps=2)
            print(f"Animation with reconstructions saved locally to: {fallback_path}")
    
    plt.close()


def create_probability_animation(
    animation_frames: List[Dict[str, Any]], 
    X_examples: np.ndarray, 
    y_examples: np.ndarray, 
    save_path: str = None, 
    title: str = "MNIST Clustering Probability Evolution",
    show_reconstructions: bool = False,
    reconstruction_frames: List[np.ndarray] = None
) -> None:
    """
    Create an animation showing probability evolution for 12 examples with sample images.
    
    Args:
        animation_frames: List of frames with probability data and cluster counts
        X_examples: Array of example images (12, 784)
        y_examples: Array of true labels for examples
        save_path: Path to save the animation (defaults to outputs/yy-mm-dd/mnist_clustering_probability_evolution.mp4)
        title: Title for the animation
        show_reconstructions: Whether to show reconstructed images below originals
        reconstruction_frames: List of reconstruction arrays for each frame (shape: frames x 12 x 784)
    """
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, "mnist_clustering_probability_evolution.mp4")
    
    # Adjust figure size and grid based on whether we show reconstructions
    if show_reconstructions:
        fig = plt.figure(figsize=(30, 20))
        # Create a 8x15 grid (8 rows, 15 columns) for reconstruction layout
        # Columns 0-11: samples (3 groups of 4 columns each: orig, recon, prob, prob)
        # Columns 12-14: cluster distribution (right side)
        gs = fig.add_gridspec(8, 15, hspace=0.4, wspace=0.2)
    else:
        fig = plt.figure(figsize=(24, 15))
        # Create a 4x12 grid (4 rows, 12 columns) for original layout
        # Columns 0-8: samples (3 groups of 3 columns each)
        # Columns 9-11: cluster distribution (right side)
        gs = fig.add_gridspec(4, 12, hspace=0.3, wspace=0.2)
    
    fig.suptitle(title, fontsize=16)
    
    # Set up the bar plots with sample images
    bars = []
    prob_axes = []
    text_objects = []
    recon_image_objects = []  # For reconstruction images (only used if show_reconstructions=True)
    
    # Validate reconstruction data if needed
    if show_reconstructions and reconstruction_frames is None:
        print("Warning: show_reconstructions=True but no reconstruction_frames provided. Disabling reconstructions.")
        show_reconstructions = False
    
    # Create the grid layout
    for i in range(12):  # 12 examples
        if show_reconstructions:
            # New layout: 2 rows per example (orig + recon), 4 columns per example
            row = (i // 3) * 2  # 0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6
            col_start = (i % 3) * 4  # 0, 4, 8, 0, 4, 8, 0, 4, 8, 0, 4, 8
            
            # Create original image (1 column)
            ax_orig = fig.add_subplot(gs[row, col_start])
            img = X_examples[i].reshape(28, 28)
            ax_orig.imshow(img, cmap='gray', interpolation='nearest', aspect='equal')
            ax_orig.set_title(f'Ex {i+1} (True: {y_examples[i]})', fontsize=8)
            ax_orig.axis('off')
            
            # Create reconstruction image (directly below original, 1 column)
            ax_recon = fig.add_subplot(gs[row + 1, col_start])
            initial_recon = np.zeros((28, 28))
            recon_im = ax_recon.imshow(initial_recon, cmap='gray', interpolation='nearest', aspect='equal', vmin=0, vmax=1)
            ax_recon.set_title(f'Recon C?', fontsize=8)
            ax_recon.axis('off')
            
            recon_image_objects.append({
                'image': recon_im,
                'axis': ax_recon
            })
            
            # Create probability axis (spans 3 columns, to the right of images)
            ax_prob = fig.add_subplot(gs[row:row+2, col_start + 1:col_start + 4])  # Span both rows
        else:
            # Original layout: 1 row per example, 3 columns per example
            row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
            col_start = (i % 3) * 3  # 0, 3, 6, 0, 3, 6, 0, 3, 6, 0, 3, 6
            
            # Create original image
            ax_orig = fig.add_subplot(gs[row, col_start])
            img = X_examples[i].reshape(28, 28)
            ax_orig.imshow(img, cmap='gray', interpolation='nearest', aspect='equal')
            ax_orig.set_title(f'Ex {i+1} (True: {y_examples[i]})', fontsize=8)
            ax_orig.axis('off')
            
            # Create probability axis (spans 2 columns)
            ax_prob = fig.add_subplot(gs[row, col_start + 1:col_start + 3])
        
        # Set up probability axis
        ax_prob.set_xlabel('Cluster Label')
        ax_prob.set_ylim(0, 100)  # Use percentage scale (0-100%)
        ax_prob.set_xlim(-0.5, 9.5)
        ax_prob.set_xticks(range(10))
        ax_prob.set_xticklabels([f'C{j}' for j in range(10)])
        ax_prob.spines['right'].set_visible(False)
        ax_prob.spines['top'].set_visible(False)
        
        prob_axes.append(ax_prob)
        
        # Initialize bars
        bar_heights = np.zeros(10)
        bars.append(ax_prob.bar(range(10), bar_heights, alpha=0.7, color='skyblue', width=0.8))
        
        # Initialize text objects for this axis
        text_objects.append([ax_prob.text(0, 0, '', ha='center', va='bottom', fontsize=7) for _ in range(10)])
    
    # Add cluster distribution plot on the right side
    if show_reconstructions:
        ax_dist = fig.add_subplot(gs[:, 12:15])  # Span all rows, last 3 columns
    else:
        ax_dist = fig.add_subplot(gs[:, 9:12])   # Span all rows, last 3 columns (original layout)
    
    ax_dist.set_title('Cluster Distribution (All Training Samples)', fontsize=12)
    ax_dist.set_xlabel('Cluster Label')
    ax_dist.set_ylabel('Number of Items')
    ax_dist.set_xlim(-0.5, 9.5)
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels([f'C{j}' for j in range(10)])
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)
    
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
                
                # Update reconstruction if enabled
                if show_reconstructions and reconstruction_frames is not None:
                    try:
                        # Use pre-calculated reconstruction for this frame and example
                        reconstruction = reconstruction_frames[frame_idx][i]  # Shape: (784,)
                        assigned_cluster = max_idx  # Use the most probable cluster
                        
                        # Update reconstruction image
                        recon_img = reconstruction.reshape(28, 28)
                        recon_image_objects[i]['image'].set_array(recon_img)
                        recon_image_objects[i]['axis'].set_title(f'Recon C{assigned_cluster}', fontsize=8)
                        
                    except Exception as e:
                        # If reconstruction fails, show placeholder
                        recon_image_objects[i]['image'].set_array(np.zeros((28, 28)))
                        recon_image_objects[i]['axis'].set_title(f'Recon Error', fontsize=8)
        
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
        
        if show_reconstructions:
            return bars + list(dist_bars) + [obj['image'] for obj in recon_image_objects]
        else:
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


def plot_closest_samples_to_centers(
    closest_samples: np.ndarray,
    y_train: np.ndarray,
    closest_indices: np.ndarray,
    run_uid: str,
    save_path: str = None
) -> None:
    """
    Plot the closest samples to each cluster center.
    
    Args:
        closest_samples: Array of closest samples (K, n_samples_per_cluster, 784)
        y_train: True labels for all training data
        closest_indices: Array of indices of closest samples (K, n_samples_per_cluster)
        run_uid: Unique identifier for this run
        save_path: Path to save the plot
    """
    K, n_samples_per_cluster = closest_samples.shape[:2]
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Closest Samples to Each Cluster Center (Run: {run_uid})', fontsize=16)
    
    # Create a grid: K rows (one per cluster), n_samples_per_cluster columns
    gs = fig.add_gridspec(K, n_samples_per_cluster, hspace=0.4, wspace=0.2)
    
    for k in range(K):
        for s in range(n_samples_per_cluster):
            ax = fig.add_subplot(gs[k, s])
            
            # Get the sample and its true label
            sample = closest_samples[k, s].reshape(28, 28)
            sample_idx = closest_indices[k, s]
            
            if sample_idx >= 0:  # Valid sample
                true_label = y_train[sample_idx]
                ax.imshow(sample, cmap='gray', interpolation='nearest')
                ax.set_title(f'C{k}-S{s+1} (Label: {true_label})', fontsize=10)
            else:  # No sample available (empty cluster)
                ax.imshow(np.zeros((28, 28)), cmap='gray')
                ax.set_title(f'C{k}-S{s+1} (Empty)', fontsize=10, color='red')
            
            ax.axis('off')
    
    # Add cluster labels on the left
    for k in range(K):
        fig.text(0.02, 1 - (k + 0.5) / K, f'Cluster {k}', 
                rotation=90, va='center', ha='center', fontsize=12, weight='bold')
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, f"mnist_clustering_closest_samples_{run_uid}.png")
    
    # Save plot with upload
    try:
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        
        try:
            uploaded_url = save_media(save_path, temp_path, content_type='image/png')
            print(f"Closest samples plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Closest samples plot saved locally to {temp_path}")
        
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save closest samples plot: {e}")
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
        print(f"Closest samples plot saved locally to: {fallback_path}")
    
    plt.close()


def create_closest_samples_animation(
    closest_samples_frames: List[np.ndarray],
    y_train: np.ndarray,
    closest_indices_frames: List[np.ndarray],
    save_path: str = None,
    title: str = "Closest Samples to Cluster Centers Evolution"
) -> None:
    """
    Create an animation showing how the closest samples to each cluster center evolve.
    
    Args:
        closest_samples_frames: List of closest samples arrays over time (frames, K, n_samples_per_cluster, 784)
        y_train: True labels for all training data
        closest_indices_frames: List of indices arrays over time (frames, K, n_samples_per_cluster)
        save_path: Path to save the animation
        title: Title for the animation
    """
    if not closest_samples_frames:
        print("No frames provided for closest samples animation")
        return
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, "mnist_clustering_closest_samples_evolution.mp4")
    
    K, n_samples_per_cluster = closest_samples_frames[0].shape[:2]
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16)
    
    # Create a grid: K rows (one per cluster), n_samples_per_cluster columns
    gs = fig.add_gridspec(K, n_samples_per_cluster, hspace=0.4, wspace=0.2)
    
    # Initialize image objects
    image_objects = []
    title_objects = []
    
    for k in range(K):
        image_row = []
        title_row = []
        for s in range(n_samples_per_cluster):
            ax = fig.add_subplot(gs[k, s])
            
            # Initialize with first frame
            sample = closest_samples_frames[0][k, s].reshape(28, 28)
            im = ax.imshow(sample, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            
            # Initialize title
            sample_idx = closest_indices_frames[0][k, s]
            if sample_idx >= 0:
                true_label = y_train[sample_idx]
                title_text = f'C{k}-S{s+1} (Label: {true_label})'
            else:
                title_text = f'C{k}-S{s+1} (Empty)'
            
            title_obj = ax.set_title(title_text, fontsize=10)
            ax.axis('off')
            
            image_row.append(im)
            title_row.append(title_obj)
        
        image_objects.append(image_row)
        title_objects.append(title_row)
    
    # Add cluster labels on the left
    for k in range(K):
        fig.text(0.02, 1 - (k + 0.5) / K, f'Cluster {k}', 
                rotation=90, va='center', ha='center', fontsize=12, weight='bold')
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    
    def animate(frame_idx):
        closest_samples = closest_samples_frames[frame_idx]
        closest_indices = closest_indices_frames[frame_idx]
        
        for k in range(K):
            for s in range(n_samples_per_cluster):
                # Update image
                sample = closest_samples[k, s].reshape(28, 28)
                image_objects[k][s].set_array(sample)
                
                # Update title
                sample_idx = closest_indices[k, s]
                if sample_idx >= 0:
                    true_label = y_train[sample_idx]
                    title_text = f'C{k}-S{s+1} (Label: {true_label})'
                    title_objects[k][s].set_color('black')
                else:
                    title_text = f'C{k}-S{s+1} (Empty)'
                    title_objects[k][s].set_color('red')
                
                title_objects[k][s].set_text(title_text)
        
        # Add frame info
        fig.text(0.02, 0.02, f'Frame {frame_idx + 1}/{len(closest_samples_frames)}', 
                transform=fig.transFigure, fontsize=12, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        return [im for row in image_objects for im in row]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(closest_samples_frames), 
        interval=500, blit=False, repeat=True
    )
    
    # Save animation with upload
    try:
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        
        # Determine writer based on file extension
        if save_path.lower().endswith('.mp4'):
            if check_ffmpeg_available():
                try:
                    anim.save(temp_path, writer='ffmpeg', fps=2)
                    print(f"MP4 animation saved to temporary file: {temp_path}")
                except Exception as mp4_error:
                    print(f"Failed to save MP4 with ffmpeg: {mp4_error}")
                    temp_path_gif = temp_path.replace('.mp4', '.gif')
                    anim.save(temp_path_gif, writer='pillow', fps=2)
                    temp_path = temp_path_gif
                    print(f"Fell back to GIF format: {temp_path}")
            else:
                print("ffmpeg not available, falling back to GIF format")
                temp_path_gif = temp_path.replace('.mp4', '.gif')
                anim.save(temp_path_gif, writer='pillow', fps=2)
                temp_path = temp_path_gif
        else:
            anim.save(temp_path, writer='pillow', fps=2)
        
        # Upload using media.py
        content_type = 'video/mp4' if save_path.lower().endswith('.mp4') else 'image/gif'
        try:
            uploaded_url = save_media(save_path, temp_path, content_type=content_type)
            print(f"Closest samples animation uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Closest samples animation saved locally to {temp_path}")
        
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save closest samples animation: {e}")
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        if save_path.lower().endswith('.mp4'):
            if check_ffmpeg_available():
                try:
                    anim.save(fallback_path, writer='ffmpeg', fps=2)
                    print(f"Closest samples animation saved locally to: {fallback_path}")
                except Exception as mp4_error:
                    print(f"MP4 save failed, falling back to GIF: {mp4_error}")
                    gif_path = fallback_path.replace('.mp4', '.gif')
                    anim.save(gif_path, writer='pillow', fps=2)
                    print(f"Closest samples animation saved locally as GIF: {gif_path}")
            else:
                print("ffmpeg not available, saving as GIF instead")
                gif_path = fallback_path.replace('.mp4', '.gif')
                anim.save(gif_path, writer='pillow', fps=2)
                print(f"Closest samples animation saved locally as GIF: {gif_path}")
        else:
            anim.save(fallback_path, writer='pillow', fps=2)
            print(f"Closest samples animation saved locally to: {fallback_path}")
    
    plt.close()


def create_enhanced_clustering_visualization(
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
    closest_samples_frames: List[np.ndarray] = None,
    closest_indices_frames: List[np.ndarray] = None,
    y_train: np.ndarray = None,
    final_closest_samples: np.ndarray = None,
    final_closest_indices: np.ndarray = None,
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
        closest_samples_frames: List of closest samples arrays over time (optional)
        closest_indices_frames: List of closest indices arrays over time (optional)
        y_train: True labels for all training data (needed for closest samples)
        final_closest_samples: Final closest samples to each cluster center (optional)
        final_closest_indices: Final closest indices (optional)
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
        anim_path = os.path.join(output_dir, f"mnist_clustering_probability_evolution_{run_uid}.mp4")
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
    
    # Create closest samples animation if data is provided
    if closest_samples_frames and closest_indices_frames and y_train is not None:
        closest_anim_path = os.path.join(output_dir, f"mnist_clustering_closest_samples_evolution_{run_uid}.mp4")
        create_closest_samples_animation(
            closest_samples_frames=closest_samples_frames,
            y_train=y_train,
            closest_indices_frames=closest_indices_frames,
            save_path=closest_anim_path,
            title="MNIST Clustering: Evolution of Closest Samples to Cluster Centers"
        )
        saved_files['closest_samples_animation'] = closest_anim_path
    
    # Create final closest samples plot if data is provided
    if final_closest_samples is not None and final_closest_indices is not None and y_train is not None:
        closest_plot_path = os.path.join(output_dir, f"mnist_clustering_final_closest_samples_{run_uid}.png")
        plot_closest_samples_to_centers(
            closest_samples=final_closest_samples,
            y_train=y_train,
            closest_indices=final_closest_indices,
            run_uid=run_uid,
            save_path=closest_plot_path
        )
        saved_files['final_closest_samples'] = closest_plot_path
    
    # Create EMA evolution plot if data is provided
    if ema_history is not None and samples_trained_per_batch is not None:
        ema_path = os.path.join(output_dir, f"mnist_clustering_ema_evolution_{run_uid}.png")
        plot_ema_evolution(ema_history, samples_trained_per_batch, ema_path)
        saved_files['ema_evolution'] = ema_path
    
    return saved_files


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
    Create all clustering visualizations in one function (legacy compatibility).
    
    This is the original function signature for backward compatibility.
    For new features like closest samples, use create_enhanced_clustering_visualization.
    """
    return create_enhanced_clustering_visualization(
        animation_frames=animation_frames,
        final_probabilities=final_probabilities,
        X_examples=X_examples,
        y_examples=y_examples,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        run_uid=run_uid,
        all_cluster_counts=all_cluster_counts,
        ema_history=ema_history,
        samples_trained_per_batch=samples_trained_per_batch,
        output_dir=output_dir
    )


def create_mnist_reconstruction_pair(fig, gs, row, col_start, img_data, recon_data, example_num, true_label, assigned_cluster):
    """
    Create an original + reconstructed MNIST image pair.
    
    Args:
        fig: matplotlib figure
        gs: GridSpec object
        row: row index
        col_start: starting column index
        img_data: original MNIST image data (784,)
        recon_data: reconstructed MNIST image data (784,)
        example_num: example number (1-based)
        true_label: true label for the image
        assigned_cluster: assigned cluster ID
        
    Returns:
        tuple: (original_axis, reconstruction_axis)
    """
    # Create original image subplot (spans 1 column)
    ax_orig = fig.add_subplot(gs[row, col_start])
    img = img_data.reshape(28, 28)
    ax_orig.imshow(img, cmap='gray', interpolation='nearest', aspect='equal')
    ax_orig.set_title(f'Ex {example_num} (True: {true_label})', fontsize=8)
    ax_orig.axis('off')
    
    # Create reconstructed image subplot (spans 1 column, next to original)
    ax_recon = fig.add_subplot(gs[row, col_start + 1])
    recon_img = recon_data.reshape(28, 28)
    ax_recon.imshow(recon_img, cmap='gray', interpolation='nearest', aspect='equal')
    ax_recon.set_title(f'Recon C{assigned_cluster}', fontsize=8)
    ax_recon.axis('off')
    
    return ax_orig, ax_recon


def create_clustering_visualization_generative(
    animation_frames: List[Dict[str, Any]],
    final_probabilities: np.ndarray,
    X_examples: np.ndarray,
    y_examples: np.ndarray,
    X_reconstructions: np.ndarray,
    train_losses: List[float],
    train_accuracies: List[float],
    run_uid: str,
    all_cluster_counts: np.ndarray,
    cluster_assignments: np.ndarray,
    reconstruction_frames: List[np.ndarray] = None,
    output_dir: str = None
) -> Dict[str, str]:
    """
    Create clustering visualizations with generative reconstructions.
    
    Args:
        animation_frames: List of animation frames
        final_probabilities: Final probability distributions (12, 10)
        X_examples: Example images (12, 784)
        y_examples: True labels for examples (12,)
        X_reconstructions: Reconstructed images using assigned cluster VAEs (12, 784)
        train_losses: Training loss history
        train_accuracies: Training accuracy history
        run_uid: Unique identifier for this run
        all_cluster_counts: Cluster distribution for all training samples (10,)
        cluster_assignments: Final cluster assignments for examples (12,)
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
    
    # Create probability evolution animation with reconstructions
    if animation_frames:
        anim_path = os.path.join(output_dir, f"mnist_clustering_probability_evolution_generative_{run_uid}.mp4")
        create_probability_animation(
            animation_frames=animation_frames,
            X_examples=X_examples,
            y_examples=y_examples,
            save_path=anim_path,
            title="MNIST Clustering: Probability Evolution with VAE Reconstructions",
            show_reconstructions=True,
            reconstruction_frames=reconstruction_frames
        )
        saved_files['animation'] = anim_path
    
    # Create training progress plot (same as before)
    progress_path = os.path.join(output_dir, f"mnist_clustering_training_progress_generative_{run_uid}.png")
    plot_training_progress(train_losses, train_accuracies, progress_path)
    saved_files['training_progress'] = progress_path
    
    # Create new generative visualization with original + reconstructed images
    generative_path = os.path.join(output_dir, f"mnist_clustering_generative_comparison_{run_uid}.png")
    plot_generative_comparison(
        final_probabilities=final_probabilities,
        X_examples=X_examples,
        y_examples=y_examples,
        X_reconstructions=X_reconstructions,
        cluster_assignments=cluster_assignments,
        run_uid=run_uid,
        all_cluster_counts=all_cluster_counts,
        save_path=generative_path
    )
    saved_files['generative_comparison'] = generative_path
    
    return saved_files


def plot_generative_comparison(
    final_probabilities: np.ndarray,
    X_examples: np.ndarray,
    y_examples: np.ndarray,
    X_reconstructions: np.ndarray,
    cluster_assignments: np.ndarray,
    run_uid: str,
    all_cluster_counts: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot comparison of original images with VAE reconstructions and probability distributions.
    
    Args:
        final_probabilities: Final probability distributions (12, 10)
        X_examples: Example images (12, 784)
        y_examples: True labels for examples (12,)
        X_reconstructions: Reconstructed images (12, 784)
        cluster_assignments: Final cluster assignments (12,)
        run_uid: Unique identifier for this run
        all_cluster_counts: Cluster distribution for all training samples (10,)
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(28, 20))
    fig.suptitle(f'VAE Clustering: Original vs Reconstructed Images (Run: {run_uid})', fontsize=16)
    
    # Create a grid: 4 rows x 15 columns
    # Columns 0-11: examples (4 groups of 3 examples each, 4 columns per example)
    # Columns 12-14: cluster distribution (right side)
    gs = fig.add_gridspec(4, 15, hspace=0.4, wspace=0.3)
    
    num_examples = len(X_examples)
    for i in range(num_examples):
        row = i // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
        example_in_row = i % 3  # 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
        col_start = example_in_row * 4  # 0, 4, 8, 0, 4, 8, 0, 4, 8, 0, 4, 8
        
        # Create original + reconstructed image pair (uses 2 columns)
        ax_orig, ax_recon = create_mnist_reconstruction_pair(
            fig, gs, row, col_start, X_examples[i], X_reconstructions[i], 
            i+1, y_examples[i], cluster_assignments[i]
        )
        
        # Create probability axis (spans 2 columns for more space)
        ax_prob = create_probability_axis(fig, gs, row, col_start + 2, remove_borders=True, span_columns=2)
        
        # Create bars with final probabilities (convert to percentages)
        probabilities_percent = final_probabilities[i] * 100
        bars = ax_prob.bar(range(10), probabilities_percent, alpha=0.7, color='lightcoral', width=0.8)
        
        # Highlight the assigned cluster (maximum probability)
        assigned_cluster = cluster_assignments[i]
        bars[assigned_cluster].set_color('red')
        bars[assigned_cluster].set_alpha(0.9)
        
        # Add probability values on bars (show as percentages with 1 decimal)
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
    
    # Add cluster distribution plot on the right side (using columns 12-14)
    ax_dist = fig.add_subplot(gs[:, 12:15])  # Span all rows, last 3 columns
    ax_dist.set_title('Final Cluster Distribution (All Training Samples)', fontsize=12)
    ax_dist.set_xlabel('Cluster Label')
    ax_dist.set_ylabel('Number of Items')
    ax_dist.set_xlim(-0.5, 9.5)
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels([f'C{j}' for j in range(10)])
    
    # Remove right and top axis borders
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)
    
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
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.4)
    
    # Use default save path if none provided
    if save_path is None:
        output_dir = get_default_output_dir()
        save_path = os.path.join(output_dir, f"mnist_clustering_generative_comparison_{run_uid}.png")
    
    # Save plot with upload
    try:
        # Save to temporary file in outputs directory first
        temp_filename = f"temp_{os.path.basename(save_path)}"
        temp_path = os.path.join(get_default_output_dir(), temp_filename)
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        
        # Upload using media.py
        try:
            uploaded_url = save_media(save_path, temp_path, content_type='image/png')
            print(f"Generative comparison plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"Generative comparison plot saved locally to {temp_path}")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save generative comparison plot: {e}")
        # Fall back to direct save in outputs directory
        fallback_path = os.path.join(get_default_output_dir(), os.path.basename(save_path))
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
        print(f"Generative comparison plot saved locally to: {fallback_path}")
    
    plt.close()
