import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, Any, List, Union, Optional
from jax import Array
import numpy as np
import matplotlib.animation as animation

# Add the parent directory to the path to import from lib
import os
import sys
from lib.media import save_matplotlib_figure, save_media


def plot_training_progress(
    train_losses: List[float], 
    train_accuracies: List[float], 
    save_path: str,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 5),
    dpi: int = 150
) -> None:
    """
    Plot training progress (loss and accuracy) and save the figure.
    
    Args:
        train_losses: List of training loss values
        train_accuracies: List of training accuracy values
        save_path: Path where to save the plot
        title: Optional title for the overall figure
        figsize: Figure size as (width, height)
        dpi: DPI for the saved image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save using media.py
    save_matplotlib_figure(save_path, fig, format='png', dpi=dpi)
    plt.close()


def plot_sample_predictions(
    params: Dict[str, Array], 
    X: Array, 
    y: Array, 
    mlp_forward_fn: callable,
    num_samples: int = 5, 
    save_path: str = "sample_predictions.png",
    img_dims: tuple[int, int] = (28, 28),
    figsize: Optional[tuple[int, int]] = None,
    dpi: int = 150
) -> None:
    """
    Plot sample predictions and save the figure.
    
    Args:
        params: Model parameters dictionary
        X: Input data array
        y: True labels array
        mlp_forward_fn: Function to perform forward pass through the model
        num_samples: Number of samples to visualize
        save_path: Path where to save the plot
        img_dims: Dimensions of the images (height, width)
        figsize: Figure size as (width, height), auto-calculated if None
        dpi: DPI for the saved image
    """
    # Get predictions for first few samples
    sample_x = X[:num_samples]
    sample_y = y[:num_samples]
    
    logits = mlp_forward_fn(params, sample_x)
    predictions = jnp.argmax(logits, axis=1)
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (3 * num_samples, 3)
    
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Reshape image back to original dimensions
        img = sample_x[i].reshape(img_dims)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {sample_y[i]}\nPred: {predictions[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save using media.py
    save_matplotlib_figure(save_path, fig, format='png', dpi=dpi)
    plt.close()


def plot_confusion_matrix(
    y_true: Array, 
    y_pred: Array, 
    class_names: Optional[List[str]] = None,
    save_path: str = "confusion_matrix.png",
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 150
) -> None:
    """
    Plot confusion matrix and save the figure.
    
    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        class_names: List of class names for labeling
        save_path: Path where to save the plot
        figsize: Figure size as (width, height)
        dpi: DPI for the saved image
    """
    from sklearn.metrics import confusion_matrix
    
    # Convert to numpy arrays if they're JAX arrays
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a nice-looking heatmap using matplotlib
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontweight='bold')
    
    # Set labels
    if class_names:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
    else:
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xticklabels(range(cm.shape[1]))
        ax.set_yticklabels(range(cm.shape[0]))
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', pad=20)
    
    # Add grid lines
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    # Save using media.py
    save_matplotlib_figure(save_path, fig, format='png', dpi=dpi)
    plt.close()


def plot_learning_curves(
    train_losses: List[float],
    train_accuracies: List[float],
    val_losses: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: str = "learning_curves.png",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 5),
    dpi: int = 150
) -> None:
    """
    Plot comprehensive learning curves including validation metrics if available.
    
    Args:
        train_losses: List of training loss values
        train_accuracies: List of training accuracy values
        val_losses: Optional list of validation loss values
        val_accuracies: Optional list of validation accuracy values
        save_path: Path where to save the plot
        title: Optional title for the overall figure
        figsize: Figure size as (width, height)
        dpi: DPI for the saved image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    if val_losses is not None:
        ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    if val_accuracies is not None:
        ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save using media.py
    save_matplotlib_figure(save_path, fig, format='png', dpi=dpi)
    plt.close()


def plot_section_accuracies(
    section_accuracies: List[float],
    section_stats: Dict[str, float],
    save_path: str = "section_accuracies.png",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150
) -> None:
    """
    Plot training section accuracies with statistics.
    
    Args:
        section_accuracies: List of accuracy values for each section
        section_stats: Dictionary containing statistics (mean, std, min, max, median)
        save_path: Path where to save the plot
        title: Optional title for the overall figure
        figsize: Figure size as (width, height)
        dpi: DPI for the saved image
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Section accuracies over sections
    ax1.plot(section_accuracies, 'b-', alpha=0.7, linewidth=1)
    ax1.axhline(y=section_stats['mean'], color='r', linestyle='--', 
                label=f"Mean: {section_stats['mean']:.4f}")
    ax1.axhline(y=section_stats['median'], color='g', linestyle='--', 
                label=f"Median: {section_stats['median']:.4f}")
    ax1.fill_between(range(len(section_accuracies)), 
                     section_stats['mean'] - section_stats['std'],
                     section_stats['mean'] + section_stats['std'],
                     alpha=0.2, color='r', label=f"±1 std: {section_stats['std']:.4f}")
    
    ax1.set_title('Training Section Accuracies')
    ax1.set_xlabel('Section Index')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of section accuracies
    ax2.hist(section_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=section_stats['mean'], color='r', linestyle='--', 
                label=f"Mean: {section_stats['mean']:.4f}")
    ax2.axvline(x=section_stats['median'], color='g', linestyle='--', 
                label=f"Median: {section_stats['median']:.4f}")
    
    ax2.set_title('Distribution of Section Accuracies')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Min: {section_stats['min']:.4f}\nMax: {section_stats['max']:.4f}\nStd: {section_stats['std']:.4f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save using media.py
    save_matplotlib_figure(save_path, fig, format='png', dpi=dpi)
    plt.close() 


def create_section_accuracies_animation(
    animation_frames: List[Dict[str, Any]],
    save_path: str = "section_accuracies_animation.gif",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
    fps: int = 10
) -> None:
    """
    Create an animation showing how training section accuracies change during training.
    
    Args:
        animation_frames: List of dictionaries containing frame data:
            - 'section_accuracies': List of accuracy values for each section
            - 'section_stats': Dictionary with statistics
            - 'epoch': Current epoch number
            - 'batch': Current batch number (optional)
            - 'total_batches': Total number of batches (optional)
        save_path: Path where to save the animation
        title: Optional title for the animation
        figsize: Figure size as (width, height)
        dpi: DPI for the saved animation
        fps: Frames per second for the animation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    def animate(frame_idx):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        frame_data = animation_frames[frame_idx]
        section_accuracies = frame_data['section_accuracies']
        section_stats = frame_data['section_stats']
        epoch = frame_data.get('epoch', 0)
        batch = frame_data.get('batch', 0)
        total_batches = frame_data.get('total_batches', 0)
        
        # Plot 1: Section accuracies over sections
        ax1.plot(section_accuracies, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=section_stats['mean'], color='r', linestyle='--', 
                    label=f"Mean: {section_stats['mean']:.4f}")
        ax1.axhline(y=section_stats['median'], color='g', linestyle='--', 
                    label=f"Median: {section_stats['median']:.4f}")
        ax1.fill_between(range(len(section_accuracies)), 
                         section_stats['mean'] - section_stats['std'],
                         section_stats['mean'] + section_stats['std'],
                         alpha=0.2, color='r', label=f"±1 std: {section_stats['std']:.4f}")
        
        ax1.set_title(f'Training Section Accuracies - Epoch {epoch}, Batch {batch}/{total_batches}')
        ax1.set_xlabel('Section Index')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)  # Set consistent y-axis limits
        
        # Plot 2: Histogram of section accuracies
        ax2.hist(section_accuracies, bins=min(20, len(section_accuracies)), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=section_stats['mean'], color='r', linestyle='--', 
                    label=f"Mean: {section_stats['mean']:.4f}")
        ax2.axvline(x=section_stats['median'], color='g', linestyle='--', 
                    label=f"Median: {section_stats['median']:.4f}")
        
        ax2.set_title('Distribution of Section Accuracies')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)  # Set consistent x-axis limits
        
        # Add statistics text
        stats_text = f"Min: {section_stats['min']:.4f}\nMax: {section_stats['max']:.4f}\nStd: {section_stats['std']:.4f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if title:
            fig.suptitle(f"{title} - Frame {frame_idx + 1}/{len(animation_frames)}", fontsize=16)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(animation_frames), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    # Save animation
    anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    
    print(f"Animation saved to {save_path}")


def create_section_accuracies_animation_mp4(
    animation_frames: List[Dict[str, Any]],
    save_path: str = "section_accuracies_animation.mp4",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
    fps: int = 10
) -> None:
    """
    Create an MP4 animation showing how training section accuracies change during training.
    
    Args:
        animation_frames: List of dictionaries containing frame data
        save_path: Path where to save the MP4 animation
        title: Optional title for the animation
        figsize: Figure size as (width, height)
        dpi: DPI for the saved animation
        fps: Frames per second for the animation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    def animate(frame_idx):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        frame_data = animation_frames[frame_idx]
        section_accuracies = frame_data['section_accuracies']
        section_stats = frame_data['section_stats']
        epoch = frame_data.get('epoch', 0)
        batch = frame_data.get('batch', 0)
        total_batches = frame_data.get('total_batches', 0)
        
        # Plot 1: Section accuracies over sections
        ax1.plot(section_accuracies, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=section_stats['mean'], color='r', linestyle='--', 
                    label=f"Mean: {section_stats['mean']:.4f}")
        ax1.axhline(y=section_stats['median'], color='g', linestyle='--', 
                    label=f"Median: {section_stats['median']:.4f}")
        ax1.fill_between(range(len(section_accuracies)), 
                         section_stats['mean'] - section_stats['std'],
                         section_stats['mean'] + section_stats['std'],
                         alpha=0.2, color='r', label=f"±1 std: {section_stats['std']:.4f}")
        
        ax1.set_title(f'Training Section Accuracies - Epoch {epoch}, Batch {batch}/{total_batches}')
        ax1.set_xlabel('Section Index')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)  # Set consistent y-axis limits
        
        # Plot 2: Histogram of section accuracies
        ax2.hist(section_accuracies, bins=min(20, len(section_accuracies)), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=section_stats['mean'], color='r', linestyle='--', 
                    label=f"Mean: {section_stats['mean']:.4f}")
        ax2.axvline(x=section_stats['median'], color='g', linestyle='--', 
                    label=f"Median: {section_stats['median']:.4f}")
        
        ax2.set_title('Distribution of Section Accuracies')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)  # Set consistent x-axis limits
        
        # Add statistics text
        stats_text = f"Min: {section_stats['min']:.4f}\nMax: {section_stats['max']:.4f}\nStd: {section_stats['std']:.4f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if title:
            fig.suptitle(f"{title} - Frame {frame_idx + 1}/{len(animation_frames)}", fontsize=16)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(animation_frames), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    # Save animation as MP4
    try:
        # Save to temporary file first
        temp_path = f"temp_{save_path}"
        anim.save(temp_path, writer='ffmpeg', fps=fps, dpi=dpi)
        
        # Upload using media.py
        try:
            uploaded_url = save_media(save_path, temp_path, content_type='video/mp4')
            print(f"MP4 animation uploaded: {uploaded_url}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            print(f"MP4 animation saved locally to {temp_path}")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Failed to save MP4 animation: {e}")
        print("Falling back to GIF format...")
        gif_path = save_path.replace('.mp4', '.gif')
        temp_gif_path = f"temp_{gif_path}"
        
        try:
            anim.save(temp_gif_path, writer='pillow', fps=fps, dpi=dpi)
            
            # Upload GIF using media.py
            try:
                uploaded_url = save_media(gif_path, temp_gif_path, content_type='image/gif')
                print(f"GIF animation uploaded: {uploaded_url}")
            except Exception as upload_error:
                print(f"Upload failed: {upload_error}")
                print(f"GIF animation saved locally to {temp_gif_path}")
            
            # Clean up temp file
            try:
                os.remove(temp_gif_path)
            except:
                pass
                
        except Exception as gif_error:
            print(f"Failed to save GIF animation: {gif_error}")
    
    plt.close() 