"""
Loss Monitoring Component - ECS-style pluggable component for tracking and visualizing training loss.

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.

Examples:
- Pure: format_accuracy_str(), should_log_batch(), create_plot_data(), get_epoch_losses(), plot_loss()
- With side effects: record_batch_loss_() (mutates state, logs), finalize_and_plot_() (file I/O, plotting)
- Methods with side effects: All methods ending with _ mutate internal state or have side effects
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional, Dict, Tuple
import os
import logging
from datetime import datetime
from shared_lib.media import save_media

# Use ggplot style for cleaner, more professional plots
plt.style.use('ggplot')

# Use Noto Sans font
plt.rcParams['font.family'] = 'Noto Sans'

# Override some ggplot defaults for better readability
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2.5

logger = logging.getLogger(__name__)


def format_accuracy_str(train_acc: Optional[float], test_acc: Optional[float]) -> str:
    """Pure function to format accuracy string."""
    acc_str = ""
    if train_acc is not None:
        acc_str += f", Train Acc = {train_acc:.4f}"
    if test_acc is not None:
        acc_str += f", Test Acc = {test_acc:.4f}"
    return acc_str


def should_log_batch(batch_idx: int, log_interval: int) -> bool:
    """Pure function to determine if a batch should be logged."""
    return (batch_idx + 1) % log_interval == 0


def create_plot_data(epoch_losses: List[float]) -> Tuple[List[int], List[float]]:
    """Pure function to create plot data from epoch losses."""
    num_epochs = len(epoch_losses)
    epochs = list(range(1, num_epochs + 1))
    return epochs, epoch_losses


def create_accuracy_dict(train_acc: Optional[float], test_acc: Optional[float]) -> Dict[str, float]:
    """Pure function to create accuracy dictionary."""
    acc_dict = {}
    if train_acc is not None:
        acc_dict['train'] = float(train_acc)
    if test_acc is not None:
        acc_dict['test'] = float(test_acc)
    return acc_dict


def compute_avg_loss(losses: List[float]) -> float:
    """Pure function to compute average loss."""
    return float(sum(losses) / len(losses)) if losses else 0.0


def plot_loss(epoch_losses: List[float], plot_title: str = "Training Loss") -> Figure:
    """
    Pure function to create a loss plot figure using subplots.
    Uses ggplot style for clean, professional appearance.
    
    Args:
        epoch_losses: List of epoch-average losses
        plot_title: Title for the loss plot
        
    Returns:
        matplotlib Figure object with the plot
    """
    epochs, losses = create_plot_data(epoch_losses)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, losses, 'b-', label='Loss', marker='o', markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(plot_title)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_accuracy(epoch_accuracies: List[Dict[str, float]], plot_title: str = "Training Accuracy") -> Figure:
    """
    Pure function to create an accuracy plot figure.
    Uses ggplot style for clean, professional appearance.
    
    Args:
        epoch_accuracies: List of dictionaries with 'train' and/or 'test' accuracy values
        plot_title: Title for the accuracy plot
        
    Returns:
        matplotlib Figure object with the plot
    """
    num_epochs = len(epoch_accuracies)
    epochs = list(range(1, num_epochs + 1))
    
    # Extract train and test accuracies, filtering out None values
    train_data = [(epochs[i], acc_dict['train']) for i, acc_dict in enumerate(epoch_accuracies) if 'train' in acc_dict and acc_dict['train'] is not None]
    test_data = [(epochs[i], acc_dict['test']) for i, acc_dict in enumerate(epoch_accuracies) if 'test' in acc_dict and acc_dict['test'] is not None]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if train_data:
        train_epochs, train_accs = zip(*train_data)
        ax.plot(train_epochs, train_accs, 'g-', label='Train Accuracy', 
                marker='s', markersize=5, alpha=0.8)
    if test_data:
        test_epochs, test_accs = zip(*test_data)
        ax.plot(test_epochs, test_accs, 'r-', label='Test Accuracy', 
                marker='^', markersize=5, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(plot_title)
    ax.legend()
    ax.set_ylim((0, 1.05))  # Accuracy is between 0 and 1
    
    plt.tight_layout()
    
    return fig


class LossMonitor:
    """
    ECS-style component for monitoring training loss.
    Can be plugged into any training loop to track and visualize losses.
    """
    
    def __init__(
        self,
        log_interval: int = 1,
        plot_title: str = "Training Loss",
        plot_filename: str = "loss_plot.png",
        accuracy_plot_filename: str = "accuracy_plot.png"
    ):
        """
        Initialize the loss monitor component.
        
        Args:
            log_interval: Log every N batches (1 = every batch, 10 = every 10 batches, etc.)
            plot_title: Title for the loss plot
            plot_filename: Filename to save the loss plot (will be saved using save_media)
            accuracy_plot_filename: Filename to save the accuracy plot (will be saved using save_media)
        """
        self.log_interval = log_interval
        self.plot_title = plot_title
        self.plot_filename = plot_filename
        self.accuracy_plot_filename = accuracy_plot_filename
        
        # Internal state
        self.batch_losses: List[float] = []
        self.epoch_losses: List[float] = []
        self.epoch_accuracies: List[Dict[str, float]] = []  # Can store train/test accuracies
        self.current_epoch = 0
        self.current_batch = 0
        self.current_epoch_batch_losses: List[float] = []  # Track batch losses for current epoch
    
    def record_batch_loss_(self, loss: float, batch_idx: int, verbose: bool = False) -> None:
        """
        Record loss for a batch. Mutates internal state.
        
        Args:
            loss: Loss value for this batch
            batch_idx: Current batch index
            verbose: Whether to log batch-level information
        """
        loss_val = float(loss)
        self.batch_losses.append(loss_val)
        self.current_epoch_batch_losses.append(loss_val)
        self.current_batch = batch_idx + 1
        
        if verbose and should_log_batch(batch_idx, self.log_interval):
            logger.info(f"  Batch {batch_idx + 1}: Loss = {loss:.4f}")
    
    def start_epoch_tracking_(self, epoch: int) -> None:
        """Start tracking a new epoch. Mutates internal state."""
        self.current_epoch = epoch
        self.current_batch = 0
        self.current_epoch_batch_losses = []  # Reset batch losses for new epoch
    
    def compute_current_epoch_avg_loss(self) -> float:
        """
        Compute average loss for the current epoch from batch losses.
        Pure function relative to current_epoch_batch_losses.
        
        Returns:
            Average loss for the current epoch
        """
        return compute_avg_loss(self.current_epoch_batch_losses)
    
    def record_epoch_loss_(
        self,
        epoch: int,
        train_acc: Optional[float] = None,
        test_acc: Optional[float] = None,
        verbose: bool = True
    ) -> None:
        """
        Record loss and metrics for an epoch. Computes avg_loss from batch losses.
        Mutates internal state.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_acc: Training accuracy (optional)
            test_acc: Test accuracy (optional)
            verbose: Whether to log epoch-level information
        """
        avg_loss = self.compute_current_epoch_avg_loss()
        self.epoch_losses.append(avg_loss)
        acc_dict = create_accuracy_dict(train_acc, test_acc)
        self.epoch_accuracies.append(acc_dict)
        
        if verbose:
            acc_str = format_accuracy_str(train_acc, test_acc)
            logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}{acc_str}")
    
    def log_loss_acc_(
        self,
        epoch: int,
        train_acc: Optional[float] = None,
        test_acc: Optional[float] = None
    ) -> None:
        """
        Log loss and accuracy for an epoch using logger. Computes avg_loss from batch losses.
        Has side effects (logging).
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_acc: Training accuracy (optional)
            test_acc: Test accuracy (optional)
        """
        avg_loss = self.compute_current_epoch_avg_loss()
        acc_str = format_accuracy_str(train_acc, test_acc)
        logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}{acc_str}")
    
    def finalize_and_plot_(self) -> None:
        """
        Finalize training and generate loss and accuracy plots. Has side effects (file I/O, plotting).
        
        Raises:
            ValueError: If no epoch losses have been recorded
        """
        if len(self.epoch_losses) == 0:
            raise ValueError("No epoch losses recorded. Cannot generate plot.")
        
        self._plot_loss_()
        
        # Plot accuracy if we have accuracy data
        if len(self.epoch_accuracies) > 0 and any(acc_dict for acc_dict in self.epoch_accuracies):
            self._plot_accuracy_()
    
    def _plot_loss_(self) -> None:
        """
        Internal method to generate and save the loss plot. Has side effects (file I/O, plotting).
        
        Raises:
            OSError: If unable to create output directory or save file
            Exception: If save_media fails
        """
        # Create plot using pure function
        fig = plot_loss(self.epoch_losses, self.plot_title)
        
        # Set up output directory
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        date_str = datetime.now().strftime('%y-%m-%d')
        date_dir = os.path.join(outputs_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Save to temporary file first
        temp_filename = f"temp_{os.path.basename(self.plot_filename)}"
        temp_path = os.path.join(date_dir, temp_filename)
        
        try:
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            plt.close(fig)
            raise OSError(f"Failed to save plot to temporary file: {e}") from e
        
        # Upload using save_media
        try:
            uploaded_url = save_media(self.plot_filename, temp_path, content_type='image/png')
            logger.info(f"Loss plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            plt.close(fig)
            # Clean up temp file before raising
            try:
                os.remove(temp_path)
            except:
                pass
            raise Exception(f"Failed to upload loss plot: {upload_error}") from upload_error
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to remove temporary file {temp_path}: {cleanup_error}")
        
        plt.close(fig)
    
    def _plot_accuracy_(self) -> None:
        """
        Internal method to generate and save the accuracy plot. Has side effects (file I/O, plotting).
        
        Raises:
            OSError: If unable to create output directory or save file
            Exception: If save_media fails
        """
        # Create plot using pure function
        accuracy_title = self.plot_title.replace("Loss", "Accuracy")
        fig = plot_accuracy(self.epoch_accuracies, accuracy_title)
        
        # Set up output directory
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        date_str = datetime.now().strftime('%y-%m-%d')
        date_dir = os.path.join(outputs_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Save to temporary file first
        temp_filename = f"temp_{os.path.basename(self.accuracy_plot_filename)}"
        temp_path = os.path.join(date_dir, temp_filename)
        
        try:
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            plt.close(fig)
            raise OSError(f"Failed to save accuracy plot to temporary file: {e}") from e
        
        # Upload using save_media
        try:
            uploaded_url = save_media(self.accuracy_plot_filename, temp_path, content_type='image/png')
            logger.info(f"Accuracy plot uploaded: {uploaded_url}")
        except Exception as upload_error:
            plt.close(fig)
            # Clean up temp file before raising
            try:
                os.remove(temp_path)
            except:
                pass
            raise Exception(f"Failed to upload accuracy plot: {upload_error}") from upload_error
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to remove temporary file {temp_path}: {cleanup_error}")
        
        plt.close(fig)
    
    def get_epoch_losses(self) -> List[float]:
        """Get the list of epoch-average losses."""
        return self.epoch_losses.copy()
    
    def get_batch_losses(self) -> List[float]:
        """Get the list of all batch losses."""
        return self.batch_losses.copy()
    
    def get_latest_loss(self) -> Optional[float]:
        """Get the most recent epoch loss."""
        return self.epoch_losses[-1] if self.epoch_losses else None
    
    def reset_(self) -> None:
        """Reset the monitor state (useful for multiple training runs). Mutates internal state."""
        self.batch_losses.clear()
        self.epoch_losses.clear()
        self.epoch_accuracies.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.current_epoch_batch_losses.clear()

