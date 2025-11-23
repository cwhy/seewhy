"""
Trainer Component - ECS-style pluggable component for managing the training loop.

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.

Examples:
- Pure: compute_avg_loss()
- With side effects: train_() (mutates params, calls callbacks)
"""
from typing import Protocol, Dict, Tuple, Callable, Optional, List, Iterator
from jax import Array
from sampler import BatchMetadata


class SamplerProtocol(Protocol):
    """Protocol interface for batch samplers."""
    
    def __iter__(self) -> Iterator[Tuple[Dict[str, Array], BatchMetadata]]:
        """Iterator that yields (batch_dict, metadata)."""
        ...


class TrainStepProtocol(Protocol):
    """Protocol interface for training step functions."""
    
    def __call__(
        self,
        params: Dict[str, Array],
        batch: Dict[str, Array],
        **kwargs
    ) -> Tuple[Dict[str, Array], Array]:
        """
        Execute a training step.
        
        Args:
            params: Model parameters
            batch: Dictionary of JAX arrays (e.g., {'X': Array, 'y': Array})
            **kwargs: Additional arguments (e.g., lr for learning rate)
        
        Returns:
            (updated_params, loss)
        """
        ...


def compute_avg_loss(losses: List[float]) -> float:
    """Pure function to compute average loss."""
    return float(sum(losses) / len(losses)) if losses else 0.0


class Trainer:
    """
    ECS-style component for managing the training loop.
    Handles epoch/batch boundaries and calls registered callbacks.
    """
    
    def __init__(
        self,
        sampler: SamplerProtocol,
        train_step: TrainStepProtocol,
        learning_rate: float,
        on_epoch_start: Optional[Callable[[int], None]] = None,
        on_epoch_end: Optional[Callable[[int, float, Dict[str, Array]], None]] = None,
        on_batch_start: Optional[Callable[[BatchMetadata], None]] = None,
        on_batch_end: Optional[Callable[[BatchMetadata, float], None]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            sampler: Batch sampler (must implement SamplerProtocol)
            train_step: Training step function (must implement TrainStepProtocol)
            learning_rate: Learning rate for training
            on_epoch_start: Callback called at epoch start (epoch: int)
            on_epoch_end: Callback called at epoch end (epoch: int, avg_loss: float, params: Dict[str, Array])
            on_batch_start: Callback called at batch start (metadata: BatchMetadata)
            on_batch_end: Callback called at batch end (metadata: BatchMetadata, loss: float)
        """
        self.sampler = sampler
        self.train_step = train_step
        self.learning_rate = learning_rate
        
        # Callbacks
        self.on_epoch_start = on_epoch_start
        self.on_epoch_end = on_epoch_end
        self.on_batch_start = on_batch_start
        self.on_batch_end = on_batch_end
    
    def train_(self, params: Dict[str, Array]) -> Dict[str, Array]:
        """
        Execute the training loop. Mutates params and calls callbacks.
        
        Args:
            params: Initial model parameters
            
        Returns:
            Final trained parameters
        """
        epoch_losses: List[float] = []
        current_epoch = -1
        
        for batch, metadata in self.sampler:
            # Handle epoch start
            if metadata.is_epoch_start:
                current_epoch = metadata.epoch
                epoch_losses = []
                if self.on_epoch_start:
                    self.on_epoch_start(current_epoch)
            
            # Handle batch start
            if self.on_batch_start:
                self.on_batch_start(metadata)
            
            # Training step
            params, loss = self.train_step(params, batch, lr=self.learning_rate)
            epoch_losses.append(float(loss))
            
            # Handle batch end
            if self.on_batch_end:
                self.on_batch_end(metadata, float(loss))
            
            # Handle epoch end
            if metadata.is_epoch_end:
                avg_loss = compute_avg_loss(epoch_losses)
                if self.on_epoch_end:
                    self.on_epoch_end(current_epoch, avg_loss, params)
        
        return params

