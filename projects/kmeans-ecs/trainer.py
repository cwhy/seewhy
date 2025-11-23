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


class ParamUpdateProtocol(Protocol):
    """Protocol interface for parameter update objects."""
    
    def train_batch(
        self,
        params: Dict[str, Array],
        batch: Dict[str, Array]
    ) -> Tuple[Dict[str, Array], Array]:
        """
        Execute a training step.
        
        Args:
            params: Model parameters
            batch: Dictionary of JAX arrays (e.g., {'X': Array, 'y': Array})
        
        Returns:
            (updated_params, loss)
        """
        ...


class Trainer:
    """
    ECS-style component for managing the training loop.
    Handles epoch/batch boundaries and calls registered callbacks.
    """
    
    def __init__(
        self,
        sampler: SamplerProtocol,
        param_update: ParamUpdateProtocol,
        on_epoch_start: Optional[Callable[[int], None]] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, Array]], None]] = None,
        on_batch_start: Optional[Callable[[BatchMetadata], None]] = None,
        on_batch_end: Optional[Callable[[BatchMetadata, float], None]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            sampler: Batch sampler (must implement SamplerProtocol)
            param_update: Parameter update object (must implement ParamUpdateProtocol)
            on_epoch_start: Callback called at epoch start (epoch: int)
            on_epoch_end: Callback called at epoch end (epoch: int, params: Dict[str, Array])
            on_batch_start: Callback called at batch start (metadata: BatchMetadata)
            on_batch_end: Callback called at batch end (metadata: BatchMetadata, loss: float)
        """
        self.sampler = sampler
        self.param_update = param_update
        
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
        current_epoch = -1
        
        for batch, metadata in self.sampler:
            # Handle epoch start
            if metadata.is_epoch_start:
                current_epoch = metadata.epoch
                if self.on_epoch_start:
                    self.on_epoch_start(current_epoch)
            
            # Handle batch start
            if self.on_batch_start:
                self.on_batch_start(metadata)
            
            # Training step
            params, loss = self.param_update.train_batch(params, batch)
            
            # Handle batch end
            if self.on_batch_end:
                self.on_batch_end(metadata, float(loss))
            
            # Handle epoch end
            if metadata.is_epoch_end:
                if self.on_epoch_end:
                    self.on_epoch_end(current_epoch, params)
        
        return params

