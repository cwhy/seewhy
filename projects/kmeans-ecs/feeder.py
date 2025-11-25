"""
Feeder Component - ECS-style pluggable component for managing the training loop.

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.

Examples:
- Pure: compute_avg_loss()
- With side effects: loop_() (mutates params, calls callbacks)
"""
from typing import Optional, Protocol, Dict, Tuple, Callable, Iterator, NamedTuple, Literal
from jax import Array
from sampler import BatchMetadata


class LoopStates(NamedTuple):
    """State of the training loop."""
    params: Dict[str, Array]
    outputs: Dict[str, Array]
    metadata: BatchMetadata

class ParamUpdateResult(NamedTuple):
    """Result of parameter update."""
    params: Dict[str, Array]
    outputs: Dict[str, Array]


TriggerKey = Literal['epoch_start', 'epoch_end', 'batch_start', 'batch_end']
TriggerCallbacks = Dict[TriggerKey, Callable[[LoopStates], None]]


class SamplerProtocol(Protocol):
    """Protocol interface for batch samplers."""
    
    def __iter__(self) -> Iterator[Tuple[Dict[str, Array], BatchMetadata]]:
        """Iterator that yields (batch_dict, metadata)."""
        ...


class ParamUpdateProtocol(Protocol):
    """Protocol interface for parameter update objects."""
    
    def batch_update(
        self,
        params: Dict[str, Array],
        batch: Dict[str, Array]
    ) -> ParamUpdateResult:
        """
        Execute a training step.
        
        Args:
            params: Model parameters
            batch: Dictionary of JAX arrays (e.g., {'X': Array, 'y': Array})
        
        Returns:
            LoopStates with updated_params and outputs (loss is a key in outputs)
        """
        ...



class Feeder:
    """
    ECS-style component for managing the training loop.
    Handles epoch/batch boundaries and calls registered callbacks.
    """
    
    def __init__(
        self,
        sampler: SamplerProtocol,
        param_update: ParamUpdateProtocol,
        triggers: Optional[TriggerCallbacks] = None
    ):
        """
        Initialize the feeder.
        
        Args:
            sampler: Batch sampler (must implement SamplerProtocol)
            param_update: Parameter update object (must implement ParamUpdateProtocol)
            triggers: Dictionary of trigger callbacks. Keys: 'epoch_start', 'epoch_end', 
                     'batch_start', 'batch_end'. Values: Callable[[LoopStates], None]
        """
        self.sampler = sampler
        self.param_update = param_update
        
        # Callbacks from triggers dictionary
        self.triggers = triggers or {}
    
    def loop_(self, params: Dict[str, Array]) -> Dict[str, Array]:
        """
        Execute the training loop. Mutates params and calls callbacks.
        
        Args:
            params: Initial model parameters
            
        Returns:
            Final trained parameters
        """
        for batch, metadata in self.sampler:
            # Handle epoch start
            if metadata.is_epoch_start:
                if 'epoch_start' in self.triggers:
                    states = LoopStates(params=params, outputs={}, metadata=metadata)
                    self.triggers['epoch_start'](states)
            
            # Handle batch start
            if 'batch_start' in self.triggers:
                states = LoopStates(params=params, outputs={}, metadata=metadata)
                self.triggers['batch_start'](states)
            
            # Training step
            result = self.param_update.batch_update(params, batch)
            params = result.params
            
            # Create states with metadata for batch/epoch end callbacks
            states_with_metadata = LoopStates(
                params=result.params,
                outputs=result.outputs,
                metadata=metadata
            )
            
            # Handle batch end
            if 'batch_end' in self.triggers:
                self.triggers['batch_end'](states_with_metadata)
            
            # Handle epoch end
            if metadata.is_epoch_end:
                if 'epoch_end' in self.triggers:
                    self.triggers['epoch_end'](states_with_metadata)
        
        return params

