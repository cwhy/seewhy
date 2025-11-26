"""
Type definitions for training components.

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.
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
            ParamUpdateResult with updated params and outputs (loss is a key in outputs)
        """
        ...

