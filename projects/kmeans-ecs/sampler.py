"""
Batch Sampler Component - ECS-style pluggable component for generating training batches with metadata.

CODING CONVENTION:
------------------
Functions with side effects or mutations have an underscore suffix (_).
Pure functions (no side effects, no mutations) have no underscore.

Examples:
- Pure: compute_batch_indices(), is_epoch_boundary()
- With side effects: __iter__() (mutates internal state), next_batch_() (mutates state)
"""
import jax
from jax import Array
from typing import NamedTuple, Iterator, Optional, Tuple, Dict
from shared_lib.random_utils import infinite_safe_keys


class BatchMetadata(NamedTuple):
    """Metadata about a batch."""
    epoch: int
    batch_idx: int
    is_epoch_start: bool
    is_epoch_end: bool
    total_batches_in_epoch: int
    global_batch_idx: int


def compute_batch_indices(batch_idx: int, batch_size: int) -> Tuple[int, int]:
    """Pure function to compute batch start and end indices."""
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    return start_idx, end_idx


def is_epoch_start(batch_idx: int) -> bool:
    """Pure function to check if this is the start of an epoch."""
    return batch_idx == 0


def compute_total_batches(n_samples: int, batch_size: int) -> int:
    """Pure function to compute total number of batches per epoch."""
    return n_samples // batch_size


def compute_is_epoch_end(batch_idx: int, total_batches: int) -> bool:
    """Pure function to check if this is the end of an epoch."""
    return batch_idx == total_batches - 1


class BatchSampler:
    """
    ECS-style component for sampling batches with epoch metadata.
    Yields batches with metadata about epochs, batch indices, etc.
    """
    
    def __init__(
        self,
        data: Dict[str, Array],
        batch_size: int,
        n_samples: int,
        num_epochs: int,
        key_gen,
        shuffle: bool = True
    ):
        """
        Initialize the batch sampler.
        
        Args:
            data: Dictionary of JAX arrays (e.g., {'X': Array, 'y': Array})
            batch_size: Size of each batch
            num_epochs: Number of epochs to generate batches for
            key_gen: Generator for random keys
            shuffle: Whether to shuffle data at the start of each epoch
        """
        self.data = data
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.key_gen = key_gen
        self.shuffle = shuffle
        
        self.n_samples = n_samples
        self.total_batches_per_epoch = compute_total_batches(self.n_samples, batch_size)
        self.total_batches = self.total_batches_per_epoch * num_epochs
        
        # Current state
        self.current_epoch = 0
        self.current_batch_in_epoch = 0
        self.global_batch_idx = 0
        self.data_shuffled: Optional[Dict[str, Array]] = None
    
    def __iter__(self) -> Iterator[Tuple[Dict[str, Array], BatchMetadata]]:
        """
        Iterator that yields batches with metadata. Mutates internal state.
        Note: Special Python method, cannot be renamed to follow _ convention.
        Yields: (batch_dict, metadata)
        """
        # Reset state
        self.global_batch_idx = 0
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.current_batch_in_epoch = 0
            
            # Shuffle data at the start of each epoch
            if self.shuffle:
                key = next(self.key_gen).get()
                perm = jax.random.permutation(key, self.n_samples)
                self.data_shuffled = {k: v[perm] for k, v in self.data.items()}
            else:
                self.data_shuffled = self.data
            
            # Generate batches for this epoch
            for batch_idx in range(self.total_batches_per_epoch):
                # Create metadata
                metadata = BatchMetadata(
                    epoch=self.current_epoch,
                    batch_idx=batch_idx,
                    is_epoch_start=is_epoch_start(batch_idx),
                    is_epoch_end=compute_is_epoch_end(batch_idx, self.total_batches_per_epoch),
                    total_batches_in_epoch=self.total_batches_per_epoch,
                    global_batch_idx=self.global_batch_idx
                )
                
                # Get batch
                start_idx, end_idx = compute_batch_indices(batch_idx, self.batch_size)
                batch = {k: v[start_idx:end_idx] for k, v in self.data_shuffled.items()}
                
                # Update state
                self.current_batch_in_epoch = batch_idx + 1
                self.global_batch_idx += 1
                
                yield batch, metadata

