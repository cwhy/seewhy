"""
k-Dyck sequence generator for procedural pre-training.

A k-Dyck sequence is a balanced parenthesis string over k bracket types.
Token encoding (offset by TOKEN_OFFSET to avoid clashing with PAD/BOS/EOS):
  - Opening bracket of type i : TOKEN_OFFSET + i          (i in 0..k-1)
  - Closing bracket of type i : TOKEN_OFFSET + k + i      (i in 0..k-1)

The generator produces batches of shape (batch_size, seq_len) as numpy int32
arrays.  Each sequence is independently generated with a deterministic seed
so that the same (epoch, batch_index) always yields the same data.

Reference: Shinnick et al. "Procedural Pretraining" (arXiv:2601.21725)

Public API:
    KDyck(k, seq_len, token_offset)
    KDyck.generate_sequence(rng) -> list[int]
    KDyck.generate_batch(batch_size, seed) -> np.ndarray (batch_size, seq_len)
    KDyck.generate_epoch(n_samples, batch_size, epoch) -> iterator of np.ndarray
"""

import numpy as np


# Default offset: skip PAD=0, BOS=1, EOS=2
DEFAULT_TOKEN_OFFSET = 3


class KDyck:
    def __init__(self, k: int = 16, seq_len: int = 128, token_offset: int = DEFAULT_TOKEN_OFFSET, p_open: float = 0.49):
        """
        Args:
            k:            Number of bracket types.
            seq_len:      Fixed output sequence length (sequences are padded or truncated).
            token_offset: Token IDs start at this value (to avoid special tokens).
            p_open:       Probability of opening a new bracket when both open/close are valid.
        """
        assert k >= 1
        assert token_offset >= 0
        self.k            = k
        self.seq_len      = seq_len
        self.token_offset = token_offset
        self.p_open       = p_open

        self.open_ids  = list(range(token_offset, token_offset + k))
        self.close_ids = list(range(token_offset + k, token_offset + 2 * k))
        self.vocab_ids = self.open_ids + self.close_ids   # 2k token IDs used

    def generate_sequence(self, rng: np.random.Generator) -> list[int]:
        """Generate one valid k-Dyck sequence of exactly self.seq_len tokens.

        Uses the iterative stack-based algorithm from the paper:
        - When stack is empty: must open.
        - When remaining slots == stack depth: must close all.
        - Otherwise: open with probability p_open, close otherwise.
        """
        tokens  = []
        stack   = []
        target  = self.seq_len

        while len(tokens) < target:
            remaining = target - len(tokens)
            depth     = len(stack)

            must_close = (remaining == depth)   # no room to close later
            must_open  = (depth == 0)            # nothing to close

            if must_close:
                action = "close"
            elif must_open:
                action = "open"
            else:
                action = "open" if rng.random() < self.p_open else "close"

            if action == "open":
                bracket_type = int(rng.integers(0, self.k))
                tokens.append(self.open_ids[bracket_type])
                stack.append(bracket_type)
            else:
                bracket_type = stack.pop()
                tokens.append(self.close_ids[bracket_type])

        return tokens

    def generate_batch(self, batch_size: int, seed: int) -> np.ndarray:
        """Generate a batch of k-Dyck sequences.

        Returns:
            np.ndarray of shape (batch_size, seq_len), dtype int32.
        """
        batch = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        for i in range(batch_size):
            rng = np.random.default_rng(seed * 10_000 + i)
            batch[i] = self.generate_sequence(rng)
        return batch

    def generate_epoch(self, n_samples: int, batch_size: int, epoch: int):
        """Yield batches covering n_samples sequences for a given epoch.

        Each (epoch, batch_index) pair yields the same data deterministically.
        """
        n_batches = n_samples // batch_size
        for batch_idx in range(n_batches):
            seed = epoch * 100_000 + batch_idx
            yield self.generate_batch(batch_size, seed)

    @property
    def n_tokens(self) -> int:
        """Number of distinct token IDs used (2k)."""
        return 2 * self.k
