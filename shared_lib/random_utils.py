from __future__ import annotations

from typing import Tuple, Literal, Iterator

import jax
from jax import Array



class SafeKey:
    """Safety wrapper for PRNG keys."""

    def __init__(self, key: Array):
        self._key = key
        self._used = False

    def _assert_not_used(self) -> None:
        if self._used:
            raise RuntimeError('Random key has been used previously.')

    def get(self) -> Array:
        self._assert_not_used()
        self._used = True
        return self._key

    def split(self, num_keys=2) -> Tuple['SafeKey', ...]:
        self._assert_not_used()
        self._used = True
        new_keys = jax.random.split(self._key, num_keys)
        return jax.tree_map(SafeKey, tuple(new_keys))

    def duplicate(self, num_keys=2) -> Tuple['SafeKey', ...]:
        self._assert_not_used()
        self._used = True
        return tuple(SafeKey(self._key) for _ in range(num_keys))


def infinite_safe_keys(seed: int) -> Iterator[SafeKey]:
    init_key = jax.random.key(seed)
    while True:
        init_key, key = jax.random.split(init_key)
        yield SafeKey(key)

def infinite_safe_keys_from_key(init_key: Array) -> Iterator[SafeKey]:
    while True:
        init_key, key = jax.random.split(init_key)
        yield SafeKey(key)