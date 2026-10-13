"""DUST vs EGGROLL. Importing this package turns on JAX's persistent compilation cache:
a DUST update compiles ~40 distinct programs (one per block and site group, per draw-chunk
width), about 4.5 minutes cold, and every later run with the same shapes should skip that."""

import os
from pathlib import Path

import jax

_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache" / "jax_dust"))
jax.config.update("jax_compilation_cache_dir", _cache)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
