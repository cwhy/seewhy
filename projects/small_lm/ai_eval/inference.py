"""
Model-agnostic inference for small_lm experiments.

Each experiment has its own forward() function (e.g. exp_rope has no pos_embed).
This module dynamically loads the correct experiment module and runs generation.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

SMALL_LM_DIR = Path(__file__).parent.parent


def _load_exp_module(exp_name: str):
    """Import projects/small_lm/{exp_name}.py as a module."""
    path = SMALL_LM_DIR / f"{exp_name}.py"
    spec = importlib.util.spec_from_file_location(exp_name, path)
    mod  = importlib.util.module_from_spec(spec)
    # Make sure small_lm is on sys.path so the module can find its own imports
    if str(SMALL_LM_DIR) not in sys.path:
        sys.path.insert(0, str(SMALL_LM_DIR))
    spec.loader.exec_module(mod)
    return mod


# Cache loaded modules so we don't re-exec the file for every question
_MODULE_CACHE: dict = {}


def generate(
    exp_name: str,
    params: dict,
    tok,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: int = 0,
) -> str:
    """Autoregressive top-k generation using the correct experiment's forward()."""
    if exp_name not in _MODULE_CACHE:
        _MODULE_CACHE[exp_name] = _load_exp_module(exp_name)
    M = _MODULE_CACHE[exp_name]

    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids  = tok.encode(text).ids[:M.MAX_SEQ_LEN - 1]
    key  = jax.random.PRNGKey(seed)

    generated = []
    for _ in range(max_new_tokens):
        if len(ids) >= M.MAX_SEQ_LEN:
            break

        input_ids = jnp.array([ids], dtype=jnp.int32)
        logits    = M.forward(params, input_ids)[0, -1, :]

        logits = logits / max(temperature, 1e-8)

        if top_k > 0 and top_k < M.VOCAB_SIZE:
            top_vals, _ = jax.lax.top_k(logits, top_k)
            threshold   = top_vals[-1]
            logits      = jnp.where(logits >= threshold, logits, jnp.finfo(logits.dtype).min)

        probs = jax.nn.softmax(logits)
        key, subkey = jax.random.split(key)
        next_id = int(jax.random.choice(subkey, M.VOCAB_SIZE, p=probs))

        if next_id == M.EOS_ID:
            break

        ids.append(next_id)
        generated.append(next_id)

    return tok.decode(generated).strip()
