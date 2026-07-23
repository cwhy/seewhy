"""
Generate text samples from a trained exp_baseline checkpoint.

Usage:
    uv run python projects/small_lm/scripts/tmp/generate_samples.py
    uv run python projects/small_lm/scripts/tmp/generate_samples.py --exp exp_baseline --temp 0.7 --topk 50
"""

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.data_processing import load_tokenizer
import exp_baseline as M

PKL_DIR  = Path(__file__).parent.parent.parent / "pkl"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

PROMPTS = [
    "hi guppy",
    "how are you feeling",
    "are you hungry",
    "it's hot today",
    "i changed your water",
    "do you get lonely",
    "what's the meaning of life",
    "tell me a joke",
    "goodnight guppy",
    "what do you think about cryptocurrency",
]


def load_params(exp_name: str) -> dict:
    path = PKL_DIR / f"params_{exp_name}.pkl"
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}


def generate(
    params: dict,
    tok,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: int = 0,
) -> str:
    """Autoregressive generation with top-k sampling."""
    # Format as chat-ml
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids  = tok.encode(text).ids[:M.MAX_SEQ_LEN - 1]  # leave room for 1 new token
    key  = jax.random.PRNGKey(seed)

    generated = []
    for _ in range(max_new_tokens):
        if len(ids) >= M.MAX_SEQ_LEN:
            break

        input_ids = jnp.array([ids], dtype=jnp.int32)            # (1, T)
        logits    = M.forward(params, input_ids)[0, -1, :]        # (VOCAB_SIZE,)

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k masking
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

    # Decode only the generated part
    response = tok.decode(generated)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",  default="exp_baseline")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--topk", type=int,   default=50)
    args = parser.parse_args()

    print(f"Loading params from pkl/{args.exp} ...")
    params = load_params(args.exp)
    tok    = load_tokenizer(DATA_DIR / "tokenizer.json")
    print(f"Ready. Generating with temp={args.temp}, top_k={args.topk}\n")
    print("=" * 60)

    for i, prompt in enumerate(PROMPTS):
        response = generate(params, tok, prompt, temperature=args.temp, top_k=args.topk, seed=i)
        print(f"User:  {prompt}")
        print(f"Guppy: {response}")
        print()


if __name__ == "__main__":
    main()
