"""
generate_kdyck_phase1.py — Run a fish question through a k-Dyck-only checkpoint.

The model has only ever seen token IDs 3–34 (32 bracket tokens, k=16).
It has never seen Guppy vocabulary or natural language.

Usage:
    uv run python projects/small_lm/scripts/tmp/generate_kdyck_phase1.py
    uv run python projects/small_lm/scripts/tmp/generate_kdyck_phase1.py --phase1
"""

import sys, pickle, argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "small_lm"))

from lib.data_processing import load_tokenizer
import exp_kdyck as model

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--phase1", action="store_true",
                    help="Use params_exp_kdyck_rope_phase1.pkl instead of grok_test")
parser.add_argument("--max-new-tokens", type=int, default=40,
                    help="Tokens to generate after the prompt")
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

PKL_DIR = ROOT / "small_lm" / "pkl"
TOK_PATH = ROOT / "small_lm" / "data" / "tokenizer.json"

if args.phase1:
    ckpt = PKL_DIR / "params_exp_kdyck_rope_phase1.pkl"
else:
    ckpt = PKL_DIR / "params_kdyck_grok_test.pkl"

print(f"Checkpoint : {ckpt.name}")
print(f"Tokenizer  : {TOK_PATH}")
print()

# ── Load ───────────────────────────────────────────────────────────────────────

with open(ckpt, "rb") as f:
    params_np = pickle.load(f)
params = {k: jnp.array(v) for k, v in params_np.items()}

tok = load_tokenizer(TOK_PATH)
vocab_size = tok.get_vocab_size()
print(f"Vocab size : {vocab_size}")
print(f"Parameters : {sum(v.size for v in params_np.values()):,}")
print()

# ── Prompt ─────────────────────────────────────────────────────────────────────

FISH_QUESTION = (
    "<|im_start|>user\n"
    "What do fish eat?\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

enc = tok.encode(FISH_QUESTION)
prompt_ids = enc.ids
print(f"Prompt     : {repr(FISH_QUESTION)}")
print(f"Prompt IDs : {prompt_ids}")
print(f"Prompt len : {len(prompt_ids)} tokens")
print()

# ── Sampling ───────────────────────────────────────────────────────────────────

@jax.jit
def get_logits(params, token_ids):
    """Return logits for the last position."""
    logits = model.forward(params, token_ids)  # (1, T, V)
    return logits[0, -1, :]  # (V,)


def top_k_sample(logits_np, top_k, temperature, rng):
    logits = logits_np / temperature
    top_indices = np.argsort(logits)[-top_k:]
    top_logits = logits[top_indices]
    top_logits -= top_logits.max()
    probs = np.exp(top_logits)
    probs /= probs.sum()
    chosen = rng.choice(top_indices, p=probs)
    return int(chosen)


rng = np.random.default_rng(42)
context = list(prompt_ids)
generated_ids = []

print("=" * 60)
print("GENERATION (step-by-step)")
print("=" * 60)

for step in range(args.max_new_tokens):
    # Truncate to MAX_SEQ_LEN - 1 to leave room for the next token
    window = context[-(model.MAX_SEQ_LEN - 1):]
    token_arr = jnp.array([window], dtype=jnp.int32)

    logits_jax = get_logits(params, token_arr)
    logits = np.array(logits_jax, dtype=np.float32)

    # Top-5 predictions
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5_dec = [tok.decode([int(i)]) for i in top5_idx]
    top5_logits = logits[top5_idx]

    next_id = top_k_sample(logits, args.top_k, args.temperature, rng)
    next_tok_str = tok.decode([next_id])

    print(f"\nStep {step+1:2d} | chosen id={next_id:4d}  repr={repr(next_tok_str)}")
    print("         Top-5 logits:")
    for i, (idx, val, dec) in enumerate(zip(top5_idx, top5_logits, top5_dec)):
        marker = " <-- chosen" if idx == next_id else ""
        in_bracket_range = 3 <= idx <= 34
        tag = " [BRACKET]" if in_bracket_range else ""
        print(f"           #{i+1}  id={idx:4d}  logit={val:7.3f}  repr={repr(dec)}{tag}{marker}")

    generated_ids.append(next_id)
    context.append(next_id)

    if next_id == model.EOS_ID:
        print("\n[EOS reached — stopping]")
        break

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Generated IDs : {generated_ids}")
decoded_full = tok.decode(generated_ids)
print(f"Decoded text  : {repr(decoded_full)}")

n_bracket = sum(1 for i in generated_ids if 3 <= i <= 34)
n_lang    = sum(1 for i in generated_ids if i > 34)
n_special = sum(1 for i in generated_ids if i < 3)
print()
print(f"Token breakdown ({len(generated_ids)} total):")
print(f"  bracket (IDs 3-34) : {n_bracket}  ({100*n_bracket/max(1,len(generated_ids)):.0f}%)")
print(f"  language (IDs >34) : {n_lang}  ({100*n_lang/max(1,len(generated_ids)):.0f}%)")
print(f"  special (IDs 0-2)  : {n_special}  ({100*n_special/max(1,len(generated_ids)):.0f}%)")
print()
print("Prompt token IDs breakdown:")
n_p_bracket = sum(1 for i in prompt_ids if 3 <= i <= 34)
n_p_lang    = sum(1 for i in prompt_ids if i > 34)
n_p_special = sum(1 for i in prompt_ids if i < 3)
print(f"  bracket (IDs 3-34) : {n_p_bracket}")
print(f"  language (IDs >34) : {n_p_lang}")
print(f"  special (IDs 0-2)  : {n_p_special}")
