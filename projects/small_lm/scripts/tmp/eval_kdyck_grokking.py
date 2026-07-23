"""
Evaluate whether the model groks the k-Dyck context-free grammar.

Trains a fresh model on k-Dyck sequences and evaluates CFG accuracy at
regular checkpoints, looking for the grokking transition.

Accuracy metrics (per position in each test sequence):
  - forced_acc : at positions where remaining_tokens == stack_depth (MUST close
                 with the exact matching bracket), fraction where argmax is correct.
                 This is the hardest test — it requires tracking nested structure.
  - valid_acc  : fraction of ALL positions where argmax is any valid next token
                 (a valid opening bracket OR the correct closing bracket).
  - bracket_acc: fraction where argmax is ANY bracket token (not a non-k-Dyck ID).

A model that truly groks the CFG should have forced_acc → ~100%.

Usage:
    uv run python projects/small_lm/scripts/tmp/eval_kdyck_grokking.py
    uv run python projects/small_lm/scripts/tmp/eval_kdyck_grokking.py --steps 5000
    uv run python projects/small_lm/scripts/tmp/eval_kdyck_grokking.py --load exp_kdyck_phase1
"""

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared_lib.random_utils import infinite_safe_keys_from_key

SMALL_LM = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SMALL_LM))

import exp_kdyck as M
from lib.kdyck import KDyck


PKL_DIR = SMALL_LM / "pkl"


# ── CFG accuracy evaluation ────────────────────────────────────────────────────

def stack_states(sequence: list[int], kdyck: KDyck) -> list[list[int]]:
    """Return the stack state BEFORE each token in the sequence."""
    states = []
    stack  = []
    for tok in sequence:
        states.append(list(stack))
        if tok in kdyck.open_ids:
            bracket_type = tok - kdyck.token_offset
            stack.append(bracket_type)
        else:
            if stack:
                stack.pop()
    return states


def valid_next_tokens(stack: list[int], remaining: int, kdyck: KDyck) -> list[int]:
    """Return list of valid next token IDs given current stack state."""
    depth = len(stack)
    must_close = (remaining == depth)
    must_open  = (depth == 0)

    valid = []
    if not must_close:
        valid.extend(kdyck.open_ids)   # any opening bracket
    if not must_open:
        valid.append(kdyck.close_ids[stack[-1]])  # only the correct close
    return valid


@jax.jit
def get_logits(params, token_ids):
    return M.forward(params, token_ids)


def evaluate_cfg_accuracy(params, kdyck: KDyck, n_test: int = 500, seed: int = 99999):
    """
    Returns dict with forced_acc, valid_acc, bracket_acc.

    - forced_acc : at positions where exactly 1 token is valid (must-close),
                   fraction where argmax == that token.
    - valid_acc  : fraction of ALL positions where argmax is in valid_next_tokens.
    - bracket_acc: fraction of ALL positions where argmax is any k-Dyck token.
    """
    test_batch = kdyck.generate_batch(n_test, seed=seed)  # (N, T)

    forced_correct = 0
    forced_total   = 0
    valid_correct  = 0
    bracket_correct = 0
    total          = 0

    # Process in mini-batches for efficiency
    bs = 32
    for i in range(0, n_test, bs):
        chunk = jnp.array(test_batch[i:i+bs])       # (bs, T)
        # Get logits for all positions: forward takes (bs, T), returns (bs, T, V)
        logits = get_logits(params, chunk)            # (bs, T, V)
        preds  = jnp.argmax(logits, axis=-1)          # (bs, T)
        preds_np = np.array(preds)
        chunk_np = test_batch[i:i+bs]

        for seq_idx in range(chunk_np.shape[0]):
            seq    = chunk_np[seq_idx].tolist()
            states = stack_states(seq, kdyck)
            T      = len(seq)

            for t in range(T - 1):  # predict token at position t+1 given seq[:t+1]
                stack     = states[t + 1]   # stack BEFORE token t+1
                remaining = T - (t + 1)     # tokens remaining including t+1
                valid     = valid_next_tokens(stack, remaining, kdyck)
                pred      = int(preds_np[seq_idx, t])
                target    = seq[t + 1]

                is_bracket = (kdyck.token_offset <= pred < kdyck.token_offset + 2 * kdyck.k)
                is_valid   = pred in valid
                is_forced  = (len(valid) == 1)   # must-close position

                total   += 1
                bracket_correct += int(is_bracket)
                valid_correct   += int(is_valid)

                if is_forced:
                    forced_total   += 1
                    forced_correct += int(pred == target)

    return {
        "forced_acc" : forced_correct / max(forced_total, 1),
        "valid_acc"  : valid_correct  / max(total, 1),
        "bracket_acc": bracket_correct / max(total, 1),
        "forced_total": forced_total,
        "total"       : total,
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train_and_eval(train_steps: int, eval_every: int, kdyck: KDyck, params=None):
    print(f"\nTraining on k-Dyck for {train_steps} steps (k={kdyck.k}, seq_len={kdyck.seq_len})")
    print(f"Evaluating every {eval_every} steps\n")

    if params is None:
        key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(42))
        params  = M.init_params(key_gen)
        print(f"  Fresh init — {M.n_params(params):,} params")
    else:
        print(f"  Loaded checkpoint — {M.n_params(params):,} params")

    optimizer  = optax.chain(
        optax.clip_by_global_norm(M.GRAD_CLIP),
        optax.adamw(M.PRETRAIN_LR, weight_decay=0.0),
    )
    opt_state  = optimizer.init(params)

    @jax.jit
    def train_step(carry, batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(M.loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss

    # Eval at step 0 (random init)
    m0 = evaluate_cfg_accuracy(params, kdyck)
    print(f"{'step':>6}  {'loss':>8}  {'forced_acc':>12}  {'valid_acc':>10}  {'bracket_acc':>12}  {'forced_n':>9}")
    print("-" * 75)
    print(f"{'0':>6}  {'—':>8}  {m0['forced_acc']:12.1%}  {m0['valid_acc']:10.1%}  {m0['bracket_acc']:12.1%}  {m0['forced_total']:9,}")

    step, epoch = 0, 0
    results = [{"step": 0, **m0}]

    while step < train_steps:
        for batch_np in kdyck.generate_epoch(10_000, M.BATCH_SIZE, epoch):
            if step >= train_steps:
                break
            (params, opt_state), loss = train_step(
                (params, opt_state), jnp.array(batch_np)
            )
            step += 1

            if step % eval_every == 0 or step == train_steps:
                m = evaluate_cfg_accuracy(params, kdyck)
                print(f"{step:6d}  {float(loss):8.4f}  {m['forced_acc']:12.1%}  "
                      f"{m['valid_acc']:10.1%}  {m['bracket_acc']:12.1%}  {m['forced_total']:9,}")
                results.append({"step": step, "loss": float(loss), **m})
        epoch += 1

    return params, results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    type=int, default=3000, help="k-Dyck training steps")
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--k",        type=int, default=M.KDYCK_K)
    parser.add_argument("--load",     type=str, default=None,
                        help="Load params from pkl/{name}.pkl (e.g. exp_kdyck_phase1)")
    args = parser.parse_args()

    kdyck = KDyck(k=args.k, seq_len=M.MAX_SEQ_LEN)

    params = None
    if args.load:
        path = PKL_DIR / f"params_{args.load}.pkl"
        print(f"Loading {path} ...")
        with open(path, "rb") as f:
            raw = pickle.load(f)
        params = {k: jnp.array(v) for k, v in raw.items()}

    params, results = train_and_eval(args.steps, args.eval_every, kdyck, params)

    # Save the trained checkpoint
    PKL_DIR.mkdir(exist_ok=True)
    out = PKL_DIR / "params_kdyck_grok_test.pkl"
    with open(out, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    print(f"\nCheckpoint saved → {out}")

    # Summary
    print("\n=== Summary ===")
    best = max(results, key=lambda r: r.get("forced_acc", 0))
    print(f"Best forced_acc: {best['forced_acc']:.1%} at step {best['step']}")
    final = results[-1]
    print(f"Final — forced_acc={final.get('forced_acc',0):.1%}  "
          f"valid_acc={final.get('valid_acc',0):.1%}  "
          f"bracket_acc={final.get('bracket_acc',0):.1%}")


if __name__ == "__main__":
    main()
