"""
Universal AR — exp2: in-context completion as matrix completion.

Frame
=====
Dataset = a (samples × positions) matrix; entries = values; one extra "label"
column. An episode = S rows, each with ~50% of its entries OBSERVED (partial).
Row factor is AMORTISED from the observed entries (not a per-sample lookup), so
it generalises to unseen samples; column roles r_pos are shared. A capacity
bottleneck (dim D + HRR superposition) forces MF-style sharing.

    P_ref = Σ_{observed pos}  r_pos[pos] ⊛ val_emb[bin(value)]     (row pattern)

Three tests ("unknown" = a held-out (pos,ref) COMBINATION, not a novel value):
    1. retrieval          — recall an OBSERVED entry (self-read from P_ref)
    2. label  generalise  — predict a withheld label column entry (leave-self-out)
    3. content generalise — predict a held-out pixel entry       (leave-self-out)

v1 objective = "generalise first": the loss reconstructs OBSERVED entries only
(masked pixel recon + leave-one-out label). Nothing held-out enters the loss —
we measure whether tests 2 & 3 emerge for free. If not, the fallback is to add
held-out entries to the loss (supervised completion).

Reads (concepts.md §5a), both reported at eval:
    linear   : p̂ = Σ_{s≠q} (cue·keyₛ)          · valueₛ
    attention: p̂ = Σ_{s≠q} softmax(β cue·keyₛ) · valueₛ

Usage:
    uv run python projects/universal-ar/scripts/run_experiments.py --bg exp2
"""

import json
import time
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import Array

from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp2"
JSONL = Path(__file__).parent / "results.jsonl"

# ── hyperparameters ───────────────────────────────────────────────────────────
DATASET        = "mnist"
D              = 256
N_VAL_BINS     = 32
N_CLASSES      = 10
POS_DIM        = 784

EP_SAMPLES     = 32       # S: rows per episode
N_OBS          = 384      # observed entries per row (~50%)
N_TGT          = 64       # recon targets (subset of observed, excluded from cue)
N_EVAL         = 64       # entries scored per row for recall/content tests
BATCH_EPISODES = 8

READ_MODE      = "attn"   # read used for the training loss
NUM_STEPS      = 4000
EVAL_EVERY     = 200
EVAL_EPISODES  = 64
LEARNING_RATE  = 1e-3
RANDOM_SEED    = 0


# ── HRR primitives ────────────────────────────────────────────────────────────
def fft(x: Array) -> Array:
    return jnp.fft.fft(x, axis=-1)


def bind(role: Array, filler: Array) -> Array:
    return jnp.fft.ifft(fft(role) * fft(filler), axis=-1).real


def unbind(bound: Array, role: Array) -> Array:
    return jnp.fft.ifft(fft(bound) * jnp.conj(fft(role)), axis=-1).real


# ── params ────────────────────────────────────────────────────────────────────
def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    config = {
        "dataset": DATASET, "D": D, "n_val_bins": N_VAL_BINS,
        "ep_samples": EP_SAMPLES, "n_obs": N_OBS, "n_tgt": N_TGT,
        "batch_episodes": BATCH_EPISODES, "read_mode": READ_MODE,
        "num_steps": NUM_STEPS, "learning_rate": LEARNING_RATE, "random_seed": RANDOM_SEED,
    }
    k = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 5)
    s = 1.0 / D ** 0.5
    params = {
        "r_pos":     jax.random.normal(k[0], (POS_DIM, D)) * s,
        "val_emb":   jax.random.normal(k[1], (N_VAL_BINS, D)) * s,
        "r_label":   jax.random.normal(k[2], (D,)) * s,
        "label_emb": jax.random.normal(k[3], (N_CLASSES, D)) * s,
        "log_beta":  jnp.array(np.log(20.0), dtype=jnp.float32),
    }
    return config, params


def n_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


# ── episode sampling (host side) ──────────────────────────────────────────────
def sample_episodes(Xflat, y, B, S, rng):
    N = Xflat.shape[0]
    idx = rng.integers(0, N, size=(B, S))
    imgs = Xflat[idx]                                    # (B,S,784)
    labels = y[idx].astype(np.int32)
    perm = rng.random((B, S, POS_DIM)).argsort(-1)       # random pixel order
    obs = perm[..., :N_OBS]                              # observed (in context)
    ho = perm[..., N_OBS:]                               # held-out (never in context)
    tgt = obs[..., :N_TGT]                               # recon targets (⊂ observed)
    cue = obs[..., N_TGT:]                               # row cue (builds P)
    rec = cue[..., :N_EVAL]                              # observed cells for test-1
    hoe = ho[..., :N_EVAL]                               # held-out cells for test-3

    def binof(pos):
        v = np.take_along_axis(imgs, pos, axis=-1)
        return np.floor(v / 255.0 * (N_VAL_BINS - 1)).astype(np.int32)

    return {k: jnp.asarray(v) for k, v in dict(
        cue_pos=cue, cue_bin=binof(cue),
        tgt_pos=tgt, tgt_bin=binof(tgt),
        rec_pos=rec, rec_bin=binof(rec),
        ho_pos=hoe, ho_bin=binof(hoe),
        labels=labels).items()}


# ── model ─────────────────────────────────────────────────────────────────────
def build_row(params, cue_pos, cue_bin):
    """Amortised row pattern P_ref from observed cue entries.  → (B,S,D)."""
    Fp = fft(params["r_pos"])[cue_pos]       # (B,S,Ncue,D) complex
    Fv = fft(params["val_emb"])[cue_bin]
    return jnp.fft.ifft(jnp.sum(Fp * Fv, axis=2), axis=-1).real


def cross_read(params, P, mode: str):
    """Leave-self-out read over rows: query row P_q against other rows.  → p̂ (B,S,D)."""
    S = P.shape[1]
    Pn = P / (jnp.linalg.norm(P, axis=-1, keepdims=True) + 1e-6)
    sim = jnp.einsum("bqd,bkd->bqk", Pn, Pn)
    eye = jnp.eye(S)[None]
    if mode == "attn":
        A = jax.nn.softmax(jnp.exp(params["log_beta"]) * sim - 1e9 * eye, axis=-1)
    else:
        A = sim * (1.0 - eye)
    return jnp.einsum("bqk,bkd->bqd", A, P)


def entries_at(params, pat, pos):
    """Unbind position roles from a pattern → value logits.  pat (B,S,D), pos (B,S,n)."""
    f = unbind(pat[:, :, None, :], params["r_pos"][pos])     # (B,S,n,D)
    return jnp.einsum("bsnd,kd->bsnk", f, params["val_emb"])  # (B,S,n,K)


def label_at(params, pat):
    f = unbind(pat, params["r_label"])
    return jnp.einsum("bsd,cd->bsc", f, params["label_emb"])  # (B,S,N_CLASSES)


def loss_fn(params, cue_pos, cue_bin, tgt_pos, tgt_bin, labels, mode: str):
    P = build_row(params, cue_pos, cue_bin)
    phat = cross_read(params, P, mode)
    pix_logits = entries_at(params, phat, tgt_pos)          # observed recon targets
    pix_loss = optax.softmax_cross_entropy_with_integer_labels(pix_logits, tgt_bin).mean()
    lab_logits = label_at(params, phat)                     # LOO label
    lab_loss = optax.softmax_cross_entropy_with_integer_labels(lab_logits, labels).mean()
    loss = pix_loss + lab_loss
    aux = (pix_loss, lab_loss,
           jnp.mean(jnp.argmax(pix_logits, -1) == tgt_bin),
           jnp.mean(jnp.argmax(lab_logits, -1) == labels))
    return loss, aux


# optimizer (arg 0) and mode (arg 8) are static
@partial(jax.jit, static_argnums=(0, 8))
def train_step(optimizer, params, opt_state, cue_pos, cue_bin, tgt_pos, tgt_bin, labels, mode):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, cue_pos, cue_bin, tgt_pos, tgt_bin, labels, mode)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


@jax.jit
def eval_all(params, b):
    """Return (recall, lab_attn, ho_attn, lab_lin, ho_lin) accuracies."""
    P = build_row(params, b["cue_pos"], b["cue_bin"])
    # test 1: self-recall of observed cells
    rec_logits = entries_at(params, P, b["rec_pos"])
    recall = jnp.mean(jnp.argmax(rec_logits, -1) == b["rec_bin"])

    def gen(mode):
        phat = cross_read(params, P, mode)
        lab = jnp.mean(jnp.argmax(label_at(params, phat), -1) == b["labels"])
        ho = jnp.mean(jnp.argmax(entries_at(params, phat, b["ho_pos"]), -1) == b["ho_bin"])
        return lab, ho

    la, ha = gen("attn")
    ll, hl = gen("linear")
    return recall, la, ha, ll, hl


def evaluate(params, Xflat, y, rng):
    acc = np.zeros(5)
    n = EVAL_EPISODES // BATCH_EPISODES
    for _ in range(n):
        b = sample_episodes(Xflat, y, BATCH_EPISODES, EP_SAMPLES, rng)
        acc += np.array([float(x) for x in eval_all(params, b)])
    r, la, ha, ll, hl = acc / n
    return dict(recall=r, label_attn=la, ho_attn=ha, label_lin=ll, ho_lin=hl)


# ── train ─────────────────────────────────────────────────────────────────────
def train(params, Xtr, ytr, Xte, yte):
    sched = optax.cosine_decay_schedule(LEARNING_RATE, NUM_STEPS)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(RANDOM_SEED)
    ev_rng = np.random.default_rng(RANDOM_SEED + 1)
    hist = {"step": [], "loss": [], "recall": [], "label_attn": [], "ho_attn": [],
            "label_lin": [], "ho_lin": []}
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        b = sample_episodes(Xtr, ytr, BATCH_EPISODES, EP_SAMPLES, rng)
        params, opt_state, loss, aux = train_step(
            optimizer, params, opt_state,
            b["cue_pos"], b["cue_bin"], b["tgt_pos"], b["tgt_bin"], b["labels"], READ_MODE)

        if step % EVAL_EVERY == 0 or step == 1:
            m = evaluate(params, Xte, yte, ev_rng)
            for key in ("recall", "label_attn", "ho_attn", "label_lin", "ho_lin"):
                hist[key].append(m[key])
            hist["step"].append(step); hist["loss"].append(float(loss))
            logging.info(
                f"step {step:5d}  loss {float(loss):.3f}  "
                f"recall {m['recall']:.3f}  "
                f"label[attn/lin] {m['label_attn']:.3f}/{m['label_lin']:.3f}  "
                f"content[attn/lin] {m['ho_attn']:.3f}/{m['ho_lin']:.3f}  "
                f"({time.perf_counter()-t0:.0f}s)")

    return params, hist, time.perf_counter() - t0


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping"); raise SystemExit(0)

    config, params = init()
    logging.info(f"Loading {DATASET}…")
    data = load_supervised_image(DATASET)
    Xtr = np.asarray(data.X.reshape(data.n_samples, -1), dtype=np.float32)
    Xte = np.asarray(data.X_test.reshape(data.n_test_samples, -1), dtype=np.float32)
    ytr = np.asarray(data.y); yte = np.asarray(data.y_test)
    logging.info(f"train {Xtr.shape}  test {Xte.shape}  n_params {n_params(params)}")
    logging.info(f"episode S={EP_SAMPLES}  observed={N_OBS}  cue={N_OBS-N_TGT}  "
                 f"recon_tgt={N_TGT}  chance: label={1/N_CLASSES:.2f} content={1/N_VAL_BINS:.3f}")

    params, hist, elapsed = train(params, Xtr, ytr, Xte, yte)

    final = {k: hist[k][-1] for k in ("recall", "label_attn", "ho_attn", "label_lin", "ho_lin")}
    logging.info(f"DONE {elapsed:.0f}s  {final}")
    row = {
        "experiment": EXP_NAME,
        "name": "matrix-completion in-context: recall / label-gen / content-gen (HRR auto-assoc)",
        "time_s": elapsed, "n_params": n_params(params),
        **final, **config, "history": hist,
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    logging.info(f"appended → {JSONL}")
