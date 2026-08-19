"""exp5 — subspace selection vs sparsification (H6).

Hypothesis (paper §6.1): embedding-only training works by steering the frozen
computation into a *low-dimensional subspace* of the hidden space in which the
target function already happens to be implemented. The competing explanation is
sparsification — that training finds a small sub-network of individual neurons.
The two make different predictions about the same activations:

    subspace selection  ->  the top 10 *principal components* explain a large
                            fraction of the variance
    sparsification      ->  the top 10 *individual neurons* do

So both are measured on the same hidden states, and it is the gap between them
that is the claim. Neither number means anything on its own: 10 of 1024
directions explaining 60% is only interesting next to 10 of 1024 neurons
explaining 3%.

This experiment trains nothing. It loads the parameters exp1 and exp4 saved and
measures their activations, so it cannot disagree with the models the other
tables describe.

Usage:
    uv run python projects/random-transformers/scripts/run_experiments.py exp5
"""

import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

from lib.model import forward
from lib.results_io import append_result, read_rows
from lib.tasks import get_task, make_memorization

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp5"
TOP_K = 10
N_INPUTS = 2048
N_LAYER, N_HEAD = 2, 4
SEEDS = (0, 1, 2)
MODES = ("random", "full")
TASKS = ("mod_add", "needle", "decimal", "parens")

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"


def explained_variance(acts: np.ndarray, k: int = TOP_K) -> tuple[float, float]:
    """``(principal_component_fraction, neuron_fraction)`` for the top ``k`` of each.

    ``acts`` is (n_points, d). Both bases are measured against the *same*
    total variance — the trace of the covariance — so the two numbers are
    directly comparable, which is the entire point of the comparison.
    """
    x = acts - acts.mean(0, keepdims=True)
    n = x.shape[0]
    total = float((x ** 2).sum() / n)
    if total == 0:
        return float("nan"), float("nan")

    # principal components: eigenvalues of the covariance = squared singular
    # values / n. Computed from the smaller Gram matrix when d > n.
    sv = np.linalg.svd(x, compute_uv=False)
    pc = float((sv[:k] ** 2).sum() / n / total)

    neuron_var = (x ** 2).sum(0) / n
    neu = float(np.sort(neuron_var)[::-1][:k].sum() / total)
    return pc, neu


def collect_activations(params, toks, mask):
    """Hidden states at the embedding output and after each block.

    Only positions the model has actually read are used: everything up to and
    including the position whose prediction is scored. Padding positions carry
    the same embedding vector in every sequence and would deflate the variance
    denominator, making both bases look better than they are.
    """
    _, hidden, _ = forward(params, toks, n_layer=N_LAYER, n_head=N_HEAD, return_hidden=True)
    scored = jnp.argmax(mask, axis=1)                       # first scored position
    valid = jnp.arange(toks.shape[1])[None, :] <= scored[:, None]
    idx = np.array(valid).reshape(-1)
    return {name: np.array(h, dtype=np.float64).reshape(-1, h.shape[-1])[idx]
            for name, h in hidden.items()}


def main():
    done = {r["cell"] for r in read_rows(JSONL) if r.get("experiment") == EXP}

    jobs = [(t, m, s, 1024, PROJECT / f"params_exp1_{t}_{m}_1024_{s}.pkl")
            for t in TASKS for m in MODES for s in SEEDS]
    jobs += [("memorization", m, s, 128, PROJECT / f"params_exp4_{m}_{s}.pkl")
             for m in MODES for s in SEEDS]

    task_cache: dict[str, object] = {}

    for task_name, mode, seed, width, path in jobs:
        cell = f"{task_name}/{mode}/{width}/{seed}"
        if cell in done:
            continue
        if not path.exists():
            logging.warning(f"  missing {path.name} — skipping {cell}")
            continue

        if task_name not in task_cache:
            task_cache.clear()
            task_cache[task_name] = (make_memorization() if task_name == "memorization"
                                     else get_task(task_name))
        task = task_cache[task_name]

        with open(path, "rb") as f:
            params = {k: jnp.asarray(v) for k, v in pickle.load(f).items()}

        toks, mask = task.test_set()
        toks, mask = toks[:N_INPUTS], mask[:N_INPUTS]
        t0 = time.time()
        acts = collect_activations(params, toks, mask)

        row = {"experiment": EXP, "cell": cell,
               "name": f"explained variance — {cell}",
               "task": task_name, "mode": mode, "seed": seed, "d": width,
               "n_layer": N_LAYER, "n_head": N_HEAD,
               "top_k": TOP_K, "n_inputs": int(N_INPUTS),
               "n_points": int(next(iter(acts.values())).shape[0])}

        for layer, a in acts.items():
            pc, neu = explained_variance(a)
            row[f"pc_{layer}"] = pc
            row[f"neuron_{layer}"] = neu
        row["time_s"] = time.time() - t0

        append_result(JSONL, row)
        logging.info(f"  {cell}  " + "  ".join(
            f"{k}={row[k]:.3f}" for k in row if k.startswith(("pc_", "neuron_"))))

    logging.info("exp5 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
