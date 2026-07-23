"""
exp4 — ICA of MNIST: global (all 60 000 images) + per-digit (one fit per class).

Saves results to ica_results_exp4.pkl for post-hoc visualization.

Usage:
    uv run python projects/component-analysis/scripts/run_experiments.py exp4
"""

import json, time, sys, pickle, logging, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_1d

sys.path.insert(0, str(Path(__file__).parent))
from lib.ica import fit_ica, project, reconstruction_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME         = "exp4"
N_COMPONENTS_GLOBAL = 50
N_COMPONENTS_DIGIT  = 20
JSONL = Path(__file__).parent / "results.jsonl"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()

    logging.info("Loading MNIST...")
    data = load_supervised_1d("mnist")
    X_tr = np.array(data.X,      dtype=np.float64)
    y_tr = np.array(data.y,      dtype=np.int32)
    X_te = np.array(data.X_test, dtype=np.float64)
    y_te = np.array(data.y_test, dtype=np.int32)

    # ── Global ICA ────────────────────────────────────────────────────────────
    logging.info(f"Fitting global ICA (n={N_COMPONENTS_GLOBAL})...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ica_global = fit_ica(X_tr, N_COMPONENTS_GLOBAL, seed=0)
        if w:
            logging.warning(f"  {w[-1].message}")
    recon_global = reconstruction_error(X_te, ica_global)
    logging.info(f"  n_iter={ica_global.n_iter}  recon_mse(test)={recon_global:.2f}")

    # ── Per-digit ICA ─────────────────────────────────────────────────────────
    ica_per_digit: dict[int, object] = {}
    recon_per_digit: dict[int, float] = {}
    for d in range(10):
        X_d = X_tr[y_tr == d]
        logging.info(f"  Fitting ICA for digit {d} (N={len(X_d)}, n={N_COMPONENTS_DIGIT})...")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ica_per_digit[d] = fit_ica(X_d, N_COMPONENTS_DIGIT, seed=d)
            if w:
                logging.warning(f"    {w[-1].message}")
        X_d_te = X_te[y_te == d]
        recon_per_digit[d] = reconstruction_error(X_d_te, ica_per_digit[d])
        logging.info(f"    n_iter={ica_per_digit[d].n_iter}  recon_mse(test)={recon_per_digit[d]:.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_path = Path(__file__).parent / f"ica_results_{EXP_NAME}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "global": ica_global,
            "per_digit": ica_per_digit,
            "X_tr": X_tr, "y_tr": y_tr,
            "X_te": X_te, "y_te": y_te,
        }, f)
    logging.info(f"Saved to {results_path}")

    elapsed = time.perf_counter() - t0
    append_result({
        "experiment": EXP_NAME,
        "name": f"ICA global (n={N_COMPONENTS_GLOBAL}) + per-digit (n={N_COMPONENTS_DIGIT})",
        "global_n_iter": ica_global.n_iter,
        "global_recon_mse": round(recon_global, 4),
        "per_digit_recon_mse": {str(d): round(recon_per_digit[d], 4) for d in range(10)},
        "time_s": round(elapsed, 1),
    })
    logging.info(f"Done. {elapsed:.1f}s")
