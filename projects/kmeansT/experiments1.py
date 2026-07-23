"""
kmeansT — Exp 1: MLP classifier on pixel-cluster features.

Pipeline:
  1. Learn pixel→cluster assignments from training images
     (each pixel is a point in R^N, grouped by co-activation).
  2. Extract K-dim features per image: mean pixel value per cluster.
  3. Train a 2-hidden-layer JAX MLP on those K-dim features.
  4. Evaluate on MNIST test set.

Sweeps:
  - K ∈ {10, 16, 32, 48, 64}              (exp1a — vary cluster count)
  - n_cluster_samples ∈ {500,2k,10k,60k}  (exp1b — vary subset size, K=32)
  - Baseline: MLP on raw 784-dim pixels

Usage:
  uv run python projects/kmeansT/experiments1.py
"""

import io, json, sys, time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media
from shared_lib.html import save_figures_page
from shared_lib.random_utils import infinite_safe_keys_from_key
from lib.kmeans import fit as kmeans_fit, extract_features

EXP_NAME = "exp1"

# ── Hyper-parameters ────────────────────────────────────────────────────────
K_SWEEP          = [10, 16, 32, 48, 64]
SUBSET_SWEEP     = [500, 2_000, 10_000, 60_000]
K_FIXED          = 32
KMEANS_ITER      = 40

MLP_HIDDEN       = 256
N_EPOCHS         = 60
BATCH_SIZE       = 256
LR               = 1e-3
SEED             = 42
# ────────────────────────────────────────────────────────────────────────────

RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── MLP (JAX) ────────────────────────────────────────────────────────────────

def init_mlp(d_in: int, d_hidden: int = MLP_HIDDEN, seed: int = SEED) -> dict:
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(seed))
    return {
        "W1": jax.random.normal(next(kg).get(), (d_hidden, d_in))  * (2 / d_in) ** 0.5,
        "b1": jnp.zeros(d_hidden),
        "W2": jax.random.normal(next(kg).get(), (d_hidden, d_hidden)) * (2 / d_hidden) ** 0.5,
        "b2": jnp.zeros(d_hidden),
        "W3": jax.random.normal(next(kg).get(), (10, d_hidden))    * (2 / d_hidden) ** 0.5,
        "b3": jnp.zeros(10),
    }


def mlp_fwd(params: dict, x: jax.Array) -> jax.Array:
    h = jax.nn.relu(x @ params["W1"].T + params["b1"])
    h = jax.nn.relu(h @ params["W2"].T + params["b2"])
    return h @ params["W3"].T + params["b3"]


def loss_fn(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = jax.vmap(mlp_fwd, in_axes=(None, 0))(params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


def make_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return step


def accuracy(params: dict, X: np.ndarray, y: np.ndarray, batch: int = 2048) -> float:
    correct = 0
    for i in range(0, len(X), batch):
        xb = jnp.array(X[i:i+batch])
        logits = jax.vmap(mlp_fwd, in_axes=(None, 0))(params, xb)
        correct += int((jnp.argmax(logits, axis=1) == jnp.array(y[i:i+batch])).sum())
    return correct / len(X)


def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    label: str,
) -> tuple[dict, float, float]:
    d_in = X_tr.shape[1]
    params = init_mlp(d_in)
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    n = X_tr.shape[0]
    rng = np.random.default_rng(SEED)

    for epoch in range(N_EPOCHS):
        idx = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            xb = jnp.array(X_tr[b])
            yb = jnp.array(y_tr[b])
            params, opt_state, loss = train_step(params, opt_state, xb, yb)

        if (epoch + 1) % 10 == 0:
            tr_acc = accuracy(params, X_tr, y_tr)
            te_acc = accuracy(params, X_te, y_te)
            print(f"    epoch {epoch+1:3d}/{N_EPOCHS}  loss={float(loss):.4f}"
                  f"  train={tr_acc:.4f}  test={te_acc:.4f}")

    tr_acc = accuracy(params, X_tr, y_tr)
    te_acc = accuracy(params, X_te, y_te)
    return params, tr_acc, te_acc


# ── Pixel clustering ─────────────────────────────────────────────────────────

def learn_pixel_clusters(X_train: np.ndarray, K: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_train.shape[0], min(n_samples, X_train.shape[0]), replace=False)
    X_T = X_train[idx].T                     # (784, n_samples) — pixels as points
    assignments, _ = kmeans_fit(X_T, K, n_iter=KMEANS_ITER, seed=seed + K)
    return assignments                        # (784,)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _save_fig(tag: str, fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    url = save_media(f"kmeansT_{tag}.svg", buf, "image/svg+xml")
    plt.close(fig)
    return url


def plot_k_sweep(results: list[dict], baseline_acc: float) -> str:
    ks   = [r["K"]        for r in results]
    accs = [r["test_acc"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, accs, "o-", color="#4c8ef7", lw=2, ms=7, label="kmeansT MLP")
    ax.axhline(baseline_acc, color="#f7874c", lw=1.5, ls="--",
               label=f"raw pixels MLP  ({baseline_acc:.4f})")
    ax.set_xlabel("K  (number of pixel clusters)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Exp 1a — MLP accuracy vs K  (clusters learned on full train set)")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_fig("exp1a_k_sweep", fig)


def plot_subset_sweep(results: list[dict], baseline_acc: float) -> str:
    ns   = [r["n_cluster_samples"] for r in results]
    accs = [r["test_acc"]          for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(ns, accs, "s-", color="#6bcb77", lw=2, ms=7,
                label=f"kmeansT K={K_FIXED}")
    ax.axhline(baseline_acc, color="#f7874c", lw=1.5, ls="--",
               label=f"raw pixels MLP  ({baseline_acc:.4f})")
    ax.set_xlabel("Samples used to learn pixel clusters  (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Exp 1b — Accuracy vs cluster-learning subset size  (K={K_FIXED})")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_fig("exp1b_subset_sweep", fig)


def plot_cluster_maps(pixel_assignments_list: list[np.ndarray], K_list: list[int]) -> str:
    import colorsys
    n = len(K_list)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, assignments, K in zip(axes, pixel_assignments_list, K_list):
        colors = np.array([colorsys.hsv_to_rgb(i / K, 0.7, 0.85) for i in range(K)])
        img = colors[assignments].reshape(28, 28, 3)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"K={K}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Pixel cluster maps (learned on full train set)", fontsize=10)
    fig.tight_layout()
    return _save_fig("exp1_cluster_maps", fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading MNIST …")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(np.float32) / 255.0
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(np.float32) / 255.0
    y_train = np.array(data.y,      dtype=np.int32)
    y_test  = np.array(data.y_test, dtype=np.int32)
    print(f"  train {X_train.shape}  test {X_test.shape}")

    # ── Baseline: raw 784-dim pixels ─────────────────────────────────────
    print(f"\n─── Baseline: raw 784-dim pixels  (MLP {MLP_HIDDEN}×{MLP_HIDDEN}) ───")
    t0 = time.perf_counter()
    _, baseline_train, baseline_test = train_mlp(X_train, y_train, X_test, y_test, "baseline")
    elapsed = time.perf_counter() - t0
    print(f"  → train={baseline_train:.4f}  test={baseline_test:.4f}  ({elapsed:.1f}s)")
    append_result({"exp": EXP_NAME, "variant": "baseline", "K": 784,
                   "n_cluster_samples": None, "train_acc": baseline_train,
                   "test_acc": baseline_test, "elapsed_s": round(elapsed, 2)})

    # ── Exp 1a: sweep K ───────────────────────────────────────────────────
    print("\n─── Exp 1a: sweep K (clusters from full train set) ───")
    exp1a, pixel_maps = [], []

    for K in K_SWEEP:
        print(f"\n  K={K} …")
        t0 = time.perf_counter()

        px = learn_pixel_clusters(X_train, K, n_samples=X_train.shape[0], seed=SEED)
        F_tr = extract_features(X_train, px, K)
        F_te = extract_features(X_test,  px, K)

        _, tr_acc, te_acc = train_mlp(F_tr, y_train, F_te, y_test, f"exp1a_K{K}")
        elapsed = time.perf_counter() - t0
        print(f"  → train={tr_acc:.4f}  test={te_acc:.4f}  ({elapsed:.1f}s)")

        rec = {"exp": EXP_NAME, "variant": "exp1a", "K": K,
               "n_cluster_samples": X_train.shape[0],
               "train_acc": tr_acc, "test_acc": te_acc,
               "baseline_test": baseline_test, "elapsed_s": round(elapsed, 2)}
        append_result(rec)
        exp1a.append(rec)
        pixel_maps.append(px)

    # ── Exp 1b: sweep subset size (K fixed) ──────────────────────────────
    print(f"\n─── Exp 1b: sweep subset size  (K={K_FIXED}) ───")
    exp1b = []

    for n_samp in SUBSET_SWEEP:
        print(f"\n  n_cluster_samples={n_samp} …")
        t0 = time.perf_counter()

        px = learn_pixel_clusters(X_train, K_FIXED, n_samples=n_samp, seed=SEED)
        F_tr = extract_features(X_train, px, K_FIXED)
        F_te = extract_features(X_test,  px, K_FIXED)

        _, tr_acc, te_acc = train_mlp(F_tr, y_train, F_te, y_test, f"exp1b_n{n_samp}")
        elapsed = time.perf_counter() - t0
        print(f"  → train={tr_acc:.4f}  test={te_acc:.4f}  ({elapsed:.1f}s)")

        rec = {"exp": EXP_NAME, "variant": "exp1b", "K": K_FIXED,
               "n_cluster_samples": n_samp,
               "train_acc": tr_acc, "test_acc": te_acc,
               "elapsed_s": round(elapsed, 2)}
        append_result(rec)
        exp1b.append(rec)

    # ── Visualize ─────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    figures = []

    url = plot_k_sweep(exp1a, baseline_test)
    figures.append(("Exp 1a — Accuracy vs K", url))
    print(f"  exp1a → {url}")

    url = plot_subset_sweep(exp1b, baseline_test)
    figures.append(("Exp 1b — Accuracy vs subset size", url))
    print(f"  exp1b → {url}")

    url = plot_cluster_maps(pixel_maps, K_SWEEP)
    figures.append(("Pixel cluster maps", url))
    print(f"  maps  → {url}")

    page_url = save_figures_page(
        f"kmeansT_{EXP_NAME}",
        "kmeansT Exp 1 — MLP on pixel-cluster features",
        figures,
    )
    print(f"\nFigures page → {page_url}")

    # ── Summary ───────────────────────────────────────────────────────────
    best1a = max(exp1a, key=lambda r: r["test_acc"])
    best1b = max(exp1b, key=lambda r: r["test_acc"])
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  baseline (784-dim)       test={baseline_test:.4f}")
    print(f"  exp1a best (K={best1a['K']:2d})        test={best1a['test_acc']:.4f}")
    print(f"  exp1b best (n={best1b['n_cluster_samples']:6d})   test={best1b['test_acc']:.4f}")


if __name__ == "__main__":
    run()
