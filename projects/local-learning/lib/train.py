"""One training driver for all four learning rules, so the timings compare.

A `Cell` is one (method, architecture, seed) run. `run_cells` executes a list of
them, skipping any already in `results.jsonl`, and appends one row per cell.

Timing method. Every number the tweet reports is a wall clock, so the harness
has to be explicit about what it excludes:

* `compile_s` — XLA compilation, measured on its own by lowering and compiling
  ahead of time. Excluded from the training figure, as it must be: it is a
  constant few seconds that would swamp a 0.3 s training run.
* `train_s` — the compiled epochs, blocked on until the device is idle.
* `infer_s` — one pass over all 10,000 test images, compiled first, then the
  fastest of three runs. Forward-Forward pays ten forward passes here.

Everything runs on one GPU in float32, and the whole epoch is a single
`lax.scan` for every method, so no method is charged for Python dispatch that
another avoids.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib import ff as ffmod
from lib.es import es_gradient, make_es_epoch
from lib.fa import alignment, init_feedback, plain_mm, residual_mm
from lib.data import Data, overlay_label
from lib.methods import bp_loss, grad_cosine, method_grad
from lib.model import (Params, mlp_forward, mlp_init, n_params, relu, res_init,
                       res_logits, res_scales, res_skips)

JSONL = Path(__file__).parent.parent / "results.jsonl"
RESIDUAL_METHODS = ("bp", "pc", "pcalm", "eggroll", "fa")
ES_METHODS = ("eggroll", "eggroll_plain")


@dataclass(frozen=True)
class Cell:
    exp: str                      # which experiment this cell belongs to
    method: str                   # bp | pc | pcalm | ff | bp_plain
    seed: int = 0
    # residual MLP (bp / pc / pcalm)
    width: int = 32
    depth: int = 32
    # plain MLP (ff / bp_plain); hidden layer widths
    hidden: tuple[int, ...] = (66, 66)
    # optimisation
    epochs: int = 1
    batch: int = 64
    lr: float | None = None       # None -> eta0 * sqrt(width/depth), the paper's rule
    eta0: float = 1e-3
    # inference (pc / pcalm)
    budget: int | None = None     # None -> 2 * depth, the paper's T = 2L
    alpha: float = 1.0
    rho: float = 1.0
    state_lr: float | None = None # None -> 1 / lambda_max, measured
    # forward-forward
    theta: float = 2.0
    hard_negatives: bool = False  # sample the wrong label the net finds plausible
    # eggroll (evolution strategies)
    sigma: float = 3e-3           # perturbation scale
    n_pop: int = 32               # antithetic pairs per weight update
    rank: int = 1                 # rank of each perturbation
    # data
    n_train: int = 60000
    n_test: int = 10000
    dataset: str = "mnist"
    tag: str = ""                 # distinguishes cells that differ in a swept field

    @property
    def name(self) -> str:
        if self.method in RESIDUAL_METHODS:
            arch = f"L{self.depth}N{self.width}"
        elif len(set(self.hidden)) == 1:
            arch = f"{len(self.hidden)}x{self.hidden[0]}"
        else:
            arch = "x".join(str(h) for h in self.hidden)
        bits = [self.exp, self.method, arch]
        if self.dataset != "mnist":
            bits.append(self.dataset)
        if self.tag:
            bits.append(self.tag)
        bits.append(f"s{self.seed}")
        return "_".join(bits)


# ── helpers ──────────────────────────────────────────────────────────────────

def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_experiments() -> set[str]:
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass
    return done


def batched(x: np.ndarray, batch: int) -> jnp.ndarray:
    n = (x.shape[0] // batch) * batch
    return jnp.asarray(x[:n].reshape(n // batch, batch, *x.shape[1:]))


def lambda_max(params, scales, skips, x, y, rho: float, key, iters: int = 60) -> float:
    """Largest curvature of the inference energy, by power iteration.

    The paper sets the activity step to eta_h = 1 / lambda_max, where lambda_max
    is the largest eigenvalue of the Hessian of the per-sample energy with
    respect to the activations. Measuring it costs a second and removes any
    dependence on the authors' per-cell lookup table.
    """
    from lib.methods import al_energy, free_init, residuals

    free0 = free_init(params, scales, skips, x, relu)
    duals = [jnp.zeros_like(r) for r in residuals(params, scales, skips, x, free0, relu)]
    batch = x.shape[0]

    def energy(f):
        return al_energy(params, scales, skips, x, y, f, duals, rho, relu) * batch

    grad_e = jax.grad(energy)

    @jax.jit
    def hvp(v):
        return jax.jvp(grad_e, (free0,), (v,))[1]

    keys = jax.random.split(key, len(free0))
    v = [jax.random.normal(k, z.shape) for k, z in zip(keys, free0)]
    norm = jnp.sqrt(sum(jnp.sum(u * u) for u in v))
    v = [u / norm for u in v]
    for _ in range(iters):
        w = hvp(v)
        norm = jnp.sqrt(sum(jnp.sum(u * u) for u in w))
        v = [u / norm for u in w]
    return float(norm)


def _time_compiled(fn, *args):
    """(compile_seconds, compiled_callable)."""
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    return time.perf_counter() - t0, compiled


def _fastest(fn, *args, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best


# ── residual-MLP training: bp / pc / pcalm ───────────────────────────────────

def train_residual(cell: Cell, data: Data) -> dict:
    scales = res_scales(cell.width, cell.depth, 784)
    skips = res_skips(cell.depth)
    budget = cell.budget if cell.budget is not None else 2 * cell.depth
    lr = cell.lr if cell.lr is not None else cell.eta0 * np.sqrt(cell.width / cell.depth)

    key = jax.random.PRNGKey(cell.seed)
    k_init, k_power, k_perm = jax.random.split(key, 3)
    params = res_init(k_init, depth=cell.depth, width=cell.width)

    state_lr = cell.state_lr
    if state_lr is None and cell.method in ("pc", "pcalm"):
        lam = lambda_max(params, scales, skips,
                         data.x_train[: cell.batch], data.y_train[: cell.batch],
                         cell.rho, k_power)
        state_lr = 1.0 / lam
        logging.info(f"  lambda_max={lam:.4f}  state_lr={state_lr:.6f}")
    state_lr = state_lr or 0.0

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    feedback = init_feedback(jax.random.fold_in(key, 5), params) if cell.method == "fa" else None
    mm = residual_mm(feedback) if feedback is not None else None

    def loss_of(p, x, y):
        return bp_loss(p, scales, skips, x, y, relu, mm)

    if cell.method == "eggroll":
        epoch = make_es_epoch(loss_of, opt, sigma=cell.sigma, n_pop=cell.n_pop,
                              rank=cell.rank)
    else:
        def step(carry, batch):
            p, s = carry
            x, y = batch
            if cell.method == "fa":
                # the only difference from bp: the gradient comes back through
                # fixed random matrices rather than the forward weights
                g = jax.grad(loss_of)(p, x, y)
            else:
                g = method_grad(p, scales, skips, x, y, method=cell.method, alpha=cell.alpha,
                                rho=cell.rho, state_lr=state_lr, budget=budget, phi=relu)
            updates, s = opt.update(g, s, p)
            return (optax.apply_updates(p, updates), s), 0.0

        def epoch(p, s, key, xb, yb):
            (p, s), _ = jax.lax.scan(step, (p, s), (xb, yb))
            return p, s, key

    def accuracy(p, x, labels):
        return jnp.mean(jnp.argmax(res_logits(p, scales, skips, x, relu), axis=-1) == labels)

    perm = np.asarray(jax.random.permutation(k_perm, data.x_train.shape[0]))
    xb = batched(np.asarray(data.x_train)[perm], cell.batch)
    yb = batched(np.asarray(data.y_train)[perm], cell.batch)

    es_key = jax.random.fold_in(key, 99)
    compile_s, epoch_fn = _time_compiled(epoch, params, opt_state, es_key, xb, yb)
    _, acc_fn = _time_compiled(accuracy, params, data.x_test, data.l_test)

    history, train_s = [], 0.0
    for ep in range(cell.epochs):
        if ep > 0:      # reshuffle; shapes are unchanged so nothing recompiles
            perm = np.random.default_rng(cell.seed + ep).permutation(data.x_train.shape[0])
            xb = batched(np.asarray(data.x_train)[perm], cell.batch)
            yb = batched(np.asarray(data.y_train)[perm], cell.batch)
        t0 = time.perf_counter()
        params, opt_state, es_key = epoch_fn(params, opt_state, es_key, xb, yb)
        jax.block_until_ready(params)
        train_s += time.perf_counter() - t0
        acc = float(acc_fn(params, data.x_test, data.l_test))
        history.append({"epoch": ep + 1, "test_acc": acc, "train_s": train_s})
        logging.info(f"  epoch {ep + 1}/{cell.epochs}  test_acc={acc:.4f}  train_s={train_s:.2f}")

    infer_s = _fastest(acc_fn, params, data.x_test, data.l_test)
    test_acc = float(acc_fn(params, data.x_test, data.l_test))
    n_eval = data.x_test.shape[0]      # the eval fn is compiled for this shape
    train_acc = float(acc_fn(params, data.x_train[:n_eval], data.l_train[:n_eval]))

    xb0, yb0 = data.x_train[: cell.batch], data.y_train[: cell.batch]
    if cell.method == "eggroll":
        g_method = es_gradient(jax.random.fold_in(key, 7), params,
                               lambda p: loss_of(p, xb0, yb0),
                               sigma=cell.sigma, n_pop=cell.n_pop, rank=cell.rank)
    elif cell.method == "fa":
        g_method = jax.grad(loss_of)(params, xb0, yb0)
    else:
        g_method = method_grad(params, scales, skips, xb0, yb0, method=cell.method,
                               alpha=cell.alpha, rho=cell.rho, state_lr=state_lr,
                               budget=budget, phi=relu)
    g_bp = jax.grad(lambda p: bp_loss(p, scales, skips, xb0, yb0, relu))(params)

    return {
        "n_params": n_params(params), "n_params_head": 0,
        "test_acc": test_acc, "test_acc_probe": None, "train_acc": train_acc,
        "train_s": train_s, "compile_s": compile_s, "infer_s": infer_s,
        "grad_cos_to_bp": float(grad_cosine(g_method, g_bp)),
        "feedback_alignment": (float(alignment(params, feedback))
                               if feedback is not None else None),
        "lr": float(lr), "state_lr": float(state_lr), "budget": int(budget),
        "history": history,
    }


# ── plain-MLP training: ff / bp_plain ────────────────────────────────────────

def train_plain(cell: Cell, data: Data) -> dict:
    key = jax.random.PRNGKey(cell.seed)
    k_init, k_head, k_perm, k_neg = jax.random.split(key, 4)
    is_ff = cell.method == "ff"

    sizes = [784, *cell.hidden] + ([] if is_ff else [10])
    params = mlp_init(k_init, sizes)
    lr = cell.lr if cell.lr is not None else cell.eta0
    opt = optax.adam(lr)

    feat_dim = sum(cell.hidden[1:]) if len(cell.hidden) > 1 else cell.hidden[0]
    head = [jax.random.normal(k_head, (10, feat_dim)) * 0.01, jnp.zeros((10,))] if is_ff else []

    if is_ff:
        opt_state = opt.init((params, head))

        def step(carry, batch):
            (p, hd), s, k = carry
            x, y, yneg, labels = batch
            # hard negatives depend on the current weights, so they cannot be
            # drawn ahead of the epoch the way uniform ones can
            k, sub = jax.random.split(k)
            yneg = ffmod.hard_negatives(p, x, labels, sub) if cell.hard_negatives else yneg

            def loss(ph):
                p_, h_ = ph
                return (ffmod.ff_loss(p_, overlay_label(x, y), overlay_label(x, yneg), cell.theta)
                        + ffmod.probe_loss(h_, p_, ffmod.neutral_input(x), y))
            g = jax.grad(loss)((p, hd))
            updates, s = opt.update(g, s, (p, hd))
            return (optax.apply_updates((p, hd), updates), s, k), 0.0

        def epoch(p, hd, s, key_, xb, yb, nb, lb):
            ((p, hd), s, key_), _ = jax.lax.scan(step, ((p, hd), s, key_), (xb, yb, nb, lb))
            return p, hd, s, key_

        def accuracy(p, x, labels):
            return jnp.mean(ffmod.predict_goodness(p, x) == labels)

        def accuracy_probe(p, hd, x, labels):
            return jnp.mean(ffmod.predict_probe(hd, p, x) == labels)
    else:
        opt_state = opt.init(params)

        fb = init_feedback(jax.random.fold_in(key, 5), params) if cell.method == "fa_plain" else None
        pmm = plain_mm(fb) if fb is not None else None

        def plain_loss(p_, x, y):
            return 0.5 * jnp.mean(jnp.sum((mlp_forward(p_, x, mm=pmm) - y) ** 2, axis=-1))

        if cell.method == "eggroll_plain":
            epoch = make_es_epoch(plain_loss, opt, sigma=cell.sigma,
                                  n_pop=cell.n_pop, rank=cell.rank)
        else:
            def step(carry, batch):
                p, s = carry
                x, y = batch
                g = jax.grad(plain_loss)(p, x, y)
                updates, s = opt.update(g, s, p)
                return (optax.apply_updates(p, updates), s), 0.0

            def epoch(p, s, key_, xb, yb):
                (p, s), _ = jax.lax.scan(step, (p, s), (xb, yb))
                return p, s, key_

        def accuracy(p, x, labels):
            return jnp.mean(jnp.argmax(mlp_forward(p, x), axis=-1) == labels)

    def epoch_data(ep: int):
        rng = np.random.default_rng(cell.seed * 1000 + ep)
        perm = rng.permutation(data.x_train.shape[0])
        xb = batched(np.asarray(data.x_train)[perm], cell.batch)
        yb = batched(np.asarray(data.y_train)[perm], cell.batch)
        if not is_ff:
            return xb, yb
        labels = np.asarray(data.l_train)[perm]
        neg = ffmod.make_negatives(jax.random.fold_in(k_neg, ep), jnp.asarray(labels))
        return xb, yb, batched(np.asarray(neg), cell.batch), batched(labels, cell.batch)

    args = epoch_data(0)
    es_key = jax.random.fold_in(key, 99)
    if is_ff:
        compile_s, epoch_fn = _time_compiled(epoch, params, head, opt_state, es_key, *args)
        _, acc_fn = _time_compiled(accuracy, params, data.x_test, data.l_test)
        _, probe_fn = _time_compiled(accuracy_probe, params, head, data.x_test, data.l_test)
    else:
        compile_s, epoch_fn = _time_compiled(epoch, params, opt_state, es_key, *args)
        _, acc_fn = _time_compiled(accuracy, params, data.x_test, data.l_test)

    history, train_s = [], 0.0
    for ep in range(cell.epochs):
        if ep > 0:
            args = epoch_data(ep)
        t0 = time.perf_counter()
        if is_ff:
            params, head, opt_state, es_key = epoch_fn(params, head, opt_state, es_key, *args)
            jax.block_until_ready(params)
        else:
            params, opt_state, es_key = epoch_fn(params, opt_state, es_key, *args)
            jax.block_until_ready(params)
        train_s += time.perf_counter() - t0
        acc = float(acc_fn(params, data.x_test, data.l_test))
        history.append({"epoch": ep + 1, "test_acc": acc, "train_s": train_s})
        logging.info(f"  epoch {ep + 1}/{cell.epochs}  test_acc={acc:.4f}  train_s={train_s:.2f}")

    infer_s = _fastest(acc_fn, params, data.x_test, data.l_test)
    test_acc = float(acc_fn(params, data.x_test, data.l_test))
    n_eval = data.x_test.shape[0]      # the eval fn is compiled for this shape
    train_acc = float(acc_fn(params, data.x_train[:n_eval], data.l_train[:n_eval]))
    probe_acc = float(probe_fn(params, head, data.x_test, data.l_test)) if is_ff else None

    cos = None
    align = float(alignment(params, fb)) if cell.method == "fa_plain" else None
    if cell.method == "eggroll_plain":
        xb0, yb0 = data.x_train[: cell.batch], data.y_train[: cell.batch]
        g_es = es_gradient(jax.random.fold_in(key, 7), params,
                           lambda p: plain_loss(p, xb0, yb0),
                           sigma=cell.sigma, n_pop=cell.n_pop, rank=cell.rank)
        g_true = jax.grad(plain_loss)(params, xb0, yb0)
        cos = float(grad_cosine(g_es, g_true))

    return {
        "n_params": n_params(params),
        "n_params_head": n_params(head) if is_ff else 0,
        "test_acc": test_acc, "test_acc_probe": probe_acc, "train_acc": train_acc,
        "train_s": train_s, "compile_s": compile_s, "infer_s": infer_s,
        "grad_cos_to_bp": cos,
        "feedback_alignment": align,
        "lr": float(lr), "state_lr": None, "budget": None,
        "history": history,
    }


# ── driver ───────────────────────────────────────────────────────────────────

def run_cells(cells: list[Cell], data: Data | None = None) -> None:
    from lib.data import load_mnist

    todo = [c for c in cells if c.name not in done_experiments()]
    logging.info(f"{len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    loaded, given = {}, data
    for i, cell in enumerate(todo, 1):
        key = (cell.dataset, cell.n_train, cell.n_test)
        if given is not None:
            data = given
        elif key not in loaded:
            loaded[key] = load_mnist(cell.n_train, cell.n_test, seed=0, dataset=cell.dataset)
            data = loaded[key]
            logging.info(f"{cell.dataset}: train {data.x_train.shape}, test {data.x_test.shape}")
        else:
            data = loaded[key]
        logging.info(f"[{i}/{len(todo)}] {cell.name}")
        t0 = time.perf_counter()
        out = (train_residual if cell.method in RESIDUAL_METHODS else train_plain)(cell, data)
        row = {"experiment": cell.name, "name": cell.name, **asdict(cell), **out,
               "time_s": time.perf_counter() - t0}
        row["hidden"] = list(cell.hidden)
        append_result(row)
        logging.info(f"  -> acc={out['test_acc']:.4f}  train={out['train_s']:.3f}s  "
                     f"infer={out['infer_s']:.4f}s  params={out['n_params']}")
