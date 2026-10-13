"""Figures for the bayes-cbp-2 paper.

No neural-network experiments: every curve is either a closed-form expression
derived in the paper or a small Monte Carlo / finite-difference check of one.
Run on the GPU box (numpy lives there):

    ~/.local/bin/uv run --no-sync python projects/bayes-cbp-2/scripts/gen_figures.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT.parent.parent))   # append, never insert(0): stray datasets.py on the box

from shared_lib.typst_plot import line_chart, long_form, write_figures  # noqa: E402

PAPER = PROJECT / "paper"
rng = np.random.default_rng(0)


def columns(series: dict[str, tuple], x_name: str, y_name: str) -> dict[str, list]:
    """Long form where each series has its own x values."""
    out = {x_name: [], y_name: [], "series": []}
    for label, (xs, ys) in series.items():
        out[x_name] += [float(v) for v in xs]
        out[y_name] += [float(v) for v in ys]
        out["series"] += [label] * len(xs)
    return out


def phi(z):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))


# ── §3: Kalman gain with and without process noise ──────────────────────────

def kalman_gain():
    """Gain k_t = weight on the newest observation, scalar Gaussian model (§3.4)."""
    r, s0 = 1.0, 1.0
    t = np.unique(np.round(np.logspace(0, 4, 81)).astype(int))
    series, hl = {}, []
    for q, label in ((0.0, "static (q = 0)"), (1e-4, "q = 1e-4"), (1e-2, "q = 1e-2")):
        P, ks = s0, []
        for _ in range(t[-1]):
            Pm = P + q
            k = Pm / (Pm + r)
            P = (1 - k) * Pm
            ks.append(k)
        series[label] = np.log10(np.array(ks)[t - 1])
        if q > 0:
            Pm = (q + np.sqrt(q * q + 4 * q * r)) / 2
            hl.append((float(np.log10(Pm / (Pm + r))), "k∞"))
    return line_chart(
        "kalman_gain", long_form(t, series, x_name="t", y_name="log10_k"),
        x="t", y="log10_k", colour="series", log_x=True,
        x_label="observations seen, t", y_label="log10 gain k_t", colour_label="",
        hlines=hl,
    )


# ── §6: soft vs hard reset, one dimension ───────────────────────────────────

def soft_vs_hard():
    """CDF of θ at t = 1 from θ0 = 3: hard reset, two soft resets, and the OU limit (§6.5)."""
    theta0, t, n = 3.0, 1.0, 200_000
    x = np.linspace(-4, 5, 181)
    ecdf = lambda s: np.searchsorted(np.sort(s), x, side="right") / len(s)
    series = {}
    # hard reset, rate 1, to N(0, 1): exact mixture
    w = np.exp(-t)
    series["hard reset, exact"] = (1 - w) * phi(x) + w * (x >= theta0)
    k = rng.poisson(t, n)
    th = np.where(k > 0, rng.normal(0, 1, n), theta0)
    series["hard reset, Monte Carlo"] = ecdf(th)
    # soft reset θ ← (1−a)θ + aξ at rate λ = 1/a, ξ ~ N(0, s²) with λa²s² = 2
    for a in (0.5, 0.1):
        lam, s2 = 1 / a, 2 / (1 / a * a * a)
        th = np.full(n, theta0)
        k = rng.poisson(lam * t, n)
        for j in range(k.max()):
            m = k > j
            th[m] = (1 - a) * th[m] + a * rng.normal(0, np.sqrt(s2), m.sum())
        series[f"soft reset a = {a:g}, Monte Carlo"] = ecdf(th)
    # OU limit γ = 1, D = 1
    series["OU limit (a → 0)"] = phi((x - theta0 * np.exp(-t)) / np.sqrt(1 - np.exp(-2 * t)))
    return line_chart(
        "soft_vs_hard", long_form(x, series, x_name="theta", y_name="cdf"),
        x="theta", y="cdf", colour="series",
        x_label="θ at t = 1 (started at θ₀ = 3, reset centre 0)", y_label="P(θ_t ≤ θ)",
        colour_label="",
    )


# ── §6–7: does a reset keep the posterior? double well ──────────────────────

H, LAM, P0_MU = 4.0, 2.0, 0.5            # barrier height, reset rate, reset centre
U = lambda th: H * (th**2 - 1) ** 2
dU = lambda th: 4 * H * th * (th**2 - 1)
log_p0 = lambda th: -0.5 * (th - P0_MU) ** 2


def double_well_runs():
    n, dt, t_end = 20_000, 1e-3, 30.0
    names = ("Langevin only", "Langevin + naive reset", "Langevin + Metropolised reset")
    th = np.full((3, n), -1.0)
    times, occ = [0.0], [[0.0] * 3]
    steps = int(t_end / dt)
    for k in range(1, steps + 1):
        th += -dU(th) * dt + np.sqrt(2 * dt) * rng.normal(size=th.shape)
        for v in (1, 2):
            hit = rng.random(n) < LAM * dt
            prop = rng.normal(P0_MU, 1.0, hit.sum())
            cur = th[v, hit]
            if v == 2:   # independence Metropolis–Hastings: target e^{-U}, proposal p0
                log_a = (-U(prop) + log_p0(cur)) - (-U(cur) + log_p0(prop))
                ok = np.log(rng.random(hit.sum())) < log_a
                prop = np.where(ok, prop, cur)
            th[v, hit] = prop
        if k % 250 == 0:
            times.append(k * dt)
            occ.append(list((th > 0).mean(axis=1)))
    return names, th, np.array(times), np.array(occ)


def double_well_figures():
    names, th, times, occ = double_well_runs()
    edges = np.linspace(-2, 2, 51)
    c = 0.5 * (edges[1:] + edges[:-1])
    grid = np.linspace(-3, 3, 60001)
    z = np.exp(-U(grid)).sum() * (grid[1] - grid[0])
    series = {"posterior π ∝ exp(−U)": np.exp(-U(c)) / z}
    for name, s in zip(names, th):
        series[name] = np.histogram(s, bins=edges, density=True)[0]
    stat = line_chart(
        "dw_stationary", long_form(c, series, x_name="theta", y_name="density"),
        x="theta", y="density", colour="series",
        x_label="θ  (U = 4(θ² − 1)², reset proposals from N(0.5, 1) at rate 2)",
        y_label="density at t = 30", colour_label="",
    )
    occ_series = {name: occ[:, i] for i, name in enumerate(names)}
    rel = line_chart(
        "dw_occupancy", long_form(times, occ_series, x_name="t", y_name="frac"),
        x="t", y="frac", colour="series",
        x_label="time t (all particles start at θ = −1)",
        y_label="fraction in right well θ > 0", colour_label="",
        hlines=[(0.5, "π")], y_limits=(0, 0.8),
    )
    return stat, rel


# ── §7: mean first-passage time with resetting ──────────────────────────────

def mfpt(force, r, reset, lo=-3.0, hi=1.0, n=801):
    """Solve T'' + F T' − r T + r R[T] = −1 on [lo, hi], T(hi) = 0, T'(lo) = 0.

    R[T] = T(x_r) for a fixed reset point, or ∫ p0 T for a reset distribution
    (p0 = N(0, 1); its mass beyond `hi` lands in the target, T = 0 there).
    Unit temperature: dθ = F dt + √2 dW.
    """
    x = np.linspace(lo, hi, n)
    h = x[1] - x[0]
    F = force(x)
    A = np.zeros((n, n))
    for i in range(n - 1):
        left, right = (i + 1, i + 1) if i == 0 else (i - 1, i + 1)
        A[i, left] += 1 / h**2 - F[i] / (2 * h)
        A[i, right] += 1 / h**2 + F[i] / (2 * h)
        A[i, i] += -2 / h**2 - r
    if isinstance(reset, float):
        w = np.zeros(n); w[np.argmin(np.abs(x - reset))] = 1.0
    else:
        w = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) * h
        w[0] += phi(np.array([lo]))[0]         # left tail → reflecting node
        w[-1] = 0.0
    A[: n - 1] += r * w[None, :]
    A[n - 1, n - 1] = 1.0
    b = -np.ones(n); b[-1] = 0.0
    T = np.linalg.solve(A, b)
    return float(np.interp(-1.0, x, T))


def mfpt_mc(force, r, n=4000, dt=1e-3, t_max=80.0):
    """Monte Carlo check: start at −1, reset to N(0, 1) at rate r, stop at θ ≥ 1."""
    th = np.full(n, -1.0)
    t_hit = np.full(n, np.nan)
    alive = np.ones(n, bool)
    for k in range(1, int(t_max / dt) + 1):
        i = np.flatnonzero(alive)
        if i.size == 0:
            break
        x = th[i] + force(th[i]) * dt + np.sqrt(2 * dt) * rng.normal(size=i.size)
        x = np.where(x < -3, -6 - x, x)                    # reflect at −3
        hit = rng.random(i.size) < r * dt
        x[hit] = rng.normal(0, 1, hit.sum())
        th[i] = x
        done = x >= 1
        t_hit[i[done]] = k * dt
        alive[i[done]] = False
    return float(np.nanmean(t_hit)), float(np.mean(~np.isnan(t_hit)))


def mfpt_figure():
    barrier = lambda x: -dU(x)
    downhill = lambda x: np.full_like(x, 2.0)
    rs = np.logspace(-3, 1, 33)
    series = {
        "double well, reset to start θ = −1": (rs, [mfpt(barrier, r, -1.0) for r in rs]),
        "double well, reset to N(0, 1)": (rs, [mfpt(barrier, r, "p0") for r in rs]),
        "constant slope, reset to start": (rs, [mfpt(downhill, r, -1.0) for r in rs]),
        "constant slope, reset to N(0, 1)": (rs, [mfpt(downhill, r, "p0") for r in rs]),
    }
    r_mc = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    mc = [mfpt_mc(barrier, r) for r in r_mc]
    for r, (m, frac) in zip(r_mc, mc):
        print(f"  MC double well, reset to N(0,1), r={r:g}: T={m:.3f}  finished={frac:.4f}")
    series["Monte Carlo check (double well, N(0, 1))"] = (r_mc, [m for m, _ in mc])
    data = columns({k: (x, np.log10(y)) for k, (x, y) in series.items()}, "r", "log10_T")
    print("  r=0 limits:", {k: round(v[1][0], 3) for k, v in series.items() if len(v[0]) > 10})
    return line_chart(
        "mfpt_reset", data, x="r", y="log10_T", colour="series", log_x=True, points=True,
        x_label="reset rate r", y_label="log10 mean time to reach θ ≥ 1 from θ = −1",
        colour_label="",
    )


# ── §8: task switch, quadratic model ────────────────────────────────────────

def pareto_figure():
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    R = np.array([[c, -s], [s, c]])
    HA = R @ np.diag([1.0, 0.05]) @ R.T
    HB = np.diag([1.0, 0.02])
    tA, tB = np.zeros(2), np.array([2.0, 1.0])
    LA = lambda th: 0.5 * (th - tA) @ HA @ (th - tA)
    LB = lambda th: 0.5 * (th - tB) @ HB @ (th - tB)
    series = {}
    betas = np.logspace(-3, 3, 121)
    pts = [np.linalg.solve(HB + b * HA, HB @ tB + b * HA @ tA) for b in betas]
    series["Pareto front = EWC with β swept"] = ([LB(p) for p in pts], [LA(p) for p in pts])
    ts = np.concatenate([[0], np.logspace(-2, np.log10(400), 160)])
    h = np.diag(HB)
    gf = [tB + np.exp(-h * t) * (tA - tB) for t in ts]
    series["gradient flow on L_B, stopped at t"] = ([LB(p) for p in gf], [LA(p) for p in gf])
    T = 0.05
    lx, ly = [], []
    for t, p in zip(ts, gf):
        C = np.diag(T * (1 - np.exp(-2 * h * t)) / h)
        lx.append(LB(p) + 0.5 * np.trace(HB @ C)); ly.append(LA(p) + 0.5 * np.trace(HA @ C))
    series["Langevin on L_B, T = 0.05 (expected)"] = (lx, ly)
    return line_chart(
        "pareto", columns(series, "new_loss", "old_loss"),
        x="new_loss", y="old_loss", colour="series",
        x_label="new-task excess loss L_B", y_label="old-task excess loss L_A (forgetting)",
        colour_label="",
    )


figs = [kalman_gain(), soft_vs_hard(), *double_well_figures(), mfpt_figure(), pareto_figure()]
# Both Typst trees read the same figures: paper/ (thematic) and steps/ (stage by stage).
for tree in (PAPER, PROJECT / "steps"):
    for p in write_figures(tree, figs):
        print(p.relative_to(PROJECT))
