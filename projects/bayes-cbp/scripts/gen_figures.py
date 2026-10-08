"""Figures for the bayes-cbp formulation paper.

No experiments: every curve is a closed-form expression from the paper, plus
two small Monte Carlo checks of the jump-process equations (sections 3.3, 3.6).
Run on the GPU box (numpy lives there):

    ~/.local/bin/uv run --no-sync python projects/bayes-cbp/scripts/gen_figures.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT.parent.parent))   # append, never insert(0): stray datasets.py on the box

from shared_lib.typst_plot import line_chart, long_form, write_figures  # noqa: E402

PAPER = PROJECT / "paper"
rng = np.random.default_rng(0)


def escape_times():
    """log10 of the waiting time: Kramers vs a Poisson reset (section 3.4)."""
    x = np.linspace(0, 12, 49)
    lam_alpha = 0.01
    pi_A = 0.5                      # chance a fresh draw lands back in the old basin
    data = long_form(x, {
        "朗之万 exp(ΔU/T)": x / np.log(10),
        "跳跃 1/(λα)": np.full_like(x, np.log10(1 / lam_alpha)),
        "跳跃 1/(λα(1−π_A))": np.full_like(x, np.log10(1 / (lam_alpha * (1 - pi_A)))),
    }, x_name="barrier", y_name="log10_tau")
    return line_chart(
        "escape_times", data, x="barrier", y="log10_tau", colour="series",
        x_label="势垒高度 ΔU/T", y_label="log10 等待时间（单位：1/前置因子）",
        colour_label="",
    )


def jump_variance():
    """Var(t) under pure reset jumps: Monte Carlo vs two candidate formulas (3.6)."""
    lam_alpha, mu0, s0 = 1.0, 1.0, 0.5
    m0, v0 = -2.0, 4.0
    n, dt, steps = 200_000, 1e-3, 3000
    theta = rng.normal(m0, np.sqrt(v0), n)
    ts, mc = [0.0], [theta.var()]
    for k in range(1, steps + 1):
        hit = rng.random(n) < lam_alpha * dt
        theta[hit] = rng.normal(mu0, s0, hit.sum())
        if k % 100 == 0:
            ts.append(k * dt); mc.append(theta.var())
    # corrected: dV = λα(σ0² − V + (μ0 − m)²);  note: dV = λα E[(ξ−θ)²] = λα(σ0² + V + (μ0 − m)²)
    def integrate(sign):
        m, v, out = m0, v0, [v0]
        for k in range(1, steps + 1):
            dv = lam_alpha * (s0**2 + sign * v + (mu0 - m) ** 2)
            m += lam_alpha * (mu0 - m) * dt; v += dv * dt
            if k % 100 == 0:
                out.append(v)
        return out
    data = long_form(ts, {
        "蒙特卡洛（20 万粒子）": mc,
        "修正式 λα(σ₀²−Var+(μ₀−m)²)": integrate(-1),
        "原文式 λα E[(ξ−θ)²]": integrate(+1),
    }, x_name="t", y_name="var")
    return line_chart(
        "jump_variance", data, x="t", y="var", colour="series", points=False,
        x_label="时间 t（λα = 1）", y_label="Var(θ)", colour_label="",
        y_limits=(0, 12),
    )


def state_dependent_stationary():
    """Stationary density with state-dependent α and no drift (3.3)."""
    s_init = 1.5
    alpha = lambda th: np.where(np.abs(th) < 1, 0.1, 1.0)
    n, dt = 200_000, 1e-2
    theta = rng.normal(0, s_init, n)
    for _ in range(4000):                         # t = 40 ≫ 1/min α = 10
        hit = rng.random(n) < alpha(theta) * dt
        theta[hit] = rng.normal(0, s_init, hit.sum())
    edges = np.linspace(-4, 4, 41)
    centers = 0.5 * (edges[1:] + edges[:-1])
    hist, _ = np.histogram(theta, bins=edges, density=True)
    p_init = np.exp(-centers**2 / (2 * s_init**2)) / np.sqrt(2 * np.pi * s_init**2)
    grid = np.linspace(-8, 8, 16001)
    pg = np.exp(-grid**2 / (2 * s_init**2)) / alpha(grid)
    z = pg.sum() * (grid[1] - grid[0])
    corrected = np.exp(-centers**2 / (2 * s_init**2)) / alpha(centers) / z
    data = long_form(centers, {
        "蒙特卡洛直方图": hist,
        "修正式 ∝ p_init/α": corrected,
        "原文式 p_init": p_init,
    }, x_name="theta", y_name="density")
    return line_chart(
        "stationary_density", data, x="theta", y="density", colour="series",
        x_label="θ", y_label="稳态密度", colour_label="",
    )


def changepoint_posterior():
    """Posterior change probability r for one weight, Gaussian likelihood (7.6)."""
    theta_A, mu, sigma2, s2, rho = 1.0, 0.0, 1.0, 0.01, 0.01

    def log_z(F, theta_B, m, v):
        # ∫ N(θ; m, v) exp(−F(θ−θ_B)²/2 + F(θ_A−θ_B)²/2) dθ, written around θ_A:
        # log-lik = g x − F x²/2 with x = θ−θ_A, g = F(θ_B−θ_A)
        g, mp = F * (theta_B - theta_A), m - theta_A
        return -0.5 * np.log1p(F * v) + (g * g * v + 2 * g * mp - F * mp * mp) / (2 * (1 + F * v))

    x = np.linspace(-3, 3, 121)
    series = {}
    for F in (0.0, 1.0, 10.0, 100.0):
        lf = np.log(rho) + log_z(F, x, mu, sigma2)
        ls = np.log1p(-rho) + log_z(F, x, theta_A, s2)
        series[f"F = {F:g}"] = 1 / (1 + np.exp(ls - lf))
    data = long_form(x, series, x_name="theta_B", y_name="r")
    return line_chart(
        "changepoint_posterior", data, x="theta_B", y="r", colour="series",
        x_label="新任务偏好的权重位置 θ_B（θ_A = 1，重置中心 μ = 0）",
        y_label="后验变点概率 r", colour_label="Fisher",
        hlines=[(rho, "先验 ρ")], y_limits=(0, 1),
    )


figs = [escape_times(), jump_variance(), state_dependent_stationary(), changepoint_posterior()]
for p in write_figures(PAPER, figs):
    print(p.relative_to(PROJECT))
