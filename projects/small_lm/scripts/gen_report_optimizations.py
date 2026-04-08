"""
Generate the combined optimizations report comparing exp_baseline, exp_muon,
exp_rope, and exp_no_tie.

Produces: projects/small_lm/report-optimizations.md
Uploads plots to R2 (or local fallback).

Usage:
    uv run python projects/small_lm/scripts/gen_report_optimizations.py
"""

import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_lib.media import save_matplotlib_figure
from scripts.gen_pareto import plot_pareto

JSONL   = Path(__file__).parent.parent / "results.jsonl"
REPORT  = Path(__file__).parent.parent / "report-optimizations.md"

EXPERIMENTS = ["exp_baseline", "exp_muon", "exp_rope", "exp_no_tie"]

COLORS = {
    "exp_baseline": "#4C72B0",
    "exp_muon":     "#DD8452",
    "exp_rope":     "#55A868",
    "exp_no_tie":   "#C44E52",
}

LABELS = {
    "exp_baseline": "baseline (AdamW, learned pos, weight-tied)",
    "exp_muon":     "muon (Muon optimizer)",
    "exp_rope":     "rope (RoPE, no learned pos embed)",
    "exp_no_tie":   "no_tie (separate LM head)",
}

DESCRIPTIONS = {
    "exp_baseline": "Direct GuppyLM port. AdamW optimizer, learned absolute position "
                    "embeddings, weight-tied LM head. 8.73M parameters.",
    "exp_muon":     "Muon optimizer: Newton-Schulz orthogonalisation applied to all "
                    "2D weight matrix gradients before the update step; AdamW handles "
                    "embeddings, biases, and LayerNorm. Same architecture as baseline.",
    "exp_rope":     "RoPE (Rotary Position Embedding) replaces learned positional "
                    "embeddings. Q and K vectors are rotated by sinusoidal frequencies "
                    "before the attention dot-product, encoding relative positions. "
                    "Saves 49K params (no pos_embed).",
    "exp_no_tie":   "Separate LM head — the output projection `lm_head` is independent "
                    "of `token_embed`, giving the model ~1.57M extra parameters (~10.3M "
                    "total). Hypothesis: untied output space allows better calibration.",
}


def load_results() -> dict[str, dict]:
    rows = {}
    if not JSONL.exists():
        return rows
    with open(JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                k = r.get("experiment")
                if k:
                    rows[k] = r   # last entry wins
            except json.JSONDecodeError:
                pass
    return rows


def plot_comparison(results: dict) -> str:
    """Side-by-side eval loss and eval PPL curves for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for exp_name in EXPERIMENTS:
        if exp_name not in results:
            continue
        r = results[exp_name]
        eval_curve = r.get("eval_loss_curve", [])
        if not eval_curve:
            continue
        steps  = [e[0] for e in eval_curve]
        losses = [e[1] for e in eval_curve]
        ppls   = [math.exp(l) for l in losses]
        color  = COLORS.get(exp_name, "#888888")
        label  = LABELS.get(exp_name, exp_name)

        axes[0].plot(steps, losses, marker="o", ms=4, color=color, label=label, lw=1.5)
        axes[1].plot(steps, ppls,   marker="o", ms=4, color=color, label=label, lw=1.5)

    for ax, ylabel, title in [
        (axes[0], "CE loss",      "Eval loss"),
        (axes[1], "perplexity",   "Eval perplexity"),
    ]:
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right")

    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.suptitle("small_lm — optimizer/arch optimizations: eval comparison", fontsize=11)
    fig.tight_layout()

    url = save_matplotlib_figure("small_lm_optim_comparison", fig, format="png", dpi=150)
    plt.close(fig)
    print(f"  comparison → {url}")
    return url


def write_report(results: dict, comparison_url: str, pareto_url: str) -> None:
    lines = []
    a = lines.append

    present = [e for e in EXPERIMENTS if e in results]
    missing = [e for e in EXPERIMENTS if e not in results]

    a("# Optimization experiments — Report")
    a("")
    a("Comparing three architectural/optimizer changes against the baseline:")
    a("**Muon optimizer**, **RoPE positional encoding**, and **separate (untied) LM head**.")
    a("")
    a("**Dataset:** 60K synthetic Guppy-fish conversations (57K train / 3K eval)  ")
    a("**Training:** 10K steps, cosine LR + warmup, batch_size=32, RTX 4090")
    a("")

    if missing:
        a(f"> **Note:** results not yet available for: {', '.join(missing)}")
        a("")

    a("---")
    a("")

    # Summary table
    a("## Results summary")
    a("")
    a("| Experiment | Params | PPL | Time (s) | vs baseline |")
    a("|---|---|---|---|---|")
    baseline_ppl  = results.get("exp_baseline", {}).get("final_eval_ppl")
    baseline_time = results.get("exp_baseline", {}).get("time_s")
    for exp_name in EXPERIMENTS:
        if exp_name not in results:
            a(f"| {exp_name} | — | — | — | — |")
            continue
        r = results[exp_name]
        ppl    = r["final_eval_ppl"]
        time_s = r["time_s"]
        npar   = r["n_params"]
        if baseline_ppl and exp_name != "exp_baseline":
            delta_ppl  = ppl - baseline_ppl
            delta_time = time_s - baseline_time
            sign_p = "+" if delta_ppl >= 0 else ""
            sign_t = "+" if delta_time >= 0 else ""
            vs = f"PPL {sign_p}{delta_ppl:.3f}, time {sign_t}{delta_time:.0f}s"
        else:
            vs = "—"
        a(f"| {exp_name} | {npar/1e6:.2f}M | **{ppl:.4f}** | {time_s:.0f} | {vs} |")
    a("")
    a(f"![Comparison curves]({comparison_url})")
    a("")
    a("---")
    a("")

    # Per-experiment sections
    a("## Experiment details")
    a("")
    for exp_name in EXPERIMENTS:
        a(f"### {exp_name}")
        a("")
        a(DESCRIPTIONS.get(exp_name, ""))
        a("")
        if exp_name in results:
            r = results[exp_name]
            a(f"- **Parameters:** {r['n_params']:,} ({r['n_params']/1e6:.2f}M)")
            a(f"- **Final eval PPL:** {r['final_eval_ppl']:.4f}")
            a(f"- **Final eval loss:** {r['final_eval_loss']:.4f}")
            a(f"- **Training time:** {r['time_s']:.0f}s")
            eval_curve = r.get("eval_loss_curve", [])
            if eval_curve:
                first_ppl = math.exp(eval_curve[0][1])
                last_ppl  = math.exp(eval_curve[-1][1])
                a(f"- **PPL trajectory:** {first_ppl:.1f} → {last_ppl:.4f} "
                  f"(×{first_ppl/last_ppl:.0f} reduction)")
        else:
            a("*(not yet run)*")
        a("")

    a("---")
    a("")

    # Analysis
    a("## Analysis")
    a("")
    if len(present) >= 2:
        ranked = sorted(
            [(e, results[e]["final_eval_ppl"]) for e in present],
            key=lambda x: x[1]
        )
        best_exp, best_ppl = ranked[0]
        a(f"**Best PPL:** {best_exp} ({best_ppl:.4f})")
        a("")

        # Muon commentary
        if "exp_muon" in results and "exp_baseline" in results:
            m = results["exp_muon"]
            b = results["exp_baseline"]
            delta_ppl  = m["final_eval_ppl"] - b["final_eval_ppl"]
            delta_time = m["time_s"] - b["time_s"]
            verdict = "improved" if delta_ppl < 0 else "did not improve"
            a(f"**Muon:** {verdict} PPL by {abs(delta_ppl):.4f} vs baseline "
              f"({'faster' if delta_time < 0 else 'slower'} by {abs(delta_time):.0f}s). "
              f"Newton-Schulz orthogonalisation constrains weight updates to the "
              f"Stiefel manifold, effectively acting as a second-order approximation "
              f"for matrix parameters.")
            a("")

        # RoPE commentary
        if "exp_rope" in results and "exp_baseline" in results:
            r_rope = results["exp_rope"]
            b = results["exp_baseline"]
            delta_ppl  = r_rope["final_eval_ppl"] - b["final_eval_ppl"]
            verdict = "improved" if delta_ppl < 0 else "did not improve"
            a(f"**RoPE:** {verdict} PPL by {abs(delta_ppl):.4f} vs baseline. "
              f"At context=128 and vocab=2375 tokens, absolute position embeddings "
              f"are unlikely to be a bottleneck — both approaches encode similar "
              f"positional information for short sequences.")
            a("")

        # No-tie commentary
        if "exp_no_tie" in results and "exp_baseline" in results:
            nt = results["exp_no_tie"]
            b  = results["exp_baseline"]
            delta_ppl  = nt["final_eval_ppl"] - b["final_eval_ppl"]
            delta_par  = nt["n_params"] - b["n_params"]
            verdict = "improved" if delta_ppl < 0 else "did not improve"
            a(f"**No-tie:** {verdict} PPL by {abs(delta_ppl):.4f} vs baseline "
              f"(+{delta_par/1e6:.2f}M parameters). Decoupling input embeddings "
              f"from the output projection gives the model more freedom but also "
              f"increases the number of parameters to optimise.")
            a("")

    a("---")
    a("")
    a("## Pareto front: training time vs perplexity")
    a("")
    a(f"![Pareto front]({pareto_url})")
    a("")
    a("★ = Pareto-optimal experiments (not dominated in both dimensions).")

    REPORT.write_text("\n".join(lines))
    print(f"\nReport written → {REPORT}")


def main():
    print("Loading results ...")
    results = load_results()
    missing = [e for e in EXPERIMENTS if e not in results]
    if missing:
        print(f"  WARNING: no results for {missing} — report will be partial")

    print("Plotting comparison curves ...")
    comparison_url = plot_comparison(results)

    print("Generating Pareto plot ...")
    pareto_url = plot_pareto()

    print("Writing report ...")
    write_report(results, comparison_url, pareto_url)


if __name__ == "__main__":
    main()
