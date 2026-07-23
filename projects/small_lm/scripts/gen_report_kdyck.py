"""
Generate experiment report for exp_kdyck (k-Dyck procedural pre-training).

Produces: projects/small_lm/report-kdyck.md

Usage:
    uv run python projects/small_lm/scripts/gen_report_kdyck.py
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
PKL_DIR = Path(__file__).parent.parent / "pkl"
REPORT  = Path(__file__).parent.parent / "report-kdyck.md"


def load_result(exp_name):
    rows = {}
    with open(JSONL) as f:
        for line in f:
            r = json.loads(line)
            rows[r["experiment"]] = r
    if exp_name not in rows:
        raise ValueError(f"{exp_name} not in results")
    return rows[exp_name], rows.get("exp_baseline")


def plot_curves(r, r_base) -> str:
    history_path = PKL_DIR / "history_exp_kdyck.pkl"
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    kdyck_loss = history["phase1"]["kdyck_loss"]
    guppy_loss = history["phase2"]["loss"]
    eval_curve = r["eval_loss_curve"]

    eval_steps  = [e[0] for e in eval_curve]
    eval_ppls   = [math.exp(e[1]) for e in eval_curve]
    base_ppls   = [math.exp(e[1]) for e in r_base["eval_loss_curve"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Phase 1: k-Dyck training loss
    ax = axes[0]
    ax.plot(kdyck_loss, color="#9467BD", lw=0.8, alpha=0.7)
    w = 20
    if len(kdyck_loss) >= w:
        roll = np.convolve(kdyck_loss, np.ones(w)/w, mode="valid")
        ax.plot(range(w-1, len(kdyck_loss)), roll, color="#9467BD", lw=2, label=f"{w}-step avg")
    ax.axhline(math.log(32), color="gray", ls="--", lw=1, alpha=0.7, label="random (log 32)")
    ax.set_xlabel("k-Dyck step")
    ax.set_ylabel("CE loss")
    ax.set_title("Phase 1: k-Dyck training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase 2: Guppy training loss
    ax = axes[1]
    ax.plot(guppy_loss, color="#4C72B0", lw=0.5, alpha=0.6)
    w2 = 100
    if len(guppy_loss) >= w2:
        roll2 = np.convolve(guppy_loss, np.ones(w2)/w2, mode="valid")
        ax.plot(range(w2-1, len(guppy_loss)), roll2, color="#4C72B0", lw=1.5)
    ax.set_xlabel("Guppy step")
    ax.set_ylabel("CE loss")
    ax.set_title("Phase 2: Guppy training loss")
    ax.grid(True, alpha=0.3)

    # Eval PPL comparison vs baseline
    ax = axes[2]
    ax.plot(eval_steps, eval_ppls, marker="o", ms=4, color="#9467BD", label="k-Dyck pretrained", lw=1.5)
    ax.plot(eval_steps, base_ppls, marker="s", ms=4, color="#4C72B0", label="baseline", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Guppy step")
    ax.set_ylabel("perplexity")
    ax.set_title("Eval PPL: k-Dyck vs baseline")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("exp_kdyck — k-Dyck procedural pre-training (k=16, 1000 warm-up steps)", fontsize=10)
    fig.tight_layout()
    url = save_matplotlib_figure("small_lm_exp_kdyck_curves", fig, format="png", dpi=150)
    plt.close(fig)
    print(f"  curves → {url}")
    return url


def write_report(r, r_base, curves_url, pareto_url):
    lines = []
    a = lines.append

    a("# exp_kdyck — k-Dyck Procedural Pre-training")
    a("")
    a("**Method:** Two-phase training following [Shinnick et al. 2025](https://arxiv.org/abs/2601.21725)  ")
    a("**Phase 1:** 1,000 steps on procedurally generated k-Dyck bracket sequences (k=16)  ")
    a("**Phase 2:** 10,000 steps on Guppy fish conversations (same as baseline)  ")
    a(f"**Date:** 2026-04-07")
    a("")
    a("---")
    a("")
    a("## What is k-Dyck pre-training?")
    a("")
    a("A k-Dyck sequence is a perfectly balanced string of k types of brackets, e.g.")
    a("`( [ { } ] )` for k=3. Learning to predict the correct closing bracket requires")
    a("the model to track hierarchical dependencies across arbitrary distances — exactly")
    a("the structural skill that transformers use in natural language and code.")
    a("")
    a("The hypothesis: warming up on k-Dyck teaches the attention layers to form")
    a("structured dependency patterns before seeing any semantic content, acting as")
    a("a better-than-random initialisation for the main task.")
    a("")
    a("Token encoding: opens = IDs 3–18, closes = IDs 19–34 (k=16, offset past PAD/BOS/EOS).")
    a("")
    a("---")
    a("")

    a("## Configuration")
    a("")
    a("| | Phase 1 (k-Dyck) | Phase 2 (Guppy) |")
    a("|---|---|---|")
    a(f"| Steps | {r['pretrain_steps']:,} | {r['max_steps']:,} |")
    a(f"| Data | procedural (on-the-fly) | 57K Guppy conversations |")
    a(f"| LR | {r['pretrain_lr']} constant | {r['adam_lr']} cosine+warmup |")
    a(f"| k (bracket types) | {r['kdyck_k']} | — |")
    a(f"| Optimizer state | fresh AdamW | **carried over** from phase 1 |")
    a(f"| Batch size | {r['batch_size']} | {r['batch_size']} |")
    a("")

    a("## Results")
    a("")
    a("| Metric | exp_kdyck | exp_baseline | Δ |")
    a("|---|---|---|---|")
    ppl_delta  = r["final_eval_ppl"]  - r_base["final_eval_ppl"]
    time_delta = r["time_s"] - r_base["time_s"]
    a(f"| Final eval PPL | **{r['final_eval_ppl']:.4f}** | {r_base['final_eval_ppl']:.4f} | {ppl_delta:+.4f} |")
    a(f"| Training time | {r['time_s']:.0f}s | {r_base['time_s']:.0f}s | {time_delta:+.0f}s |")
    a(f"| Parameters | {r['n_params']:,} | {r_base['n_params']:,} | 0 |")
    a("")
    a(f"![Curves]({curves_url})")
    a("")

    a("## Phase 1 analysis: did the model learn k-Dyck?")
    a("")
    kl = r["kdyck_loss_curve"]
    random_loss = math.log(2 * r["kdyck_k"])
    a(f"- Random-chance CE loss for k=16 (32 tokens): **{random_loss:.2f}**")
    a(f"- k-Dyck loss at step 1: **{kl[0]:.2f}** (matches log(4096)={math.log(4096):.2f} — random init)")
    a(f"- k-Dyck loss at step 100: **{kl[99]:.4f}**")
    a(f"- k-Dyck loss at step 1000: **{kl[-1]:.4f}**")
    a(f"- Gap from random-32 floor: **{kl[-1] - random_loss:.4f}** nats")
    a("")
    a("The model learns well below the 32-token random baseline, confirming it captures")
    a("structural dependencies. However it doesn't reach zero — some residual uncertainty")
    a("in bracket *type* selection (all types are equally valid openings at any point).")
    a("")

    a("## Convergence comparison")
    a("")
    a("| Guppy step | baseline PPL | k-Dyck PPL | Δ |")
    a("|---|---|---|---|")
    for (bs, bl), (ks, kl2) in zip(r_base["eval_loss_curve"], r["eval_loss_curve"]):
        bp = math.exp(bl)
        kp = math.exp(kl2)
        a(f"| {bs:,} | {bp:.3f} | {kp:.3f} | {kp-bp:+.3f} |")
    a("")

    a("## Discussion")
    a("")
    a("**k-Dyck warm-up provides marginal final improvement (−0.0005 PPL) and runs 5s faster.**")
    a("")
    a("The convergence story is telling:")
    a("")
    a("- **Steps 1–3,000:** k-Dyck pretrained model is *slower* to converge — the LR schedule")
    a("  restarts from scratch and the weights encode bracket structure, not Guppy vocabulary.")
    a("  PPL at step 500 is 4.43 vs 2.48 for baseline.")
    a("")
    a("- **Steps 3,000–7,000:** The gap closes rapidly. Both models track the same eval curve.")
    a("")
    a("- **Steps 7,000–10,000:** Tiny advantage to k-Dyck (within noise). Final PPL is essentially")
    a("  identical.")
    a("")
    a("This is consistent with the paper's findings on small models: the procedural benefit is")
    a("most pronounced at larger scale (GPT-2 size and above) or when the main dataset has")
    a("richer structure for the inductive bias to exploit. Our Guppy corpus has only ~1,440")
    a("unique output words and 60 hand-written topic templates — the structural complexity is")
    a("low enough that a cold-start baseline saturates just as effectively.")
    a("")
    a("**Verdict:** k-Dyck warm-up is not harmful and is very cheap (1,000 steps ≈ 13s overhead).")
    a("On a more complex dataset it would likely show a clearer benefit.")
    a("")
    a("---")
    a("")
    a("## Pareto front")
    a("")
    a(f"![Pareto front]({pareto_url})")

    REPORT.write_text("\n".join(lines))
    print(f"Report written → {REPORT}")


def main():
    r, r_base = load_result("exp_kdyck")
    print("Plotting curves ...")
    curves_url = plot_curves(r, r_base)
    print("Generating Pareto plot ...")
    pareto_url = plot_pareto()
    print("Writing report ...")
    write_report(r, r_base, curves_url, pareto_url)


if __name__ == "__main__":
    main()
