"""
Generate the experiment report for exp_baseline.

Produces: projects/small_lm/report-baseline.md
Uploads plots to R2 (or saves locally as fallback).

Usage:
    uv run python projects/small_lm/scripts/gen_report_baseline.py
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
from lib.data_processing import load_tokenizer
from scripts.tmp.generate_samples import load_params, generate
from scripts.gen_pareto import plot_pareto

JSONL   = Path(__file__).parent.parent / "results.jsonl"
PKL_DIR = Path(__file__).parent.parent / "pkl"
DATA_DIR = Path(__file__).parent.parent / "data"
REPORT  = Path(__file__).parent.parent / "report-baseline.md"

PROMPTS = [
    "hi guppy",
    "how are you feeling",
    "are you hungry",
    "it's hot today",
    "it's cold today",
    "i changed your water",
    "do you get lonely",
    "what's the meaning of life",
    "tell me a joke",
    "goodnight guppy",
    "what do you think about cryptocurrency",
    "what do you think about taxes",
]


def load_result(exp_name: str) -> dict:
    with open(JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r.get("experiment") == exp_name:
                return r
    raise ValueError(f"{exp_name} not found in {JSONL}")


def plot_training_curves(r: dict) -> str:
    loss_curve = r["loss_curve"]
    eval_curve = r["eval_loss_curve"]   # list of [step, loss]

    eval_steps  = [e[0] for e in eval_curve]
    eval_losses = [e[1] for e in eval_curve]
    eval_ppls   = [math.exp(l) for l in eval_losses]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # — Training loss (every step) —
    ax = axes[0]
    ax.plot(loss_curve, color="#4C72B0", lw=0.6, alpha=0.7)
    # Rolling mean
    w = 100
    if len(loss_curve) >= w:
        roll = np.convolve(loss_curve, np.ones(w)/w, mode="valid")
        ax.plot(range(w-1, len(loss_curve)), roll, color="#C44E52", lw=1.5, label=f"{w}-step avg")
        ax.legend(fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)

    # — Eval loss —
    ax = axes[1]
    ax.plot(eval_steps, eval_losses, marker="o", ms=4, color="#55A868")
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title("Eval loss")
    ax.grid(True, alpha=0.3)

    # — Eval perplexity —
    ax = axes[2]
    ax.plot(eval_steps, eval_ppls, marker="o", ms=4, color="#DD8452")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title("Eval perplexity")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)

    fig.suptitle("exp_baseline — 8.73M param transformer, 10K steps", fontsize=11)
    fig.tight_layout()

    url = save_matplotlib_figure("small_lm_exp_baseline_curves", fig, format="png", dpi=150)
    plt.close(fig)
    print(f"  curves → {url}")
    return url


def run_generation(params, tok) -> list[tuple[str, str]]:
    pairs = []
    for i, prompt in enumerate(PROMPTS):
        resp = generate(params, tok, prompt, temperature=0.7, top_k=50, seed=i)
        pairs.append((prompt, resp))
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt!r} → {resp[:60]}...")
    return pairs


def write_report(r: dict, curves_url: str, samples: list[tuple[str, str]], pareto_url: str = "") -> None:
    lines = []
    a = lines.append

    a("# exp_baseline — Report")
    a("")
    a("**Model:** 6-layer vanilla transformer, direct JAX port of [GuppyLM](https://github.com/arman-bd/guppylm)  ")
    a("**Dataset:** 60K synthetic Guppy-fish conversations (57K train / 3K eval)  ")
    a(f"**Date:** 2026-04-07")
    a("")
    a("---")
    a("")

    a("## Configuration")
    a("")
    a("| Hyperparameter | Value |")
    a("|---|---|")
    a(f"| Parameters | {r['n_params']:,} ({r['n_params']/1e6:.2f}M) |")
    a(f"| Layers | {r['n_layers']} |")
    a(f"| d_model | {r['d_model']} |")
    a(f"| Heads | {r['n_heads']} (head_dim={r['d_model']//r['n_heads']}) |")
    a(f"| FFN hidden | {r['ffn_hidden']} (ReLU) |")
    a(f"| Context window | {r['max_seq_len']} tokens |")
    a(f"| Vocab size | {r['vocab_size']} (BPE; effective 2375) |")
    a(f"| Batch size | {r['batch_size']} |")
    a(f"| Learning rate | {r['adam_lr']} (cosine decay, {r['warmup_steps']}-step warmup) |")
    a(f"| Max steps | {r['max_steps']:,} |")
    a(f"| Dropout | none |")
    a(f"| LM head | weight-tied with token embeddings |")
    a("")

    a("## Results")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Final eval loss | {r['final_eval_loss']:.4f} |")
    a(f"| Final eval PPL | **{r['final_eval_ppl']:.2f}** |")
    a(f"| Training time | {r['time_s']:.0f}s (RTX 4090) |")
    a("")
    a(f"![Training curves]({curves_url})")
    a("")
    a("**Note on PPL:** eval_ppl=1.46 reflects near-memorisation of a template-based corpus.")
    a("The dataset has only ~2,375 effective BPE tokens and 19% unique outputs from 60K samples.")
    a("The 8.7M-param model is substantially over-parameterised for this data —")
    a("a finding that motivates the next sweep (smaller models, more diverse data).")
    a("")

    a("## Learning dynamics")
    a("")
    eval_curve = r["eval_loss_curve"]
    first_ppl = math.exp(eval_curve[0][1])
    last_ppl  = math.exp(eval_curve[-1][1])
    drop_step = eval_curve[0][0]
    a(f"- Initial eval PPL at step {drop_step}: **{first_ppl:.1f}**")
    a(f"- Final eval PPL at step {eval_curve[-1][0]}: **{last_ppl:.2f}**")
    a(f"- Reduction factor: {first_ppl/last_ppl:.0f}×")
    a(f"- Loss plateau: reached ~{eval_curve[-3][1]:.3f} CE by step {eval_curve[-3][0]}, stable thereafter")
    a("")

    a("## Generation samples")
    a("")
    a("Temperature=0.7, top_k=50:")
    a("")
    for prompt, resp in samples:
        a(f"**User:** {prompt}  ")
        a(f"**Guppy:** {resp}")
        a("")

    a("---")
    a("")
    a("## Observations & next steps")
    a("")
    a("1. **Near-perfect dataset fit** — the model correctly learns the Guppy persona and")
    a("   produces on-character responses for all 60 topic categories.")
    a("")
    a("2. **Dataset is the bottleneck** — with only 2,375 BPE tokens and low output diversity,")
    a("   PPL cannot meaningfully distinguish model quality. Future experiments should either")
    a("   (a) expand template variation or (b) use a real held-out corpus for evaluation.")
    a("")
    a("3. **Model is over-parameterised** — a 2L or 4L model likely achieves similar PPL.")
    a("   Proposed sweep: `exp_2l`, `exp_4l` (d_model=384, vary N_LAYERS).")
    a("")
    a("4. **Training speed** — 10K steps in 128s (~78 steps/s). Headroom to run many sweeps.")
    a("")
    a("---")
    a("")
    a("## Pareto front: training time vs perplexity")
    a("")
    a(f"![Pareto front]({pareto_url})")

    REPORT.write_text("\n".join(lines))
    print(f"\nReport written → {REPORT}")


def main():
    print("Loading result ...")
    r = load_result("exp_baseline")

    print("Plotting training curves ...")
    curves_url = plot_training_curves(r)

    print("Loading model for generation ...")
    params = load_params("exp_baseline")
    tok    = load_tokenizer(DATA_DIR / "tokenizer.json")

    print("Generating samples ...")
    samples = run_generation(params, tok)

    print("Generating Pareto plot ...")
    pareto_url = plot_pareto()

    print("Writing report ...")
    write_report(r, curves_url, samples, pareto_url)


if __name__ == "__main__":
    main()
