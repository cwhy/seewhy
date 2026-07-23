"""
Generate the Phase B report: ES experiments + the optimizer-strength findings.

Saves HTML locally to projects/rosa/report_phaseB.html and uploads to R2.

Usage: uv run python projects/rosa/scripts/gen_report_phaseB.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJ_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))

from shared_lib.media import save_matplotlib_figure       # noqa: E402
from shared_lib.html import save_html_file                 # noqa: E402

LOCAL_HTML = PROJ_DIR / "report_phaseB.html"
PHASE_A_URL = "https://media.tanh.xyz/seewhy/26-05-08/rosa_report_phaseA.html"


# ── Load results ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_exp(rows):
    return {r["experiment"]: r for r in rows}


R = by_exp(load_jsonl(PROJ_DIR / "results.jsonl"))

# Phase A no-ROSA baselines from tmp jsons
TMP = PROJ_DIR / "scripts" / "tmp"
def load_tmp(name):
    with open(TMP / name) as f:
        return json.load(f)

dense_fmnist = load_tmp("dense_baseline_result.fashion_mnist.json")
vq_fmnist    = load_tmp("vq_only_baseline_result.fashion_mnist.json")


# ── Plot 1: All ES results vs gradient baselines ──────────────────────────────

def plot_es_summary():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars = [
        # name, test_acc, color, group
        ("dense (gradient)\n60K",                dense_fmnist["final_test_acc"],         "#9ca3af", "gradient"),
        ("exp1 (grad + ROSA stopgrad)\n60K",     R["exp1_fmnist"]["test_acc"],          "#3b82f6", "gradient"),
        ("exp6 (grad + ROSA no-stopgrad)\n60K",  R["exp6_fmnist"]["test_acc"],          "#3b82f6", "gradient"),
        ("exp10 (ES + ROSA)\nn_pop=32 r=1, 10K", R["exp10_fmnist"]["test_acc"],         "#dc2626", "es"),
        ("exp11 (ES no ROSA)\nn_pop=32 r=1, 10K",R["exp11_fmnist"]["test_acc"],         "#fca5a5", "es"),
        ("exp10b (ES + ROSA)\nn_pop=32 r=1, 60K",R["exp10b_fmnist"]["test_acc"],        "#dc2626", "es"),
        ("exp11b (ES no ROSA)\nn_pop=32 r=1, 60K",R["exp11b_fmnist"]["test_acc"],       "#fca5a5", "es"),
        ("exp10c (ES + ROSA)\nn_pop=32 r=4, 60K",R["exp10c_fmnist"]["test_acc"],        "#dc2626", "es"),
        ("exp10d (ES + ROSA)\nn_pop=128 r=4, 60K",R["exp10d_fmnist"]["test_acc"],       "#dc2626", "es"),
        ("exp11d (ES no ROSA)\nn_pop=128 r=4, 60K",R["exp11d_fmnist"]["test_acc"],      "#fca5a5", "es"),
    ]
    xs = np.arange(len(bars))
    accs = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    ax.bar(xs, accs, color=cols, edgecolor="black", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.012, f"{a*100:.1f}%", ha="center", fontsize=9, family="monospace")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.8, family="monospace", rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy", family="monospace")
    ax.set_title("Fashion-MNIST: all experiments. Grey = baseline · Blue = gradient · Red = ES with ROSA · Pink = ES no ROSA",
                 family="monospace", fontsize=11)
    ax.axhline(y=dense_fmnist["final_test_acc"], color="#666", linestyle="--",
               linewidth=0.5, alpha=0.6)
    ax.axhline(y=0.10, color="#999", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


# ── Plot 2: ROSA contribution vs optimizer strength ───────────────────────────

def plot_rosa_contribution():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    # ROSA contribution = (with ROSA) - (no ROSA)
    rows = [
        # x_label, with_rosa, no_rosa, optimizer_label
        ("ES\nn_pop=32, r=1, 10K",    R["exp10_fmnist"]["test_acc"],  R["exp11_fmnist"]["test_acc"]),
        ("ES\nn_pop=32, r=1, 60K",    R["exp10b_fmnist"]["test_acc"], R["exp11b_fmnist"]["test_acc"]),
        ("ES\nn_pop=128, r=4, 60K",   R["exp10d_fmnist"]["test_acc"], R["exp11d_fmnist"]["test_acc"]),
        # Gradient: ROSA contribution is C - A from lesion
        ("Gradient\n+ no stopgrad\nfrom lesion",     R["exp6_fmnist"]["test_acc"],   R["exp6_fmnist"]["test_acc"] - 0.0010),
        ("Gradient\n+ stopgrad\nfrom lesion",        R["exp1_fmnist"]["test_acc"],   R["exp1_fmnist"]["test_acc"] - 0.0001),
    ]
    xs = np.arange(len(rows))
    diffs = [r[1] - r[2] for r in rows]
    cols = ["#16a34a" if d > 0 else "#dc2626" for d in diffs]
    bars = ax.bar(xs, [d * 100 for d in diffs], color=cols, edgecolor="black", linewidth=0.5)
    for i, d in enumerate(diffs):
        ax.text(i, d * 100 + (0.15 if d > 0 else -0.4), f"{d*100:+.2f}pp",
                ha="center", fontsize=9, family="monospace",
                color="#15803d" if d > 0 else "#991b1b")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5, family="monospace")
    ax.set_ylabel("ROSA contribution (pp)", family="monospace")
    ax.set_title("ROSA's contribution flips sign with optimizer strength.\n"
                 "Weak optimizer + medium data → ROSA helps. Strong optimizer → ROSA hurts.",
                 family="monospace", fontsize=10.5)
    ax.axhline(0, color="#444", linewidth=1.0)
    ax.set_ylim(-6, 6)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


# ── Plot 3: ES learning curves (test acc per epoch) ───────────────────────────

def plot_es_curves():
    fig, ax = plt.subplots(figsize=(10, 5))

    cfg = [
        ("exp10  (ROSA, n_pop=32, r=1, 10K)",   "exp10_fmnist",  "#dc2626", "-"),
        ("exp11  (no ROSA, n_pop=32, r=1, 10K)","exp11_fmnist",  "#dc2626", ":"),
        ("exp10b (ROSA, n_pop=32, r=1, 60K)",   "exp10b_fmnist", "#7c3aed", "-"),
        ("exp11b (no ROSA, n_pop=32, r=1, 60K)","exp11b_fmnist", "#7c3aed", ":"),
        ("exp10c (ROSA, n_pop=32, r=4, 60K)",   "exp10c_fmnist", "#0ea5e9", "-"),
        ("exp10d (ROSA, n_pop=128, r=4, 60K)",  "exp10d_fmnist", "#16a34a", "-"),
        ("exp11d (no ROSA, n_pop=128, r=4, 60K)","exp11d_fmnist","#16a34a", ":"),
    ]
    for label, key, color, ls in cfg:
        h = R[key]["history"]
        # history["test_acc"] is a list of (epoch, acc) pairs in our ES experiments
        # but in the gradient experiments it's just a list of acc per epoch. Check format.
        ta = h["test_acc"]
        if ta and isinstance(ta[0], (list, tuple)):
            ep = [t[0] for t in ta]
            v  = [t[1] for t in ta]
        else:
            ep = list(range(len(ta)))
            v = ta
        ax.plot(ep, v, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.set_xlabel("epoch", family="monospace")
    ax.set_ylabel("test accuracy", family="monospace")
    ax.set_title("ES learning curves on Fashion-MNIST. Solid = with ROSA, dotted = no ROSA.\n"
                 "Same colors = same data + ES config; solid vs dotted shows ROSA's effect.",
                 family="monospace", fontsize=11)
    ax.set_ylim(0.3, 0.8)
    ax.legend(loc="lower right", prop={"family": "monospace", "size": 8.5},
              ncol=1, framealpha=0.92)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


# ── Plot 4: train vs test gap across ES configs ───────────────────────────────

def plot_es_gen_gap():
    fig, ax = plt.subplots(figsize=(9, 4.5))

    rows = [
        ("exp10\nn=10K, r=1, pop=32",   R["exp10_fmnist"]),
        ("exp11\nn=10K, r=1, pop=32",   R["exp11_fmnist"]),
        ("exp10b\nn=60K, r=1, pop=32",  R["exp10b_fmnist"]),
        ("exp11b\nn=60K, r=1, pop=32",  R["exp11b_fmnist"]),
        ("exp10c\nn=60K, r=4, pop=32",  R["exp10c_fmnist"]),
        ("exp10d\nn=60K, r=4, pop=128", R["exp10d_fmnist"]),
        ("exp11d\nn=60K, r=4, pop=128", R["exp11d_fmnist"]),
        ("exp1\n(gradient, 60K)",       R["exp1_fmnist"]),
    ]
    xs = np.arange(len(rows))
    width = 0.36
    tr = [r[1]["train_acc"] for r in rows]
    te = [r[1]["test_acc"]  for r in rows]
    ax.bar(xs - width/2, tr, width, label="train", color="#9ca3af",
           edgecolor="black", linewidth=0.5)
    ax.bar(xs + width/2, te, width, label="test", color="#3b82f6",
           edgecolor="black", linewidth=0.5)
    for i, (a, b) in enumerate(zip(tr, te)):
        ax.text(i - width/2, a + 0.008, f"{a*100:.1f}", ha="center", fontsize=8, family="monospace")
        ax.text(i + width/2, b + 0.008, f"{b*100:.1f}", ha="center", fontsize=8, family="monospace")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8, family="monospace")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", family="monospace")
    ax.set_title("Train vs test across ES configs. ES never overfits (gap < 5pp); gradient overfits to ~10pp gap.",
                 family="monospace", fontsize=10.5)
    ax.legend(loc="lower right", prop={"family": "monospace", "size": 9})
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig


# ── Build the HTML ────────────────────────────────────────────────────────────

def fmt_pct(x):
    return f"{x*100:.2f}%"


def build_html(plot_urls):
    head = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>ROSA on MNIST — Phase B (ES findings)</title>\n"
        "<script src='https://cdn.tailwindcss.com'></script>\n"
        "<style>\n"
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }\n"
        "  pre, code, .mono { font-family: 'SF Mono', 'Menlo', monospace; }\n"
        "  table { border-collapse: collapse; }\n"
        "  th, td { padding: 0.4rem 0.7rem; border: 1px solid #e5e7eb; font-size: 0.92rem; }\n"
        "  th { background: #f3f4f6; text-align: left; }\n"
        "  .num { text-align: right; font-family: 'SF Mono', 'Menlo', monospace; }\n"
        "  .hi { background: #fef3c7; font-weight: 600; }\n"
        "  .neg { color: #b91c1c; }\n"
        "  .pos { color: #15803d; }\n"
        "  blockquote { border-left: 4px solid #d1d5db; padding: 0.5rem 1rem; color: #4b5563; "
        "margin: 1rem 0; background: #f9fafb; }\n"
        "</style>\n"
        "</head>\n"
    )

    e10  = R["exp10_fmnist"]
    e11  = R["exp11_fmnist"]
    e10b = R["exp10b_fmnist"]
    e11b = R["exp11b_fmnist"]
    e10c = R["exp10c_fmnist"]
    e10d = R["exp10d_fmnist"]
    e11d = R["exp11d_fmnist"]
    e1f  = R["exp1_fmnist"]
    e6f  = R["exp6_fmnist"]

    rosa_at_10K_pop32  = (e10["test_acc"] - e11["test_acc"]) * 100
    rosa_at_60K_pop32  = (e10b["test_acc"] - e11b["test_acc"]) * 100
    rosa_at_60K_pop128 = (e10d["test_acc"] - e11d["test_acc"]) * 100

    body = f"""
<body class='bg-white p-6 max-w-4xl mx-auto'>

  <h1 class='text-3xl font-mono mb-2'>ROSA on MNIST — Phase B</h1>
  <h2 class='text-xl font-mono text-gray-500 mb-2'>Gradient-free training, and the optimizer-strength signature</h2>
  <p class='text-sm text-gray-600 mb-6'>
    Continuation of the <a class='text-blue-600 underline' href='{PHASE_A_URL}'>Phase A report</a>,
    which covered the gradient experiments + the decoupled (ROSA-as-classifier) overfitting finding.
  </p>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>TL;DR — what Phase B added</h3>
    <ul class='list-disc ml-6 space-y-2'>
      <li>Trained the joint pipeline with <strong>EGGROLL</strong> antithetic ES instead of gradient. This removes the stop_grad / non-differentiability problem entirely: ES doesn't need gradients, so ROSA's discrete predictions and the codebook's argmin lookup are no longer barriers to the optimizer.</li>
      <li>Ran a clean ablation matrix over data size (10K vs 60K Fashion-MNIST), perturbation rank (1 vs 4), and population (32 vs 128) — with and without ROSA — total 7 ES experiments + the existing gradient runs.</li>
      <li><strong>Headline result.</strong> ROSA's contribution <em>flips sign</em> with optimizer strength:
        <ul class='list-disc ml-6 mt-1'>
          <li>Weakest ES (10K, n_pop=32, rank=1): ROSA <span class='neg'>hurts by −4.15pp</span>.</li>
          <li>Medium ES (60K, n_pop=32, rank=1): ROSA <span class='pos'>helps by +2.79pp</span>.</li>
          <li>Strong ES (60K, n_pop=128, rank=4): ROSA <span class='neg'>hurts by −4.31pp</span>.</li>
          <li>Gradient (the strongest optimizer): ROSA's contribution is ~0pp (lesion confirms it's ignored).</li>
        </ul>
      </li>
      <li>ES <strong>never overfits</strong> — train-test gap stays under 5pp across all 7 ES runs, vs ~10pp for gradient on the same architecture. ES can't memorize specific samples, only find broad parameter directions.</li>
      <li>The honest reading: ROSA is effectively a weak inductive bias. When the optimizer can't fully shape the codebook, ROSA's class-correlated guesses help fill the gap. Once the optimizer has enough data + samples per step, the codebook can be shaped precisely for classification, and ROSA's discrete predictions become redundant noise.</li>
    </ul>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>1 · Why ES?</h3>
    <p class='mb-3'>
      Phase A established that <strong>under gradient training, ROSA is structurally ignored</strong>:
    </p>
    <ul class='list-disc ml-6 mb-3'>
      <li>With <span class='mono'>stop_grad</span> on ROSA's predicted-token embedding (exp1), the output module learns to zero out the ROSA input channel — confirmed by the lesion test (A − C ≈ 0).</li>
      <li>Removing the <span class='mono'>stop_grad</span> (exp6) lets gradient flow through the codebook entries ROSA picks, but the gain is only +0.10pp.</li>
    </ul>
    <p class='mb-3'>
      The user proposed an architectural pivot: <em>force ROSA to be the model</em> via decoupled training (Phase A's section 5). That hit a different wall — the SAM is a perfect memorizer (100% train) but reconstruction-optimal tokens aren't class-discriminative enough for generalization (54% test).
    </p>
    <p>
      Phase B's idea: <strong>ES removes the structural barrier</strong>. There's no <span class='mono'>stop_grad</span> concept in gradient-free optimization. If ROSA's contribution can be learned at all under some optimizer, ES should reveal it. Plus ES naturally handles the non-differentiabilities (DiVeQ argmin, ROSA's discrete argmax) — they're inside the forward pass, the optimizer only sees the scalar loss.
    </p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>2 · ES setup</h3>
    <p class='mb-3'>
      Architecture: identical to exp1 (input MLP + DiVeQ codebook + 11-class dense-fire output). Optimizer swapped to <strong>antithetic ES with rank-k perturbations</strong> (EGGROLL pattern from <span class='mono'>projects/es/experiments3.py</span>).
    </p>
    <p class='mb-3'>Per ES step on each batch:</p>
    <ol class='list-decimal ml-6 mb-3'>
      <li>Sample <span class='mono'>n_pop</span> rank-k perturbations <span class='mono'>ε_k</span>.</li>
      <li>For each k, evaluate loss at <span class='mono'>params + ε_k</span> and <span class='mono'>params - ε_k</span> (antithetic), with ROSA queries on the per-pop tokens.</li>
      <li>ES gradient estimate: <span class='mono'>Σ_k fitness_k · ε_k / (n_pop · σ²)</span>.</li>
      <li>Adam update on the ES gradient.</li>
      <li>Commit batch's tokens to ROSA (one set, using updated params).</li>
    </ol>
    <p class='mb-3'>
      Hyperparameters held fixed across all ES runs: V=1024, D=64, K_sample=16, K_label=4, BATCH=64, 30 epochs, σ=0.001, Adam lr cosine 1e-3 → 1e-4, seed 42.
      Varied: data size (10K, 60K), perturbation rank (1, 4), and population (32, 128).
    </p>

    <figure class='my-6'>
      <img src='{plot_urls['summary']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        All Fashion-MNIST results. Grey: no-ROSA baselines · Blue: gradient + ROSA · Red: ES + ROSA · Pink: ES, no ROSA.
        The 89.5% dashed line is the dense-baseline ceiling.
      </figcaption>
    </figure>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>3 · The 7 ES experiments</h3>

    <table class='w-full mb-4'>
      <thead><tr><th>Run</th><th>Data</th><th>ROSA</th><th class='num'>rank</th><th class='num'>n_pop</th><th class='num'>Test acc</th><th class='num'>Train acc</th><th class='num'>Time</th></tr></thead>
      <tbody>
        <tr><td>exp10</td><td class='num'>10K</td><td>✓</td><td class='num'>1</td><td class='num'>32</td><td class='num'>{fmt_pct(e10['test_acc'])}</td><td class='num'>{fmt_pct(e10['train_acc'])}</td><td class='num'>{e10['time_s']/60:.1f} min</td></tr>
        <tr><td>exp11</td><td class='num'>10K</td><td>—</td><td class='num'>1</td><td class='num'>32</td><td class='num hi'>{fmt_pct(e11['test_acc'])}</td><td class='num'>{fmt_pct(e11['train_acc'])}</td><td class='num'>{e11['time_s']/60:.1f} min</td></tr>
        <tr><td>exp10b</td><td class='num'>60K</td><td>✓</td><td class='num'>1</td><td class='num'>32</td><td class='num hi'>{fmt_pct(e10b['test_acc'])}</td><td class='num'>{fmt_pct(e10b['train_acc'])}</td><td class='num'>{e10b['time_s']/60:.1f} min</td></tr>
        <tr><td>exp11b</td><td class='num'>60K</td><td>—</td><td class='num'>1</td><td class='num'>32</td><td class='num'>{fmt_pct(e11b['test_acc'])}</td><td class='num'>{fmt_pct(e11b['train_acc'])}</td><td class='num'>{e11b['time_s']/60:.1f} min</td></tr>
        <tr><td>exp10c</td><td class='num'>60K</td><td>✓</td><td class='num'>4</td><td class='num'>32</td><td class='num'>{fmt_pct(e10c['test_acc'])}</td><td class='num'>{fmt_pct(e10c['train_acc'])}</td><td class='num'>{e10c['time_s']/60:.1f} min</td></tr>
        <tr><td>exp10d</td><td class='num'>60K</td><td>✓</td><td class='num'>4</td><td class='num'>128</td><td class='num'>{fmt_pct(e10d['test_acc'])}</td><td class='num'>{fmt_pct(e10d['train_acc'])}</td><td class='num'>{e10d['time_s']/60:.0f} min</td></tr>
        <tr><td>exp11d</td><td class='num'>60K</td><td>—</td><td class='num'>4</td><td class='num'>128</td><td class='num hi'>{fmt_pct(e11d['test_acc'])}</td><td class='num'>{fmt_pct(e11d['train_acc'])}</td><td class='num'>{e11d['time_s']/60:.0f} min</td></tr>
      </tbody>
    </table>

    <p class='text-sm text-gray-700'>
      The yellow cells mark the better of each (with-ROSA, no-ROSA) pair at matched optimizer strength. At weakest and strongest optimizer strengths, no-ROSA wins; at intermediate strength, with-ROSA wins.
    </p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>4 · The optimizer-strength flip</h3>

    <figure class='my-6'>
      <img src='{plot_urls['rosa_contrib']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        ROSA's measured contribution as (with-ROSA accuracy) − (no-ROSA accuracy), across optimizer strengths.
        The U-shape — helping at intermediate strength, hurting at both extremes — is the signature finding of Phase B.
      </figcaption>
    </figure>

    <p class='mb-3'>The pattern across optimizer strengths:</p>

    <table class='w-full mb-4'>
      <thead><tr><th>Optimizer config</th><th>Pair</th><th class='num'>with ROSA</th><th class='num'>no ROSA</th><th class='num'>ROSA contribution</th></tr></thead>
      <tbody>
        <tr><td>ES, weak (n_pop=32, r=1) + small data (10K)</td><td>exp10 vs exp11</td><td class='num'>{fmt_pct(e10['test_acc'])}</td><td class='num'>{fmt_pct(e11['test_acc'])}</td><td class='num neg'>{rosa_at_10K_pop32:+.2f}pp</td></tr>
        <tr><td>ES, medium (n_pop=32, r=1) + full data (60K)</td><td>exp10b vs exp11b</td><td class='num'>{fmt_pct(e10b['test_acc'])}</td><td class='num'>{fmt_pct(e11b['test_acc'])}</td><td class='num pos'>{rosa_at_60K_pop32:+.2f}pp</td></tr>
        <tr><td>ES, strong (n_pop=128, r=4) + full data (60K)</td><td>exp10d vs exp11d</td><td class='num'>{fmt_pct(e10d['test_acc'])}</td><td class='num'>{fmt_pct(e11d['test_acc'])}</td><td class='num neg'>{rosa_at_60K_pop128:+.2f}pp</td></tr>
        <tr><td>Gradient (strongest) — lesion, no stopgrad (exp6)</td><td>A − C lesion</td><td class='num'>{fmt_pct(e6f['test_acc'])}</td><td class='num'>{fmt_pct(e6f['test_acc'] - 0.001)}</td><td class='num pos'>+0.10pp</td></tr>
        <tr><td>Gradient (strongest) — lesion, with stopgrad (exp1)</td><td>A − C lesion</td><td class='num'>{fmt_pct(e1f['test_acc'])}</td><td class='num'>{fmt_pct(e1f['test_acc'] - 0.0001)}</td><td class='num'>−0.01pp</td></tr>
      </tbody>
    </table>

    <p>The interpretation: <strong>ROSA is a weak inductive bias.</strong></p>
    <ul class='list-disc ml-6 mt-3 mb-3'>
      <li>When the optimizer is <em>underpowered</em>, the codebook can't develop class-discriminative entries on its own. ROSA's class-correlated guesses (drawn from training-stream patterns) bridge the gap and improve accuracy.</li>
      <li>When the optimizer is <em>overpowered</em> for the task, the codebook gets shaped precisely for digit classification regardless of ROSA. Now ROSA's discrete predictions — based on whichever-training-sample-had-this-suffix — are noise relative to the precise gradient signal, and they hurt.</li>
      <li>The 10K result hurts because <em>neither</em> the codebook nor ROSA has enough data: ROSA only sees ~10K bursts so its predictions are weak, and the optimizer has only 156 batches/epoch to shape the codebook. Both are underpowered, and ROSA can't help.</li>
    </ul>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>5 · ES never overfits</h3>

    <figure class='my-6'>
      <img src='{plot_urls['gap']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Train vs test across ES configs. Generalization gaps stay under 5pp.
        Gradient training of the same architecture (right) overfits to a ~10pp gap (train ~99%, test ~89%).
      </figcaption>
    </figure>

    <p class='mb-3'>
      A real architectural property of ES: <strong>perturbations can't memorize specific training samples</strong>, only find broad parameter regions whose accuracy is high on average. So ES naturally regularizes — train and test accuracy stay close throughout training.
    </p>
    <p>
      Compare to the decoupled ROSA-as-classifier run from Phase A (train 100% / test 54.47%, a 45pp gap). That extreme overfit was because ROSA's SAM is a literal memorizer. ES + ROSA, by contrast, only commits the <em>post-step</em> tokens to the SAM — by then the codebook is somewhere in the broad region ES is exploring, and individual-sample memorization can't develop.
    </p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>6 · Learning curves</h3>

    <figure class='my-6'>
      <img src='{plot_urls['curves']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Test accuracy across epochs for the 7 ES runs. Solid lines = with ROSA, dotted = no ROSA.
        The (red) 10K pair shows ROSA hurting throughout; the (purple) 60K-n_pop=32 pair shows ROSA helping;
        the (green) 60K-n_pop=128 pair shows ROSA hurting at convergence.
      </figcaption>
    </figure>

    <p>
      Note that <strong>ES is still climbing at epoch 29</strong> in the n_pop=128 runs (both with and without ROSA). Longer training would likely add another few percentage points. But the relative ROSA effect is already decisive — exp11d is consistently above exp10d throughout.
    </p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>7 · Discussion</h3>

    <p class='mb-3'>
      Combining Phase A and Phase B, we have eleven Fashion-MNIST runs spanning gradient and ES, with and without ROSA, three optimizer strengths, two data sizes, and the decoupled "ROSA-is-the-classifier" architecture. The story is internally consistent:
    </p>
    <ul class='list-disc ml-6 mb-3'>
      <li><strong>ROSA's information is largely redundant with the recent-window of token embeddings</strong>, which the output module sees directly. The "training-history correlations" ROSA uniquely encodes only matter when neither the optimizer nor the codebook can extract them from the window content. That's a narrow regime.</li>
      <li><strong>The non-differentiability of ROSA was not the structural blocker.</strong> Switching to ES (which doesn't need gradients) reveals ROSA's contribution only at one specific optimizer strength; at any strength the optimizer can solve the task without ROSA, it does so and prefers that path.</li>
      <li><strong>The decoupled ROSA-as-classifier setup hit a ceiling at 54.47%</strong> because reconstruction-optimal VQ-AE tokens aren't class-discriminative enough for ROSA's suffix-matching to recover the right digit on novel test images. ROSA can perfectly memorize training streams (100% train) but novel images fall through to wrong-class training matches.</li>
      <li><strong>ES regularizes the joint pipeline.</strong> A real architectural advantage independent of ROSA. Train-test gaps stay under 5pp across all 7 ES runs, vs ~10pp for gradient.</li>
    </ul>

    <p class='mb-3'><strong>The honest reading:</strong></p>

    <blockquote>
      ROSA's exact-match memory is a real signal. It's just not a useful signal for image classification with a learned codebook, because the codebook can be shaped to put class info directly in the tokens — making ROSA's lookups redundant. ROSA's natural domain is probably <em>discrete sequences with literal repetition</em>: text, code, or log streams, where exact-match has irreducible value that can't be replicated by a small clustered vocabulary.
    </blockquote>

    <p>
      As BlinkDL noted in the original ROSA-Plus README — ROSA is a "database-like" predictor that excels at surface structure (syntax, grammar, exact substrings) but lacks the deeper semantic understanding of an NN. Image classification on a 1024-symbol vocabulary turns out to be more semantics than surface, so the neural classifier wins.
    </p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>8 · What's next</h3>
    <ul class='list-disc ml-6 space-y-2'>
      <li><strong>Pivot to a domain where exact-match has irreducible value.</strong> Sequential prediction on text or character streams, where ROSA's database structure aligns with the natural data structure. The original ROSA-Plus task (Shakespearean generation) lives here.</li>
      <li><strong>One-shot / few-shot classification.</strong> ROSA can perfectly recall a single training example after one exposure — useful when the goal is "match this exemplar," not "average over a class."</li>
      <li><strong>Hybrid scaling.</strong> ES + ROSA at n_pop=128 was still climbing at epoch 29. A larger run (n_pop=256, 60 epochs, multiple seeds) could close more of the gap to gradient. But the marginal-utility-of-ROSA finding suggests this is mostly an ES-vs-gradient comparison rather than a ROSA test.</li>
      <li><strong>Better tokenizer for the decoupled setup.</strong> If the Stage 1 VQ-AE were replaced with a contrastive or class-aware encoder, the decoupled ROSA approach might break through the 55% ceiling — but at that point ROSA is hitching a ride on the encoder's class knowledge.</li>
    </ul>
  </section>

  <hr class='my-8 border-gray-300'>
  <p class='text-xs font-mono text-gray-500'>
    Code: <span class='mono'>projects/rosa/</span> ·
    Phase A report: <a class='text-blue-600 underline' href='{PHASE_A_URL}'>rosa_report_phaseA.html</a> ·
    All runs in <span class='mono'>projects/rosa/results.jsonl</span>
  </p>

</body>
</html>
"""

    return head + body


def main():
    print("Building plots...")
    f1 = plot_es_summary();        u1 = save_matplotlib_figure("rosa_phaseB_summary",     f1, format="svg"); plt.close(f1)
    f2 = plot_rosa_contribution(); u2 = save_matplotlib_figure("rosa_phaseB_contrib",     f2, format="svg"); plt.close(f2)
    f3 = plot_es_curves();         u3 = save_matplotlib_figure("rosa_phaseB_curves",      f3, format="svg"); plt.close(f3)
    f4 = plot_es_gen_gap();        u4 = save_matplotlib_figure("rosa_phaseB_gen_gap",     f4, format="svg"); plt.close(f4)

    plot_urls = {"summary": u1, "rosa_contrib": u2, "curves": u3, "gap": u4}

    print("Building HTML...")
    html_str = build_html(plot_urls)
    LOCAL_HTML.write_text(html_str)
    print(f"Saved local HTML: {LOCAL_HTML}  ({len(html_str):,} bytes)")

    print("Uploading to R2...")
    url = save_html_file("rosa_report_phaseB", LOCAL_HTML)
    print(f"\nReport: {url}")


if __name__ == "__main__":
    main()
