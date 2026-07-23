"""
Generate the Phase A report: methodology, experiments, results, plots.
Saves HTML locally to projects/rosa/report_phaseA.html and uploads to R2.

Usage: uv run python projects/rosa/scripts/gen_report_phaseA.py
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

LOCAL_HTML = PROJ_DIR / "report_phaseA.html"
TMP        = PROJ_DIR / "scripts" / "tmp"


# ── Load all results ──────────────────────────────────────────────────────────

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


def load_tmp_json(name):
    with open(TMP / name) as f:
        return json.load(f)


jsonl_rows = by_exp(load_jsonl(PROJ_DIR / "results.jsonl"))
vq_mnist   = load_tmp_json("vq_only_baseline_result.mnist.json")
vq_fmnist  = load_tmp_json("vq_only_baseline_result.fashion_mnist.json")
dense_mnist  = load_tmp_json("dense_baseline_result.mnist.json")
dense_fmnist = load_tmp_json("dense_baseline_result.fashion_mnist.json")
decoupled    = load_tmp_json("decoupled_rosa.fashion_mnist.json")

# Lesion numbers (read from logs by hand into here)
LESION_EXP1_FMNIST = {"A_zero": 0.8899, "B_const": 0.8899, "C_real": 0.8900}
LESION_EXP1N_FMNIST = {"A_zero": 0.8913, "B_const": 0.8911, "C_real": 0.8912}
LESION_EXP6_FMNIST = {"A_zero": 0.8904, "B_const": 0.8905, "C_real": 0.8914}


# ── Plot 1: Test accuracy on Fashion-MNIST, all variants ──────────────────────

def plot_fmnist_summary():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = [
        ("vq_only\n(no ROSA, no NO_OUT)",              vq_fmnist["final_test_acc"],     "#888"),
        ("dense\n(no ROSA, NO_OUT class)",             dense_fmnist["final_test_acc"],  "#888"),
        ("exp1\n(stopgrad ROSA)",                      jsonl_rows["exp1_fmnist"]["test_acc"],  "#3b82f6"),
        ("exp1n\n(no indicators)",                     jsonl_rows["exp1n_fmnist"]["test_acc"], "#3b82f6"),
        ("exp6\n(no stopgrad)",                        jsonl_rows["exp6_fmnist"]["test_acc"],  "#3b82f6"),
        ("decoupled v1\n(SAM-1NN argmax)",             0.4163,                                  "#dc2626"),
        ("decoupled v2\n(LM distribution)",            decoupled["test_acc"],                   "#dc2626"),
    ]
    xs = np.arange(len(bars))
    accs = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    ax.bar(xs, accs, color=cols, edgecolor="black", linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.005, f"{a*100:.2f}%", ha="center", fontsize=9, family="monospace")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5, family="monospace")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy", family="monospace")
    ax.set_title("Fashion-MNIST: ROSA contribution across variants",
                 family="monospace", fontsize=12)
    ax.axhline(y=dense_fmnist["final_test_acc"], color="#888", linestyle="--",
               linewidth=0.5, alpha=0.6)
    ax.axhline(y=0.10, color="#888", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.text(6.4, 0.10 + 0.01, "chance (10%)", fontsize=7, color="#888", family="monospace")
    ax.text(6.4, dense_fmnist["final_test_acc"] - 0.025, "dense baseline",
            fontsize=7, color="#888", family="monospace")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    return fig


# ── Plot 2: Train vs test gap, highlighting decoupled overfit ─────────────────

def plot_train_test_gap():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    rows = [
        ("vq_only",              vq_fmnist["final_train_acc"],     vq_fmnist["final_test_acc"]),
        ("dense",                dense_fmnist["final_train_acc"],  dense_fmnist["final_test_acc"]),
        ("exp1 (stopgrad)",      jsonl_rows["exp1_fmnist"]["train_acc"],  jsonl_rows["exp1_fmnist"]["test_acc"]),
        ("exp6 (no stopgrad)",   jsonl_rows["exp6_fmnist"]["train_acc"],  jsonl_rows["exp6_fmnist"]["test_acc"]),
        ("decoupled v2",         decoupled["train_acc"],                  decoupled["test_acc"]),
    ]
    xs = np.arange(len(rows))
    train = [r[1] for r in rows]
    test  = [r[2] for r in rows]
    width = 0.36
    ax.bar(xs - width/2, train, width, label="train (10K)", color="#9ca3af", edgecolor="black", linewidth=0.5)
    ax.bar(xs + width/2, test,  width, label="test (10K)",  color="#3b82f6", edgecolor="black", linewidth=0.5)
    for i, (t, v) in enumerate(zip(train, test)):
        ax.text(i - width/2, t + 0.005, f"{t*100:.1f}", ha="center", fontsize=8, family="monospace")
        ax.text(i + width/2, v + 0.005, f"{v*100:.1f}", ha="center", fontsize=8, family="monospace")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, family="monospace")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", family="monospace")
    ax.set_title("Fashion-MNIST: train vs. test accuracy. Decoupled ROSA shows extreme overfitting.",
                 family="monospace", fontsize=11)
    ax.legend(loc="lower left", prop={"family": "monospace", "size": 9})
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    return fig


# ── Plot 3: Lesion deltas — does ROSA contribute when we let it? ──────────────

def plot_lesion():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    rows = [
        ("exp1\n(stopgrad ON)",        LESION_EXP1_FMNIST),
        ("exp1n\n(no indicators)",     LESION_EXP1N_FMNIST),
        ("exp6\n(stopgrad OFF)",       LESION_EXP6_FMNIST),
    ]
    xs = np.arange(len(rows))
    width = 0.27
    A = [r[1]["A_zero"]  for r in rows]
    B = [r[1]["B_const"] for r in rows]
    C = [r[1]["C_real"]  for r in rows]
    ax.bar(xs - width, A, width, label="A: rosa = 0",          color="#9ca3af", edgecolor="black", linewidth=0.5)
    ax.bar(xs,         B, width, label="B: rosa = const",      color="#a78bfa", edgecolor="black", linewidth=0.5)
    ax.bar(xs + width, C, width, label="C: rosa = real ROSA",  color="#3b82f6", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, family="monospace")
    ax.set_ylim(0.880, 0.895)
    ax.set_ylabel("Test accuracy", family="monospace")
    ax.set_title("Lesion: does ROSA actually contribute? "
                 "C − A is the answer (≈0 for stopgrad, +0.10pp for no-stopgrad)",
                 family="monospace", fontsize=10.5)
    ax.legend(loc="lower right", prop={"family": "monospace", "size": 9})
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    for i, (a, b, c) in enumerate(zip(A, B, C)):
        ax.text(i - width, a + 0.0001, f"{a*100:.2f}", ha="center", fontsize=8, family="monospace")
        ax.text(i,         b + 0.0001, f"{b*100:.2f}", ha="center", fontsize=8, family="monospace")
        ax.text(i + width, c + 0.0001, f"{c*100:.2f}", ha="center", fontsize=8, family="monospace")
    return fig


# ── Plot 4: Learning curves for exp1 + exp6 ───────────────────────────────────

def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1 = axes
    for label, key, color in [
        ("exp1 (stopgrad)",    "exp1_fmnist",  "#3b82f6"),
        ("exp1n (no indicators)", "exp1n_fmnist", "#10b981"),
        ("exp6 (no stopgrad)", "exp6_fmnist",  "#dc2626"),
    ]:
        h = jsonl_rows[key]["history"]
        ep = list(range(len(h["test_acc"])))
        ax0.plot(ep, h["test_acc"], label=label, color=color, linewidth=1.4)
        ax1.plot(ep, h["cls"],      label=label, color=color, linewidth=1.4)
    ax0.set_xlabel("epoch", family="monospace")
    ax0.set_ylabel("test_acc", family="monospace")
    ax0.set_title("Test accuracy", family="monospace", fontsize=11)
    ax0.set_ylim(0.7, 0.92)
    ax0.legend(loc="lower right", prop={"family": "monospace", "size": 9})
    ax0.grid(alpha=0.2, linewidth=0.5)
    ax1.set_xlabel("epoch", family="monospace")
    ax1.set_ylabel("classification loss", family="monospace")
    ax1.set_title("Total classification loss (CE + λ_digit·CE_digit)",
                  family="monospace", fontsize=11)
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", prop={"family": "monospace", "size": 9})
    ax1.grid(alpha=0.2, linewidth=0.5)
    fig.suptitle("Learning curves: Fashion-MNIST, exp1 / exp1n / exp6",
                 family="monospace", fontsize=12)
    fig.tight_layout()
    return fig


# ── Plot 5: VQ-AE reconstruction loss curve from decoupled ────────────────────

def plot_vqae_curve():
    fig, ax = plt.subplots(figsize=(7, 4))
    h = decoupled["vqae_history"]
    ep = list(range(len(h["test_recon"])))
    ax.plot(ep, h["recon"],      label="train recon", color="#9ca3af", linewidth=1.2)
    ax.plot(ep, h["test_recon"], label="test recon",  color="#3b82f6", linewidth=1.4)
    ax.set_xlabel("epoch", family="monospace")
    ax.set_ylabel("reconstruction MSE", family="monospace")
    ax.set_title("Phase A · Stage 1 — VQ-AE training (Fashion-MNIST images)",
                 family="monospace", fontsize=11)
    ax.legend(loc="upper right", prop={"family": "monospace", "size": 9})
    ax.grid(alpha=0.2, linewidth=0.5)
    return fig


# ── Build the HTML ────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def build_html(plot_urls: dict[str, str]) -> str:
    head = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>ROSA on MNIST — Phase A report</title>\n"
        "<script src='https://cdn.tailwindcss.com'></script>\n"
        "<style>\n"
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }\n"
        "  pre, code, .mono { font-family: 'SF Mono', 'Menlo', monospace; }\n"
        "  table { border-collapse: collapse; }\n"
        "  th, td { padding: 0.4rem 0.8rem; border: 1px solid #e5e7eb; font-size: 0.92rem; }\n"
        "  th { background: #f3f4f6; text-align: left; }\n"
        "  .num { text-align: right; font-family: 'SF Mono', 'Menlo', monospace; }\n"
        "  .hi { background: #fef3c7; font-weight: 600; }\n"
        "  blockquote { border-left: 4px solid #d1d5db; padding: 0.5rem 1rem; color: #4b5563; "
        "margin: 1rem 0; background: #f9fafb; }\n"
        "</style>\n"
        "</head>\n"
    )

    e1m  = jsonl_rows["exp1"]
    e1f  = jsonl_rows["exp1_fmnist"]
    e1nf = jsonl_rows["exp1n_fmnist"]
    e6f  = jsonl_rows["exp6_fmnist"]

    body = f"""
<body class='bg-white p-6 max-w-4xl mx-auto'>

  <h1 class='text-3xl font-mono mb-2'>ROSA on MNIST</h1>
  <h2 class='text-xl font-mono text-gray-500 mb-6'>Memory, memorization, and the limits of statistical tokens</h2>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>TL;DR</h3>
    <ul class='list-disc ml-6 space-y-2'>
      <li>We tried to use <strong>ROSA</strong>
        (a pure-statistics suffix-automaton next-token predictor from
        <a href='https://github.com/bcml-labs/rosa-plus' class='text-blue-600 underline'>bcml-labs/rosa-plus</a>)
        as the memory layer of an MNIST/Fashion-MNIST classifier — neural input + ROSA + neural output.</li>
      <li>On <strong>MNIST</strong>, ROSA's contribution is small and noisy: <span class='mono'>+0.15pp</span> over the no-ROSA baseline.</li>
      <li>On <strong>Fashion-MNIST</strong>, ROSA is <em>actively ignored</em> by the output module
        when its predicted-token-embedding is stop-gradient'd — a lesion test gives
        <span class='mono'>{LESION_EXP1_FMNIST['C_real']*100 - LESION_EXP1_FMNIST['A_zero']*100:+.2f}pp</span>.
        Removing the stop_grad lets ROSA contribute exactly <span class='mono'>+0.10pp</span> — real, but tiny.</li>
      <li>The most striking finding is the <strong>decoupled experiment</strong>: train a VQ-AE on images, build ROSA over the resulting tokens, classify by reading off ROSA's distribution at the &lt;LABEL&gt; position. Result on Fashion-MNIST: <span class='mono hi'>train 100% / test 54.47%</span>. ROSA <em>perfectly memorizes</em> training streams but generalizes poorly because reconstruction-optimal tokens aren't class-discriminative enough.</li>
      <li>The takeaway: ROSA's exact-match memory is real, but on image-classification benchmarks the recent-window of token content already encodes the same information ROSA could provide — and the parts ROSA <em>uniquely</em> knows (training-history correlations) get washed out by codebook collisions.</li>
    </ul>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>1 · Background — what is ROSA?</h3>
    <p class='mb-3'>
      ROSA (introduced by <a class='text-blue-600 underline' href='https://x.com/BlinkDL_AI/status/1976912771985146184'>BlinkDL</a> as an extension to RWKV) is a <strong>pure-statistics next-token predictor</strong> built around a <strong>suffix automaton</strong> (SAM). After ingesting a token stream, ROSA can be queried with any prefix and returns:
    </p>
    <ol class='list-decimal ml-6 mb-3'>
      <li><strong>SAM-deterministic</strong>: the next token after the rightmost training match of the longest suffix of the prefix that exists in the SAM (a literal "1-NN by token-suffix" rule).</li>
      <li><strong>Witten–Bell fallback</strong>: when no deterministic answer is found, an interpolated n-gram distribution along the suffix-link chain.</li>
    </ol>
    <p class='mb-3'>
      ROSA has <em>zero trainable parameters</em>. It's a frozen data structure that grows monotonically as tokens stream in. The bcml-labs/rosa-plus implementation is character-level for English text; we ported it to integer tokens (<a class='text-blue-600 underline' href='https://github.com/bcml-labs/rosa-plus/blob/main/rosaplus.py'>rosaplus.py</a> &rarr; <span class='mono'>lib/rosa_core.py</span>).
    </p>
    <p>
      The question we set out to ask: <strong>can ROSA's exact-match memory provide a useful prior for visual classification</strong> when slotted into a learned tokenization pipeline?
    </p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>2 · Architecture</h3>
    <p class='mb-3'>The pipeline (<span class='mono'>experiments1.py</span>):</p>
    <pre class='bg-gray-50 border border-gray-200 p-3 text-xs overflow-x-auto rounded'>image (28×28)                                  label (digit 0..9)
   │                                                  │
   ▼                                                  ▼
recurrent input MLP × K_SAMPLE=16 steps     recurrent input MLP × K_LABEL=4 steps
(RMSNorm + GELU + residual)                 (same shared body, different head)
   │                                                  │
   ▼ z (continuous)                                   ▼ z (continuous)
DiVeQ quantize (V=1024, D=64)               DiVeQ quantize
   │ argmin → integer IDs                             │
   ▼                                                  ▼
[ &lt;SAMPLE&gt;=0,  s_1..s_16,  &lt;LABEL&gt;=1,  l_1..l_4 ]    ← 22-token stream
   │
   ▼
ROSA SAM (continual; grows across all training)
   │ predicted next-token ID at every position
   ▼
codebook[predicted_id]  →  rosa_pred_emb (stop_grad in exp1; no stopgrad in exp6)
                              │
   ┌─ 8-token sliding window ─┘
   │
   ▼
output module: 11-class softmax (10 digits + NO_OUTPUT)
   - fires at every position
   - target: digit at LABEL position (index 17), NO_OUTPUT elsewhere
   - λ_digit_boost = 20 to counter ~21:1 imbalance</pre>
    <p class='mt-3'>Stage-1 pipeline (Phase A, <span class='mono'>scripts/tmp/decoupled_rosa.py</span>) drops the joint loss and trains the encoder + codebook as a pure VQ-AE on images, then builds ROSA over the resulting frozen tokens, then predicts digits by reading ROSA's distribution at &lt;LABEL&gt; with zero trainable parameters in the output head.</p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>3 · Experiments &amp; results</h3>
    <p class='mb-3'>All runs use V=1024, D=64, K_sample=16, K_label=4, batch 128, 50 epochs, Adam lr=3e-4 cosine to 5%, seed 42. Times are wall-clock on one RTX 4090.</p>

    <table class='w-full mb-4'>
      <thead><tr><th>Run</th><th>Dataset</th><th class='num'>Test acc</th><th class='num'>Train acc</th><th class='num'>cb_used</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>0.3 vq_only</td><td>MNIST</td><td class='num'>{fmt_pct(vq_mnist['final_test_acc'])}</td><td class='num'>{fmt_pct(vq_mnist['final_train_acc'])}</td><td class='num'>{vq_mnist['cb_used']}/1022</td><td>VQ-MLP, no ROSA, no NO_OUTPUT — neural ceiling</td></tr>
        <tr><td>0.4 dense</td><td>MNIST</td><td class='num'>{fmt_pct(dense_mnist['final_test_acc'])}</td><td class='num'>{fmt_pct(dense_mnist['final_train_acc'])}</td><td class='num'>{dense_mnist['cb_used']}/1022</td><td>NO_OUTPUT mechanism, no ROSA</td></tr>
        <tr><td>exp1</td><td>MNIST</td><td class='num hi'>{fmt_pct(e1m['test_acc'])}</td><td class='num'>{fmt_pct(e1m['train_acc'])}</td><td class='num'>{e1m['cb_used']}/1022</td><td>full ROSA pipeline, +0.15pp over 0.3</td></tr>
        <tr><td colspan='6' style='border-top: 2px solid #d1d5db; height: 6px; background: #f9fafb;'></td></tr>
        <tr><td>0.3 vq_only</td><td>Fashion-MNIST</td><td class='num'>{fmt_pct(vq_fmnist['final_test_acc'])}</td><td class='num'>{fmt_pct(vq_fmnist['final_train_acc'])}</td><td class='num'>{vq_fmnist['cb_used']}/1022</td><td></td></tr>
        <tr><td>0.4 dense</td><td>Fashion-MNIST</td><td class='num'>{fmt_pct(dense_fmnist['final_test_acc'])}</td><td class='num'>{fmt_pct(dense_fmnist['final_train_acc'])}</td><td class='num'>{dense_fmnist['cb_used']}/1022</td><td>dense fire is essentially lossless on Fashion too</td></tr>
        <tr><td>exp1</td><td>Fashion-MNIST</td><td class='num'>{fmt_pct(e1f['test_acc'])}</td><td class='num'>{fmt_pct(e1f['train_acc'])}</td><td class='num'>{e1f['cb_used']}/1022</td><td>−0.49pp vs dense — ROSA appears to hurt</td></tr>
        <tr><td>exp1n</td><td>Fashion-MNIST</td><td class='num'>{fmt_pct(e1nf['test_acc'])}</td><td class='num'>{fmt_pct(e1nf['train_acc'])}</td><td class='num'>{e1nf['cb_used']}/1022</td><td>indicators removed; basically unchanged</td></tr>
        <tr><td>exp6</td><td>Fashion-MNIST</td><td class='num hi'>{fmt_pct(e6f['test_acc'])}</td><td class='num'>{fmt_pct(e6f['train_acc'])}</td><td class='num'>{e6f['cb_used']}/1022</td><td>no stop_grad on rosa_pred_emb</td></tr>
        <tr><td>decoupled v1 (SAM-1NN)</td><td>Fashion-MNIST</td><td class='num'>{fmt_pct(0.4163)}</td><td class='num'>{fmt_pct(1.0)}</td><td class='num'>~138/1012</td><td>ROSA is the entire classifier; argmax of deterministic rule</td></tr>
        <tr><td>decoupled v2 (LM dist)</td><td>Fashion-MNIST</td><td class='num hi'>{fmt_pct(decoupled['test_acc'])}</td><td class='num hi'>{fmt_pct(decoupled['train_acc'])}</td><td class='num'>~138/1012</td><td>same setup, argmax over LM digit-probs</td></tr>
      </tbody>
    </table>

    <figure class='my-6'>
      <img src='{plot_urls['summary']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Test accuracy across all variants on Fashion-MNIST. Blue = joint (ROSA in the loop); red = decoupled (ROSA is the classifier); grey = no-ROSA baselines.
      </figcaption>
    </figure>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>4 · Lesion: is ROSA actually contributing?</h3>
    <p class='mb-3'>For each trained joint model (exp1, exp1n, exp6) we re-evaluated on the test set with three different values of <span class='mono'>rosa_pred_emb</span>:</p>
    <ul class='list-disc ml-6 mb-3'>
      <li><strong>A · zero</strong>: <span class='mono'>rosa_pred_emb = 0</span> at every position.</li>
      <li><strong>B · const</strong>: <span class='mono'>rosa_pred_emb = codebook[0]</span> (a fixed indicator-style vector).</li>
      <li><strong>C · real</strong>: ROSA's actual predicted-token embedding, looked up from the saved 3.3 GB SAM.</li>
    </ul>
    <p class='mb-3'><strong>C − A</strong> measures ROSA's actual contribution. ≈0 means the output module ignores ROSA; positive means ROSA contributes; negative means ROSA hurts.</p>

    <figure class='my-6'>
      <img src='{plot_urls['lesion']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Lesion: real ROSA vs. zero-and-constant ablations. Stop_grad on (exp1, exp1n) → ROSA is exactly ignored.
        Stop_grad off (exp6) → first non-zero contribution: real ROSA gives +0.10pp over zeroed ROSA.
      </figcaption>
    </figure>

    <p>The pattern matches a failure mode the plan explicitly anticipated:</p>
    <blockquote>
      "The output module may learn to ignore ROSA's feature (it's stop-grad'd anyway) and just classify from the recent-window pool — degenerating to a vanilla VQ-MLP classifier."
    </blockquote>
    <p>That's exactly what happens. The W=8 sliding window of token embeddings already contains everything that <em>determines</em> ROSA's prediction (since ROSA is a deterministic function of recent stream + entire training history). The genuinely new information ROSA provides is the <em>training-history correlations</em>, but with stop_grad the output module has no way to ask the codebook to encode that history usefully — so it learns to zero out the ROSA channel.</p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>5 · Phase A — ROSA as the entire classifier (decoupled)</h3>
    <p class='mb-3'>Architecturally we wanted to test the opposite extreme: <strong>force ROSA to be the answer</strong>, with as little neural assist as possible. Three stages:</p>
    <ol class='list-decimal ml-6 mb-3'>
      <li><strong>VQ-AE</strong> on Fashion-MNIST images only (no labels). Recurrent input MLP + codebook + simple decoder, 50 epochs, reconstruction MSE + commitment loss.</li>
      <li><strong>Build ROSA.</strong> Frozen encoder. Per-sample stream <span class='mono'>[&lt;SAMPLE&gt;, s_1..s_16, &lt;LABEL&gt;, y_token]</span> where the digit's token ID is reserved (<span class='mono'>y + 2</span> ∈ <span class='mono'>{{2..11}}</span>). Feed all 60K streams to the SAM, build the Witten–Bell LM.</li>
      <li><strong>Eval (zero parameters).</strong> Encode test image → 16 tokens → walk SAM along <span class='mono'>[&lt;SAMPLE&gt;, s_1..s_16, &lt;LABEL&gt;]</span> → ROSA's distribution at the resulting state → sum probability mass over IDs 2..11 → argmax.</li>
    </ol>

    <figure class='my-6'>
      <img src='{plot_urls['vqae']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Stage 1: VQ-AE training on Fashion-MNIST. Test reconstruction MSE plateaus at 0.0136 — typical for VQ-VAE-class autoencoders on this dataset.
      </figcaption>
    </figure>

    <p class='mb-3'>Two prediction modes were tested in Stage 3:</p>

    <table class='w-full mb-4'>
      <thead><tr><th>Mode</th><th class='num'>Test acc</th><th class='num'>Train acc</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>v1 — SAM-deterministic shortcut</td><td class='num'>{fmt_pct(0.4163)}</td><td class='num'>{fmt_pct(1.0)}</td><td>1-NN-by-suffix; brittle on novel test images</td></tr>
        <tr><td>v2 — Witten–Bell distribution argmax</td><td class='num hi'>{fmt_pct(decoupled['test_acc'])}</td><td class='num hi'>{fmt_pct(decoupled['train_acc'])}</td><td>aggregates counts up the suffix chain</td></tr>
      </tbody>
    </table>

    <figure class='my-6'>
      <img src='{plot_urls['gap']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Train vs test on Fashion-MNIST. The decoupled ROSA variant has the largest train–test gap by a wide margin — train 100%, test 54%, a textbook signature of memorization without generalization.
      </figcaption>
    </figure>

    <p class='mb-3'>The 100% train / 54% test split is the most interesting result of the whole project. ROSA is doing exactly what it was designed to do — perfectly remember the training stream — and that's why it can recover every training-set digit by direct lookup. But for a novel test image whose token sequence isn't an exact training match, ROSA's suffix-fallback finds <em>some</em> training stream that ends in this prefix, and that training stream's class is the one ROSA returns. With reconstruction-optimal tokens, similar-looking-but-different-class images often share token prefixes, so the wrong class wins.</p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>6 · Discussion</h3>
    <p class='mb-3'><strong>Why doesn't ROSA help in the joint setting?</strong> The output module's recent-window of W=8 tokens already encodes everything that determines ROSA's argmax prediction (since ROSA is a deterministic function of recent stream + entire training history). The truly-novel info ROSA carries is "training-history correlations on this exact suffix" — but:</p>
    <ul class='list-disc ml-6 mb-3'>
      <li>With stop_grad, the output module can't shape what those training-history correlations <em>mean</em>;</li>
      <li>Without stop_grad, it can — but the gradient through <span class='mono'>codebook[predicted_id]</span> is biased (changing the codebook would change ROSA's choice, but we don't account for that). The result is a real but tiny improvement (+0.10pp).</li>
    </ul>

    <figure class='my-6'>
      <img src='{plot_urls['curves']}' class='w-full border border-gray-200 rounded'>
      <figcaption class='text-xs font-mono text-gray-500 mt-2 text-center'>
        Learning curves for the three joint variants on Fashion-MNIST. exp6 (no stopgrad) reaches roughly the same final test accuracy as exp1 (stopgrad), confirming the lesion result: the gradient through ROSA's path is doing a small amount of useful work.
      </figcaption>
    </figure>

    <p class='mb-3'><strong>Why does decoupled ROSA-as-classifier overfit?</strong> Because a SAM is a perfect memory device: it can answer "what came after this exact suffix in training?" with literal recall. Train accuracy is 100% by construction — every training stream's <span class='mono'>&lt;LABEL&gt;</span>-state has its own correct y_token recorded. Test accuracy is whatever the codebook's class-discriminativity allows: with reconstruction-optimal tokens, similar-looking items in different classes share suffixes, so the SAM hands back the wrong class.</p>

    <p class='mb-3'><strong>The ceiling for "ROSA-pure" on Fashion-MNIST is ~55%.</strong> Real but unimpressive against the 89% dense baseline. The 35 percentage points of headroom would have to come from making the tokens more class-discriminative — but as soon as we do that with supervision (a small classification head in Stage 1, e.g.) we're back to a regular classifier with ROSA bolted on.</p>

    <p><strong>The honest interpretation:</strong> ROSA's exact-match statistics are a <em>real</em> signal — strong enough to lift random tokens from chance to 41% on Fashion-MNIST and to perfectly recover training data — but on image classification benchmarks the W-token sliding window is already capacity-rich enough that ROSA's contribution is largely redundant. ROSA's natural domain is probably <em>discrete sequences with literal repetition</em> (text, code, log streams) where its database-style memory has more unique value. As BlinkDL noted in the original ROSA-Plus README: ROSA is a "database-like" predictor that excels at surface structure — coherent on syntax, brittle on semantics — and image classification on a small alphabet of clustered codebook tokens turns out to be more "semantics" than "surface".</p>
  </section>

  <section class='mb-8'>
    <h3 class='text-lg font-mono mt-6 mb-3 border-b pb-1'>7 · What's next</h3>
    <ul class='list-disc ml-6 space-y-2'>
      <li><strong>EGGROLL/ES with ROSA in the loop</strong> — black-box-optimize the input MLP + codebook on classification accuracy with ROSA inside the loss. Forces the optimizer to discover tokens that make ROSA's predictions accurate. Slower, but architecturally aligns the optimizer with the structural role of ROSA.</li>
      <li><strong>Pivot the question</strong> — if ROSA's natural strength is exact-match memory, image classification was the wrong testbed. A small one-shot/few-shot benchmark, or a sequential-prediction task (next-pixel, next-character) is closer to ROSA's home turf.</li>
      <li><strong>ROSA-as-logit-prior</strong> — use ROSA's full distribution to initialize the output module's logits, with the neural network producing only a residual. ROSA becomes structurally unavoidable; the decoupled finding suggests this would land somewhere between the joint and decoupled numbers.</li>
    </ul>
  </section>

  <hr class='my-8 border-gray-300'>
  <p class='text-xs font-mono text-gray-500'>
    Code: <span class='mono'>projects/rosa/</span> ·
    Plan: <span class='mono'>projects/rosa/plan.md</span> ·
    Concepts: <span class='mono'>projects/rosa/concepts.md</span>
  </p>

</body>
</html>
"""

    return head + body


def main():
    print("Building plots...")
    fig1 = plot_fmnist_summary()
    url1 = save_matplotlib_figure("rosa_phaseA_summary", fig1, format="svg")
    plt.close(fig1)

    fig2 = plot_train_test_gap()
    url2 = save_matplotlib_figure("rosa_phaseA_gap", fig2, format="svg")
    plt.close(fig2)

    fig3 = plot_lesion()
    url3 = save_matplotlib_figure("rosa_phaseA_lesion", fig3, format="svg")
    plt.close(fig3)

    fig4 = plot_learning_curves()
    url4 = save_matplotlib_figure("rosa_phaseA_curves", fig4, format="svg")
    plt.close(fig4)

    fig5 = plot_vqae_curve()
    url5 = save_matplotlib_figure("rosa_phaseA_vqae", fig5, format="svg")
    plt.close(fig5)

    plot_urls = {"summary": url1, "gap": url2, "lesion": url3,
                 "curves": url4, "vqae": url5}

    print("Building HTML...")
    html_str = build_html(plot_urls)
    LOCAL_HTML.write_text(html_str)
    print(f"Saved local HTML: {LOCAL_HTML}  ({len(html_str):,} bytes)")

    print("Uploading to R2...")
    url = save_html_file("rosa_report_phaseA", LOCAL_HTML)
    print(f"\nReport: {url}")


if __name__ == "__main__":
    main()
