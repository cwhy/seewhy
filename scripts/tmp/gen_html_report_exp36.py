"""
Generate a self-contained HTML report for exp36 (linear probe + k-NN on exp33/34/35 encoders).

Usage:
    uv run python scripts/tmp/gen_html_report_exp36.py
"""

import sys, json, base64, io
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_media

EXP      = "exp36"
PROJECT  = ROOT / "projects" / "variations"
OUT_HTML = PROJECT / f"report_{EXP}.html"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Load exp36 results ─────────────────────────────────────────────────────────

results = []
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("experiment") == EXP:
        results.append(r)

if not results:
    print("No exp36 results found in results.jsonl — run experiments36.py first.")
    sys.exit(1)

# Keyed by source experiment
by_src = {r["source_exp"]: r for r in results}
src_order = [s for s in ["exp33", "exp34", "exp35"] if s in by_src]

# Also pull best recon_mse for each source exp for context
recon_mse = {}
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    r = json.loads(line)
    exp = r.get("experiment", "")
    if exp in ("exp33", "exp34", "exp35") and "final_recon_mse" in r:
        recon_mse[exp] = r["final_recon_mse"]

# ── Comparison bar chart ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#fafafa")
ax.set_facecolor("#fafafa")

x      = np.arange(len(src_order))
width  = 0.22
colors = {"probe_acc_test": "#2563eb", "knn5_acc": "#16a34a", "knn1_acc": "#dc2626"}
labels = {"probe_acc_test": "Linear probe (test)", "knn5_acc": "k-NN k=5", "knn1_acc": "k-NN k=1"}
offsets = [-width, 0, width]

for offset, (key, color) in zip(offsets, colors.items()):
    vals = [by_src[s][key] for s in src_order]
    bars = ax.bar(x + offset, vals, width, label=labels[key], color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(src_order, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, min(1.0, max(by_src[s]["probe_acc_test"] for s in src_order) + 0.12))
ax.set_title("CIFAR-10 classification from frozen encoder (exp36)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()

bar_buf = io.BytesIO()
fig.savefig(bar_buf, format="png", dpi=130, bbox_inches="tight")
bar_uri = "data:image/png;base64," + base64.b64encode(bar_buf.getvalue()).decode()
plt.close(fig)
print("bar chart rendered")

# ── Results table rows ─────────────────────────────────────────────────────────

table_rows = ""
for src in src_order:
    r = by_src[src]
    rmse = recon_mse.get(src, "—")
    rmse_str = f"{rmse:.6f}" if isinstance(rmse, float) else rmse
    table_rows += f"""
  <tr>
    <td><strong>{src}</strong></td>
    <td>{rmse_str}</td>
    <td><strong>{r['probe_acc_test']:.4f}</strong></td>
    <td>{r['probe_acc_train']:.4f}</td>
    <td>{r['knn1_acc']:.4f}</td>
    <td>{r['knn5_acc']:.4f}</td>
    <td>{int(r.get('time_probe_s', 0))} s</td>
  </tr>"""

# ── HTML ───────────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>exp36 — Linear Probe &amp; k-NN on Frozen Variations Encoder</title>
  <style>
    :root {{--blue:#2563eb;--green:#16a34a;--red:#dc2626;--gray:#6b7280;--light:#f8fafc;--border:#e2e8f0;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:"Segoe UI",system-ui,sans-serif;font-size:15px;line-height:1.7;color:#1e293b;background:#fff;max-width:900px;margin:0 auto;padding:2.5rem 1.5rem 5rem;}}
    h1{{font-size:2rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem;line-height:1.2;}}
    h2{{font-size:1.25rem;font-weight:700;color:#0f172a;border-bottom:2px solid var(--border);padding-bottom:0.4rem;margin:2.8rem 0 1rem;}}
    h3{{font-size:1rem;font-weight:600;color:#374151;margin:1.6rem 0 0.4rem;}}
    p{{margin-bottom:0.9rem;}}
    .subtitle{{color:var(--gray);font-size:0.93rem;margin:0.2rem 0 2rem;}}
    .summary-card{{background:var(--light);border:1px solid var(--border);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:2.2rem;}}
    .metrics{{display:flex;gap:1rem;flex-wrap:wrap;margin:0.8rem 0 1rem;}}
    .metric{{background:#fff;border:1px solid var(--border);border-radius:8px;padding:0.55rem 1rem;min-width:130px;flex:1;}}
    .metric .label{{font-size:0.75rem;color:var(--gray);text-transform:uppercase;letter-spacing:0.05em;}}
    .metric .value{{font-size:1.3rem;font-weight:700;color:#0f172a;}}
    .callout{{border-left:4px solid var(--blue);background:#eff6ff;border-radius:0 8px 8px 0;padding:0.85rem 1.2rem;margin:1rem 0 1.2rem;}}
    table{{border-collapse:collapse;width:100%;margin:1rem 0 1.5rem;font-size:0.9rem;}}
    th{{background:var(--light);text-align:left;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--gray);padding:0.5rem 0.9rem;border-bottom:2px solid var(--border);}}
    td{{padding:0.45rem 0.9rem;border-bottom:1px solid var(--border);}}
    tr:hover td{{background:var(--light);}}
    .plot{{display:block;max-width:100%;border-radius:8px;border:1px solid var(--border);margin:0.8rem 0 1.6rem;}}
    footer{{margin-top:4rem;padding-top:1.2rem;border-top:1px solid var(--border);font-size:0.8rem;color:var(--gray);text-align:center;}}
  </style>
</head>
<body>

<h1>Linear Probe &amp; k-NN on Frozen Variations Encoder</h1>
<p class="subtitle">
  exp36 &nbsp;·&nbsp; CIFAR-10 &nbsp;·&nbsp;
  Post-hoc evaluation of exp33 / exp34 / exp35 representations &nbsp;·&nbsp;
  Encoder frozen, no classification signal during training
</p>

<div class="summary-card">
  <strong style="font-size:1rem;">At a glance</strong>
  <div class="metrics">
    <div class="metric">
      <div class="label">Best probe acc (test)</div>
      <div class="value">{max(by_src[s]['probe_acc_test'] for s in src_order):.4f}</div>
    </div>
    <div class="metric">
      <div class="label">Best source exp</div>
      <div class="value">{max(src_order, key=lambda s: by_src[s]['probe_acc_test'])}</div>
    </div>
    <div class="metric">
      <div class="label">Probe epochs</div>
      <div class="value">{results[0]['probe_epochs']}</div>
    </div>
    <div class="metric">
      <div class="label">Probe batch</div>
      <div class="value">{results[0]['probe_batch']}</div>
    </div>
  </div>
  <p>
    The variations model was trained purely via reconstruction + variation recall loss — never
    with classification signal. Linear probing measures how much class-discriminative structure
    the encoder learned incidentally. A frozen encoder is probed with a single linear layer
    (W: 2048→10) trained with Adam + cosine LR for 100 epochs, plus k-NN at k=1 and k=5.
  </p>
</div>

<h2>Results</h2>

<img src="{bar_uri}" alt="comparison bar chart" class="plot">

<table>
  <tr>
    <th>Source Exp</th>
    <th>Recon MSE</th>
    <th>Probe Acc (test)</th>
    <th>Probe Acc (train)</th>
    <th>k-NN k=1</th>
    <th>k-NN k=5</th>
    <th>Probe time</th>
  </tr>
  {table_rows}
</table>

<div class="callout">
  <strong>Interpretation:</strong> CIFAR-10 linear probe on a recon-only encoder typically
  achieves 50–65%. Higher accuracy indicates the encoder's latent space has spontaneously
  organized by object class despite receiving no label supervision.
</div>

<h2>Probe configuration</h2>
<table>
  <tr><th>Setting</th><th>Value</th></tr>
  <tr><td>Probe epochs</td><td>{results[0]['probe_epochs']}</td></tr>
  <tr><td>Probe LR schedule</td><td>cosine decay {results[0]['probe_lr']} → {results[0]['probe_lr']*0.1:.0e}</td></tr>
  <tr><td>Probe batch size</td><td>{results[0]['probe_batch']}</td></tr>
  <tr><td>Latent dim</td><td>{results[0]['latent_dim']}</td></tr>
  <tr><td>k-NN metric</td><td>Euclidean (sklearn, n_jobs=-1)</td></tr>
  <tr><td>Input normalization</td><td>x / 255.0 (same as training)</td></tr>
  <tr><td>Encoder</td><td>Frozen weights from source exp (no grad)</td></tr>
</table>

<footer>
  Conditioned variation model &nbsp;·&nbsp; {EXP} &nbsp;·&nbsp;
  CIFAR-10 linear probe evaluation &nbsp;·&nbsp; generated 2026-04-20
</footer>
</body>
</html>
"""

r2_url = save_media(
    f"variations_{EXP}_report.html",
    io.BytesIO(html.encode("utf-8")),
    "text/html"
)
print(f"R2 URL → {r2_url}")
OUT_HTML.write_text(html, encoding="utf-8")
print(f"Local  → {OUT_HTML}")
