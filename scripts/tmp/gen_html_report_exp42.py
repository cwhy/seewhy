"""
Generate a self-contained HTML report for exp42
(Local IMLE, N_VAR=96, 100ep schedule, stop_gradient on kNN).
Usage: uv run python scripts/tmp/gen_html_report_exp42.py
"""
import sys, pickle, json, base64, io
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_lib.media import save_media

EXP     = "exp42"
PROJECT = ROOT / "projects" / "variations"
PLOTS   = PROJECT / "plots"

with open(PROJECT / f"history_{EXP}.pkl", "rb") as f:
    history = pickle.load(f)

result = None
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("experiment") == EXP:
        result = r; break

n_params_val = result["n_params"]
time_s       = result["time_s"]
final_recon  = result["final_recon_mse"]
final_recall = result["final_recall_mse"]
latent_dim   = result["latent_dim"]
n_var        = result["n_var_samples"]
batch_size   = result["batch_size"]
alpha        = result["alpha"]
dc_c1, dc_c2, dc_c3, dc_stem = result["dc_c1"], result["dc_c2"], result["dc_c3"], result["dc_stem"]
enc_c1, enc_c2, enc_c3, enc_fc1 = result["enc_c1"], result["enc_c2"], result["enc_c3"], result["enc_fc1"]
n_cat, vocab_size = result["n_cat"], result["vocab_size"]
e_dim        = result.get("e_dim", result.get("cond_dim", 16) // max(1, result.get("n_cat", 4)))
k_nn         = result.get("k_nn", 16)
epochs_run   = len(history["loss"])
best_recall  = min(v for _, v in result["eval_recall_mse_curve"])

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def to_uri(p): return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
curves_uri = to_uri(PLOTS / f"variations_{EXP}_curves.png")
recon_uri  = to_uri(PLOTS / f"variations_{EXP}_recon.png")
class_uris = {c: to_uri(PLOTS / f"variations_{EXP}_var_class{c}.png") for c in range(10)}
print("plots loaded")

def get_r(exp):
    for line in (PROJECT / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("experiment") == exp: return r
    return {}

def recall_f(exp): r=get_r(exp); c=r.get("eval_recall_mse_curve",[]); return c[-1][1] if c else None
def recon_f(exp):  r=get_r(exp); return r.get("final_recon_mse", r.get("final_mse"))
def fmt(v): return f"{v:.5f}" if v is not None else "—"

OUTER_VALS_TEXT = "(0,0) (4,0) (0,4) (4,4)"
class_sections = "".join(f"""
    <div class="class-block">
      <h3>Class {c} &mdash; {CIFAR_CLASSES[c].capitalize()}</h3>
      <p class="caption">
        <strong>Top row:</strong> source + 10 global NN. &nbsp;
        <strong>Middle row:</strong> zero-cat decode + closest of 256 variations to each NN. &nbsp;
        <strong>Bottom:</strong> 16×16 mosaic — cat1×cat2, quadrants = cat3×cat4 {OUTER_VALS_TEXT}.
      </p>
      <img src="{class_uris[c]}" alt="class {c}" class="plot full-width">
    </div>""" for c in range(10))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{EXP} — Local IMLE N_VAR=96 100ep CIFAR-10</title>
  <style>
    :root{{--blue:#2563eb;--red:#dc2626;--green:#059669;--gray:#6b7280;--light:#f8fafc;--border:#e2e8f0;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:"Segoe UI",system-ui,sans-serif;font-size:15px;line-height:1.7;color:#1e293b;
         background:#fff;max-width:980px;margin:0 auto;padding:2.5rem 1.5rem 5rem;}}
    h1{{font-size:2rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem;}}
    h2{{font-size:1.25rem;font-weight:700;color:#0f172a;border-bottom:2px solid var(--border);
        padding-bottom:0.4rem;margin:2.8rem 0 1rem;}}
    h3{{font-size:1rem;font-weight:600;margin:1.6rem 0 0.4rem;}}
    p{{margin-bottom:0.9rem;}}
    .subtitle{{color:var(--gray);font-size:0.93rem;margin:0.2rem 0 2rem;}}
    .summary-card{{background:var(--light);border:1px solid var(--border);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:2.2rem;}}
    .metrics{{display:flex;gap:1rem;flex-wrap:wrap;margin:0.8rem 0 1rem;}}
    .metric{{background:#fff;border:1px solid var(--border);border-radius:8px;padding:0.55rem 1rem;min-width:130px;flex:1;}}
    .metric .label{{font-size:0.75rem;color:var(--gray);text-transform:uppercase;letter-spacing:0.05em;}}
    .metric .value{{font-size:1.3rem;font-weight:700;color:#0f172a;}}
    .warn{{border-left:4px solid #f59e0b;background:#fffbeb;border-radius:0 8px 8px 0;padding:0.85rem 1.2rem;margin:1rem 0 1.2rem;}}
    .callout{{border-left:4px solid var(--blue);background:#eff6ff;border-radius:0 8px 8px 0;padding:0.85rem 1.2rem;margin:1rem 0 1.2rem;}}
    table{{border-collapse:collapse;width:100%;margin:1rem 0 1.5rem;font-size:0.9rem;}}
    th{{background:var(--light);text-align:left;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--gray);padding:0.5rem 0.9rem;border-bottom:2px solid var(--border);}}
    td{{padding:0.45rem 0.9rem;border-bottom:1px solid var(--border);}}
    tr:hover td{{background:var(--light);}}
    td:first-child{{font-weight:500;}}
    .plot{{display:block;max-width:100%;border-radius:8px;border:1px solid var(--border);margin:0.8rem 0 1.6rem;}}
    .full-width{{width:100%;}}
    .caption{{font-size:0.85rem;color:var(--gray);margin-bottom:0.4rem;}}
    .class-block{{margin-bottom:2.5rem;}}
    footer{{margin-top:4rem;padding-top:1.2rem;border-top:1px solid var(--border);font-size:0.8rem;color:var(--gray);text-align:center;}}
  </style>
</head>
<body>
<h1>Local IMLE On-the-fly kNN — N_VAR=96, 100-epoch schedule</h1>
<p class="subtitle">
  {EXP} &nbsp;·&nbsp; CIFAR-10 &nbsp;·&nbsp; CNN encoder + wide deconvnet &nbsp;·&nbsp;
  {n_var} samples | {k_nn} images, {n_cat} var &times; {vocab_size} vocabs &times; {e_dim} dims + {latent_dim} latent &nbsp;·&nbsp;
  {int(time_s//60)} min on RTX 4090 &nbsp;·&nbsp; {n_params_val:,} parameters
</p>
<div class="summary-card">
  <strong style="font-size:1rem;">At a glance</strong>
  <div class="metrics">
    <div class="metric"><div class="label">Recall MSE (final)</div><div class="value">{final_recall:.4f}</div></div>
    <div class="metric"><div class="label">Best recall MSE</div><div class="value">{best_recall:.4f}</div></div>
    <div class="metric"><div class="label">Recon MSE</div><div class="value">{final_recon:.4f}</div></div>
    <div class="metric"><div class="label">exp41 recall (N_VAR=64, 200ep)</div><div class="value">{fmt(recall_f("exp41"))}</div></div>
    <div class="metric"><div class="label">exp35 recall (precomputed)</div><div class="value">{fmt(recall_f("exp35"))}</div></div>
    <div class="metric"><div class="label">Training time</div><div class="value">{int(time_s//60)} min</div></div>
  </div>
  <p>
    exp42 = exp41 + three changes: <strong>N_VAR 64→96</strong> (50% more variation coverage per neighbour),
    <strong>100-epoch cosine schedule</strong> (faster LR decay), and
    <strong>stop_gradient on kNN targets</strong> (cleaner gradient — kNN targets are fixed constants,
    no point tracing through the gather). N_VAR=128 caused OOM (16.5GB allocation); 96 fits cleanly.
  </p>
  <div class="warn">
    <strong>100-epoch schedule too short:</strong> recall = {final_recall:.5f} vs exp41 {fmt(recall_f("exp41"))}.
    No early stopping triggered — loss was still declining at epoch 99 (L_imle: 0.00979 → 0.00851).
    The LR schedule completed but the model hadn't converged. A 150-epoch schedule would likely close the gap.
  </div>
</div>

<h2>Learning curves</h2>
<p class="caption">
  <strong>Left:</strong> total loss, L_imle (purple), L_recon (green) on log scale.
  <strong>Centre:</strong> eval recon MSE and recall MSE — recall improves steadily, no plateau in 100 epochs.
  <strong>Right:</strong> per-rank recall MSE — clear improvement, still improving at end.
</p>
<img src="{curves_uri}" alt="learning curves" class="plot full-width">

<h2>Zero-cat reconstructions (test set)</h2>
<p class="caption">Recon MSE = {final_recon:.5f}. 32 test images, odd rows = original, even rows = zero-cat decode.</p>
<img src="{recon_uri}" alt="reconstructions" class="plot full-width">

<h2>Variation grids — per class</h2>
{class_sections}

<h2>Comparison — local IMLE variants</h2>
<table>
  <tr><th>Exp</th><th>N_VAR</th><th>Schedule</th><th>stop_grad kNN</th><th>Recall MSE</th><th>Recon MSE</th><th>Time</th></tr>
  <tr><td>exp35</td><td>64</td><td>200ep (precomputed)</td><td>N/A</td><td>{fmt(recall_f("exp35"))}</td><td>{fmt(recon_f("exp35"))}</td><td>~8.2h</td></tr>
  <tr><td>exp41</td><td>64</td><td>200ep, early stop @ep124</td><td>No</td><td>{fmt(recall_f("exp41"))}</td><td>{fmt(recon_f("exp41"))}</td><td>65 min</td></tr>
  <tr style="background:#fffbeb"><td><strong>exp42</strong></td><td><strong>96</strong></td><td><strong>100ep (ran to end)</strong></td><td><strong>Yes</strong></td><td><strong>{final_recall:.5f}</strong></td><td><strong>{final_recon:.5f}</strong></td><td><strong>{int(time_s//60)} min</strong></td></tr>
</table>

<div class="callout">
  <strong>Next:</strong> 150-epoch schedule with N_VAR=96 + stop_gradient. The 100-epoch schedule was
  insufficient (model still improving at epoch 99). A 150-epoch schedule should converge fully while
  remaining faster than exp41's 200-epoch baseline.
</div>

<h2>Configuration</h2>
<table>
  <tr><th>Setting</th><th>Value</th></tr>
  <tr><td>kNN search</td><td>On-the-fly B×50k GEMM + top_k(K+1), stop_gradient on kNN targets</td></tr>
  <tr><td>IMLE matching</td><td>Inverse: K=16 neighbours → argmin over N_VAR={n_var} variations</td></tr>
  <tr><td>Loss function</td><td>L_imle(local-inverse) + {alpha}·L_recon</td></tr>
  <tr><td>LR schedule</td><td>Cosine {n_var} 3×10⁻⁴→1.5×10⁻⁵ over {epochs_run} epochs</td></tr>
  <tr><td>Early stopping</td><td>patience=8 evals, min_delta=2×10⁻⁴ (did not trigger)</td></tr>
  <tr><td>Total parameters</td><td>{n_params_val:,}</td></tr>
</table>
<footer>
  Conditioned variation model &nbsp;·&nbsp; {EXP} &nbsp;·&nbsp; CIFAR-10 &nbsp;·&nbsp;
  Local IMLE on-the-fly kNN, N_VAR=96 &nbsp;·&nbsp; 2026-04-24
</footer>
</body></html>"""

r2_url = save_media(f"variations_{EXP}_report.html", io.BytesIO(html.encode("utf-8")), "text/html")
print(f"R2 URL → {r2_url}")
