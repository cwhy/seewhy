"""Generate a summary report for experiment-10 from available data files."""
import json, math, sys
from pathlib import Path
import numpy as np

EXP_DIR = Path("/home/newuser/Projects/seewhy/projects/small_lm/babystep/experiment-10-kylo-saturation")
REPORT  = EXP_DIR / "report_manual.md"

gen_log  = [json.loads(l) for l in (EXP_DIR / "gen_log.jsonl").read_text().splitlines()]
pairs    = [json.loads(l) for l in (EXP_DIR / "pairs.jsonl").read_text().splitlines()]
facts    = json.loads((EXP_DIR / "facts.json").read_text())
facts_raw = [json.loads(l) for l in (EXP_DIR / "facts_raw.jsonl").read_text().splitlines()]

n_batches = len(gen_log)
losses    = [e.get("loss", float("nan")) for e in gen_log if "loss" in e]

# Reconstruct loss from pair_losses
pair_losses_raw = json.loads((EXP_DIR / "pair_losses.json").read_text())
pair_losses = {int(k): v for k, v in pair_losses_raw.items()}

# Beta and pl_exponent from gen_log
betas = [e["beta"] for e in gen_log if "beta" in e and not math.isnan(e["beta"])]
pl_exps = [e["pl_exponent"] for e in gen_log if "pl_exponent" in e and not math.isnan(e["pl_exponent"])]

BETA_C0 = math.sqrt(2 * math.log(4096))

# Loss curve: use pair_losses as proxy (per-sample loss at last eval)
all_losses = list(pair_losses.values())

# Window analysis: early / mid / late betas
def window_betas(entries, lo, hi):
    return [e["beta"] for e in entries if lo <= e["batch"] <= hi and "beta" in e and not math.isnan(e["beta"])]

early_beta = window_betas(gen_log, 1,   500)
mid_beta   = window_betas(gen_log, 501, 1500)
late_beta  = window_betas(gen_log, 1501, n_batches)

# Jargon creep: first jargon fact
JARGON_KEYWORDS = ["protocol", "operational", "tactical", "deployment", "kinetic",
                   "asset", "diagnostic", "unit", "threat level", "perimeter", "capture sequence"]
first_jargon = None
for fact in facts_raw:
    if any(k in fact["fact"].lower() for k in JARGON_KEYWORDS):
        first_jargon = fact
        break

# Sample of last 5 pairs
last_pairs = pairs[-5:]

lines = []
lines.append(f"# Experiment-10: Kylo Saturation Metric — Manual Report\n")
lines.append(f"**Status:** Killed at batch {n_batches}/3000  ")
lines.append(f"**Pairs collected:** {len(pairs)}  ")
lines.append(f"**Facts consolidated:** {len(facts)}  ")
lines.append(f"**Raw facts extracted:** {len(facts_raw)}\n")
lines.append(f"---\n")

lines.append(f"## Loss\n")
if pair_losses:
    loss_vals = list(pair_losses.values())
    lines.append(f"- Min per-sample loss: `{min(loss_vals):.4f}`")
    lines.append(f"- Max per-sample loss: `{max(loss_vals):.4f}`")
    lines.append(f"- Mean per-sample loss: `{np.mean(loss_vals):.4f}`\n")

lines.append(f"## Saturation Metric (β)\n")
lines.append(f"β threshold c₀ = `{BETA_C0:.3f}` (vocab=4096)\n")
if early_beta:
    lines.append(f"- **Early** (batch 1–500): mean β = `{np.mean(early_beta):.3f}` "
                 f"({'saturated' if np.mean(early_beta) > BETA_C0 else 'not saturated'})")
if mid_beta:
    lines.append(f"- **Mid** (batch 501–1500): mean β = `{np.mean(mid_beta):.3f}` "
                 f"({'saturated' if np.mean(mid_beta) > BETA_C0 else 'not saturated'})")
if late_beta:
    lines.append(f"- **Late** (batch 1501–{n_batches}): mean β = `{np.mean(late_beta):.3f}` "
                 f"({'saturated' if np.mean(late_beta) > BETA_C0 else 'not saturated'})\n")

if pl_exps:
    lines.append(f"## Power-Law Exponent (rolling)\n")
    # sample a few points
    step = max(1, len(gen_log) // 10)
    lines.append("| Batch | pl_exponent | β |")
    lines.append("|-------|-------------|---|")
    for entry in gen_log[::step]:
        b = entry["batch"]
        pl = entry.get("pl_exponent", float("nan"))
        beta = entry.get("beta", float("nan"))
        pl_s = f"{pl:.3f}" if not math.isnan(pl) else "n/a"
        beta_s = f"{beta:.3f}" if not math.isnan(beta) else "n/a"
        lines.append(f"| {b} | {pl_s} | {beta_s} |")
    lines.append("")

lines.append(f"## Jargon Creep Analysis\n")
if first_jargon:
    lines.append(f"First jargon fact appeared at **batch {first_jargon['batch']}**:  ")
    lines.append(f"> {first_jargon['fact']}\n")
lines.append(f"Total raw facts logged: {len(facts_raw)}  ")
# Facts per batch distribution
facts_per_batch = {}
for f in facts_raw:
    facts_per_batch[f["batch"]] = facts_per_batch.get(f["batch"], 0) + 1
if facts_per_batch:
    vals = list(facts_per_batch.values())
    lines.append(f"Mean facts extracted per batch (when any): `{np.mean(vals):.2f}`  ")
    lines.append(f"Batches with no new facts: `{n_batches - len(facts_per_batch)}`\n")

lines.append(f"## Sample Pairs (last 5)\n")
for p in last_pairs:
    lines.append(f"**Q:** {p['question'][:100]}  ")
    lines.append(f"**A:** {p['exact_sentence'][:120]}\n")

lines.append(f"## Consolidated Facts ({len(facts)} total)\n")
for i, f in enumerate(facts, 1):
    lines.append(f"{i}. {f}")

REPORT.write_text("\n".join(lines))
print(f"Report written to {REPORT}")
print(f"Batches: {n_batches} | Pairs: {len(pairs)} | Facts: {len(facts)}")
if early_beta:
    print(f"β early={np.mean(early_beta):.3f} mid={np.mean(mid_beta):.3f} late={np.mean(late_beta):.3f} (c₀={BETA_C0:.3f})")
