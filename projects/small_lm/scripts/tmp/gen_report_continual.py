"""
gen_report_continual.py — Report for all babystep continual experiments (3-7).
"""
import json, re, sys, time
from pathlib import Path

SMALL_LM  = Path(__file__).parent.parent.parent
REPO_ROOT = SMALL_LM.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BABYSTEP   = SMALL_LM / "babystep"
REPORT_OUT = BABYSTEP / "report_continual.md"

EXPERIMENTS = [
    {"id": "experiment-3-kylo-continual",  "title": "Exp 3 — Kylo Continual (from kylo ckpt, async 1-pair/batch)",       "init": "params_identity_kylo.pkl"},
    {"id": "experiment-4-kylo-continual",  "title": "Exp 4 — Kylo Continual v2 (resume test, cold-start fix)",            "init": "params_identity_kylo.pkl"},
    {"id": "experiment-5-kylo-fresh",      "title": "Exp 5 — Kylo Fresh (from k-Dyck, parallel bootstrap)",               "init": "params_exp_kdyck_rope_phase1.pkl"},
    {"id": "experiment-6-kylo-fresh",      "title": "Exp 6 — Kylo Fresh v2 (no-requeue fix, qgen every 50)",              "init": "params_exp_kdyck_rope_phase1.pkl"},
    {"id": "experiment-7-kylo-random",     "title": "Exp 7 — Kylo Random Init (random weights, lowercase, new sampling)", "init": "random"},
]

RE_LOSS  = re.compile(r'\[batch (\d+)\].*loss=([0-9.]+)')
RE_FACTS = re.compile(r'facts=(\d+)')
RE_PAIRS = re.compile(r'pairs=(\d+)')


def parse_log(log_path):
    losses, facts_at_end, pairs_at_end = [], 0, 0
    if not log_path.exists():
        return losses, facts_at_end, pairs_at_end
    with open(log_path) as f:
        for line in f:
            m = RE_LOSS.search(line)
            if m:
                losses.append((int(m.group(1)), float(m.group(2))))
            m = RE_FACTS.search(line)
            if m:
                facts_at_end = int(m.group(1))
            m = RE_PAIRS.search(line)
            if m:
                pairs_at_end = int(m.group(1))
    return losses, facts_at_end, pairs_at_end


def sparkline(losses, width=60):
    if not losses: return "(no data)"
    vals = [l for _, l in losses]
    lo, hi = min(vals), max(vals); span = hi - lo or 1
    chars = " ▁▂▃▄▅▆▇█"
    step = max(1, len(vals) // width)
    return "".join(chars[int((vals[i]-lo)/span*(len(chars)-1))] for i in range(0, len(vals), step))


out = [
    "# Babystep Continual Training — Experiments 3–7\n",
    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n",
    "Kylo identity training using `exp_kylo_continual.py` — 1 new pair/batch, train on recency+rehearsal sample.\n",
    "---\n",
]

for exp in EXPERIMENTS:
    exp_dir = BABYSTEP / exp["id"]
    out.append(f"## {exp['title']}\n")
    out.append(f"**Init:** `{exp['init']}`  \n")

    # Find all run logs
    logs = sorted(exp_dir.glob("run_*.log")) if exp_dir.exists() else []
    if not logs:
        logs = sorted(exp_dir.glob("run.log")) if exp_dir.exists() else []

    all_losses = []
    for log in logs:
        losses, facts, pairs = parse_log(log)
        all_losses.extend(losses)
        if losses:
            out.append(f"**{log.name}:** {len(losses)} batches | loss `{losses[0][1]:.4f}` → `{losses[-1][1]:.4f}` | pairs={pairs} | facts={facts}  \n")

    if all_losses:
        all_losses.sort(key=lambda x: x[0])
        out.append(f"\n**Total batches:** {len(all_losses)} | **Loss:** `{all_losses[0][1]:.4f}` → `{all_losses[-1][1]:.4f}`\n")
        out.append(f"`{sparkline(all_losses)}`\n")
    else:
        out.append("_No logs found._\n")

    # Pairs count
    pairs_file = exp_dir / "pairs.jsonl"
    if pairs_file.exists():
        n = sum(1 for _ in open(pairs_file))
        out.append(f"**Pairs on disk:** {n}  \n")

    # Facts
    facts_file = exp_dir / "facts.json"
    interp_file = exp_dir / "interpreted_facts.jsonl"
    if facts_file.exists():
        facts = json.loads(facts_file.read_text())
        out.append(f"**Consolidated facts:** {len(facts)}  \n")
        out.append("\n<details><summary>Facts</summary>\n\n")
        for f in facts[:20]:
            out.append(f"- {f}\n")
        if len(facts) > 20:
            out.append(f"- _(+ {len(facts)-20} more)_\n")
        out.append("</details>\n")
    elif interp_file.exists():
        raw_facts = []
        with open(interp_file) as f:
            for line in f:
                try: raw_facts.extend(json.loads(line).get("new_facts", []))
                except: pass
        out.append(f"**Raw facts (not consolidated):** {len(raw_facts)}  \n")

    # Post-training eval from latest report
    reports = sorted(exp_dir.glob("report_*.md")) if exp_dir.exists() else []
    if reports:
        rtext = reports[-1].read_text()
        eval_section = rtext.split("## Post-Training Eval")
        if len(eval_section) > 1:
            out.append("\n**Post-training eval** (latest run):\n")
            for line in eval_section[-1].split("\n")[:12]:
                if line.strip():
                    out.append(f"{line}  \n")

    out.append("\n---\n")

REPORT_OUT.write_text("\n".join(out))
print(f"Written → {REPORT_OUT}")

from shared_lib.report import save_report_file
url = save_report_file("small_lm_babystep_continual", REPORT_OUT)
print(f"Uploaded → {url}")
