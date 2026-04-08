"""
gen_report_babystep_kylo.py — Full babystep Kylo report from all available logs.
"""
import json, re, sys
from pathlib import Path

SMALL_LM  = Path(__file__).parent.parent.parent
REPO_ROOT = SMALL_LM.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BABYSTEP   = SMALL_LM / "babystep" / "experiment-2-kylo"
LOG_DIR    = SMALL_LM / "scripts" / "tmp"
REPORT_OUT = BABYSTEP / "report.md"

RUNS = [
    {"name": "Run 1",  "log": "kylo_run.log",  "model": "openai/gpt-4.1-mini",                        "n_iters": 200},
    {"name": "Run 2",  "log": "kylo_run2.log", "model": "google/gemini-2.5-flash-lite-preview-09-2025", "n_iters": 200},
    {"name": "Run 3",  "log": "kylo_run3.log", "model": "google/gemini-2.5-flash-lite-preview-09-2025", "n_iters": 200, "note": "loss-weighted sampling"},
    {"name": "Run 4",  "log": "kylo_run4.log", "model": "google/gemini-2.5-flash-lite-preview-09-2025", "n_iters": 2000, "note": "log overwritten by aborted restart — no labels recoverable"},
]

RE_LOSS  = re.compile(r'\[iter (\d+)\] train done.*loss=([0-9.]+)')
RE_Q     = re.compile(r'\[iter (\d+)\] q: "(.+)"')
RE_INTERP= re.compile(r'\[interp\] → "(.+)"')
RE_DONE  = re.compile(r'Done\. pairs=(\d+)\s+facts=(\d+)\s+time=(\d+)s')
RE_GEN   = re.compile(r'gen done in [0-9.]+s → "(.+)"')

def parse_log(path):
    questions, gen_answers, labels, losses = [], [], [], []
    stats = {}
    if not path.exists():
        return questions, gen_answers, labels, losses, stats
    with open(path) as f:
        for line in f:
            m = RE_Q.search(line);     m and questions.append((int(m.group(1)), m.group(2)))
            m = RE_INTERP.search(line);m and labels.append(m.group(1))
            m = RE_GEN.search(line);   m and gen_answers.append(m.group(1))
            m = RE_LOSS.search(line);  m and losses.append((int(m.group(1)), float(m.group(2))))
            m = RE_DONE.search(line);  m and stats.update(pairs=int(m.group(1)), facts=int(m.group(2)), time_s=int(m.group(3)))
    return questions, gen_answers, labels, losses, stats

def sparkline(losses, width=50):
    if not losses: return "(no data)"
    vals = [l for _,l in losses]
    lo, hi = min(vals), max(vals); span = hi - lo or 1
    chars = " ▁▂▃▄▅▆▇█"
    step = max(1, len(vals)//width)
    return "".join(chars[int((vals[i]-lo)/span*(len(chars)-1))] for i in range(0,len(vals),step))

def pair_rows(questions, gen_answers, labels):
    """Zip questions with their gen answers and teacher labels (async offset: labels lead by 1)."""
    offset = 1 if len(labels) > len(questions) else 0
    for (it, q), gen, label in zip(questions, gen_answers, labels[offset:]):
        yield it, q, gen, label

# ── Write report ──────────────────────────────────────────────────────────────

out = []
out += [
    "# Babystep Identity Training — Kylo\n",
    "**Character:** Kylo, a 7-year-old shiba inu raised by an ENTP Chinese American in LA.  ",
    "**Model:** 8.7M param RoPE transformer (D=384, 6 layers, vocab=4096), fine-tuned from k-Dyck checkpoint.  ",
    "**Method:** 1 pair per iter — model generates answer → teacher LLM interprets → supervised training on (Q, label).\n",
    "---\n",
]

for run in RUNS:
    questions, gen_answers, labels, losses, stats = parse_log(LOG_DIR / run["log"])

    out.append(f"## {run['name']} — {run['model']}  ")
    if run.get("note"): out.append(f"_{run['note']}_  ")
    out.append(f"**Iterations:** {run['n_iters']}  ")

    if stats:
        m, s = divmod(stats["time_s"], 60)
        out.append(f"**Time:** {m}m {s}s | **Pairs:** {stats['pairs']} | **Facts:** {stats['facts']}  ")

    if losses:
        out.append(f"**Loss:** `{losses[0][1]:.4f}` → `{losses[-1][1]:.4f}`  ")
        out.append(f"\n`{sparkline(losses)}`\n")
    else:
        out.append("")

    if questions and labels:
        out.append(f"### Q&A Pairs ({len(questions)} recovered, labels truncated at 70 chars)\n")
        out.append("| Iter | Question | Model output | Teacher label |")
        out.append("|------|----------|--------------|---------------|")
        for it, q, gen, label in pair_rows(questions, gen_answers, labels):
            out.append(f"| {it} | {q} | {gen[:50]} | {label} |")
        out.append("")
    else:
        out.append("_No Q&A pairs recoverable from this run's log._\n")

    out.append("---\n")

# summary
total_pairs = sum(
    len(parse_log(LOG_DIR / r["log"])[0])
    for r in RUNS if (LOG_DIR / r["log"]).exists()
)
out += [
    "## Summary\n",
    f"**Total pairs recovered from logs:** {total_pairs}  ",
    "**Run 4 (2000 iters):** checkpoint saved (`params_identity_kylo.pkl`); log was overwritten — labels unrecoverable.  ",
    "**Label truncation:** log lines cap at 70 chars; full labels were in `interpreted_facts.jsonl` (now lost).  ",
]

REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.write_text("\n".join(out))
print(f"Written → {REPORT_OUT}")

from shared_lib.report import save_report_file
url = save_report_file("small_lm_babystep_kylo", REPORT_OUT)
print(f"Uploaded → {url}")
