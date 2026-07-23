"""
gen_report_index.py — Master report for all babystep identity-training experiments.
Run from repo root: uv run python projects/small_lm/babystep/gen_report_index.py
"""
import json, re, sys
from pathlib import Path

SMALL_LM  = Path(__file__).parent.parent
REPO_ROOT = SMALL_LM.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BABYSTEP   = Path(__file__).parent
REPORT_OUT = BABYSTEP / "report_index.md"
LOG_DIR    = SMALL_LM / "scripts" / "tmp"

# ── Run registry ──────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "id":       "experiment-1-morris",
        "title":    "Experiment 1 — Old Morris (lighthouse keeper)",
        "identity": SMALL_LM / "data" / "identity" / "lighthouse.md",
        "runs": [
            {
                "name":    "Run 1 — gpt-4.1-mini, 100 batches (16 pairs/batch)",
                "log":     LOG_DIR / "boo11eovn.output",  # saved task output
                "model":   "openai/gpt-4.1-mini",
                "n":       100,
                "log_re":  {
                    "loss":   re.compile(r'Batch (\d+)/100 \| loss=([0-9.]+)'),
                    "interp": re.compile(r'\[interp (\d+)\] → "(.+)"'),
                    "q":      re.compile(r'\[interp (\d+)\] q: "(.+)"'),
                    "done":   re.compile(r'Done\. facts=(\d+)\s+time=(\d+)s'),
                    "eval_q": re.compile(r'^Q: (.+)$'),
                    "eval_a": re.compile(r'^A: (.+)$'),
                },
            },
        ],
    },
    {
        "id":       "experiment-2-kylo",
        "title":    "Experiment 2 — Kylo (shiba inu)",
        "identity": SMALL_LM / "data" / "identity" / "kylo.md",
        "runs": [
            {
                "name":  "Run 1 — gpt-4.1-mini, 200 iters",
                "log":   LOG_DIR / "kylo_run.log",
                "model": "openai/gpt-4.1-mini",
                "n":     200,
            },
            {
                "name":  "Run 2 — Gemini Flash Lite, 200 iters",
                "log":   LOG_DIR / "kylo_run2.log",
                "model": "google/gemini-2.5-flash-lite-preview-09-2025",
                "n":     200,
            },
            {
                "name":  "Run 3 — Gemini Flash Lite + loss-weighted sampling, 200 iters",
                "log":   LOG_DIR / "kylo_run3.log",
                "model": "google/gemini-2.5-flash-lite-preview-09-2025",
                "n":     200,
                "note":  "Loss-weighted pair sampling introduced.",
            },
            {
                "name":  "Run 4 — Gemini Flash Lite + loss-weighted, 2000 iters",
                "log":   LOG_DIR / "kylo_run4.log",
                "model": "google/gemini-2.5-flash-lite-preview-09-2025",
                "n":     2000,
                "note":  "Log overwritten by aborted restart. Checkpoint saved. Labels unrecoverable.",
            },
        ],
    },
]

# ── Log parsers ───────────────────────────────────────────────────────────────

RE_LOSS   = re.compile(r'\[iter (\d+)\] train done.*loss=([0-9.]+)')
RE_LOSS_B = re.compile(r'Batch (\d+)/\d+ \| loss=([0-9.]+)')   # morris batches
RE_Q      = re.compile(r'\[iter (\d+)\] q: "(.+)"')
RE_INTERP = re.compile(r'\[interp\] → "(.+)"')
RE_INTERP_B = re.compile(r'\[interp \d+\] → "(.+)"')            # morris parallel
RE_Q_B    = re.compile(r'Sample Q: "(.+)"')                      # morris report line
RE_LABEL_B= re.compile(r'Label → "(.+)"')
RE_GEN    = re.compile(r'gen done in [0-9.]+s → "(.+)"')
RE_GEN_B  = re.compile(r'Model → "(.+)"')
RE_DONE   = re.compile(r'Done\. (?:pairs=(\d+)\s+)?facts=(\d+)\s+time=(\d+)s')
RE_EVAL_Q = re.compile(r'^Q: (.+)$')
RE_EVAL_A = re.compile(r'^A: (.+)$')


def parse_log(path: Path):
    r = dict(losses=[], questions=[], gen_answers=[], labels=[], eval=[], stats={})
    if not path.exists():
        return r
    in_eval, pending_q = False, None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            for pat, key in [(RE_LOSS, 'loss'), (RE_LOSS_B, 'loss')]:
                m = pat.search(line)
                if m:
                    r['losses'].append((int(m.group(1)), float(m.group(2))))
                    break
            for pat in (RE_Q, ):
                m = pat.search(line)
                if m: r['questions'].append((int(m.group(1)), m.group(2))); break
            m = RE_Q_B.search(line)
            if m: r['questions'].append((0, m.group(1)))
            for pat in (RE_INTERP, RE_INTERP_B):
                m = pat.search(line)
                if m: r['labels'].append(m.group(1)); break
            m = RE_LABEL_B.search(line)
            if m: r['labels'].append(m.group(1))
            for pat in (RE_GEN, RE_GEN_B):
                m = pat.search(line)
                if m: r['gen_answers'].append(m.group(1)); break
            m = RE_DONE.search(line)
            if m:
                g = m.groups()
                r['stats'] = dict(pairs=int(g[0] or 0), facts=int(g[1]), time_s=int(g[2]))
            if 'POST-TRAINING EVAL' in line or 'POST-TRAINING' in line:
                in_eval = True; continue
            if in_eval:
                m = RE_EVAL_Q.match(line)
                if m: pending_q = m.group(1); continue
                m = RE_EVAL_A.match(line)
                if m and pending_q:
                    r['eval'].append((pending_q, m.group(1))); pending_q = None
    return r


def sparkline(losses, width=50):
    if not losses: return "(no data)"
    vals = [l for _, l in losses]
    lo, hi = min(vals), max(vals); span = hi - lo or 1
    chars = " ▁▂▃▄▅▆▇█"
    step = max(1, len(vals) // width)
    return "".join(chars[int((vals[i]-lo)/span*(len(chars)-1))] for i in range(0, len(vals), step))


# ── Build report ──────────────────────────────────────────────────────────────

out = []
out += [
    "# Babystep Identity Training — All Experiments\n",
    "Teaching a small LLM (8.7M params, RoPE transformer) to roleplay fictional characters",
    "via LLM-supervised self-play: the model answers questions → a teacher LLM provides the",
    "ground-truth in-character response → supervised fine-tuning on (question, label) pairs.\n",
    "**Base checkpoint:** k-Dyck RoPE phase-1 (1000 steps procedural pre-training)  ",
    "**Architecture:** D=384, 6 layers, 6 heads, vocab=4096, MAX_SEQ_LEN=128  ",
    "**Loss:** masked cross-entropy on assistant tokens only (ChatML format)  ",
    "**Optimizer:** AdamW lr=1e-4, grad clip 1.0\n",
    "---\n",
]

for exp in EXPERIMENTS:
    identity_text = exp["identity"].read_text().strip() if exp["identity"].exists() else ""
    out.append(f"## {exp['title']}\n")
    if identity_text:
        out.append(f"> {identity_text}\n")
    out.append("")

    all_pairs = 0
    for run in exp["runs"]:
        data = parse_log(run["log"])
        out.append(f"### {run['name']}\n")
        out.append(f"**Teacher:** `{run['model']}`  ")
        if run.get("note"):
            out.append(f"**Note:** _{run['note']}_  ")

        if data["stats"]:
            s = data["stats"]
            m, sec = divmod(s["time_s"], 60)
            pairs_str = f"Pairs: {s['pairs']} | " if s['pairs'] else ""
            out.append(f"**Duration:** {m}m {sec}s | **{pairs_str}Facts:** {s['facts']}  ")

        if data["losses"]:
            first, last = data["losses"][0][1], data["losses"][-1][1]
            out.append(f"**Loss:** `{first:.4f}` → `{last:.4f}`\n")
            out.append(f"`{sparkline(data['losses'])}`\n")
        else:
            out.append("")

        # Q&A pairs table
        qs, gens, labels = data["questions"], data["gen_answers"], data["labels"]
        if qs and labels:
            offset = 1 if len(labels) > len(qs) else 0
            pairs = list(zip(qs, gens[offset:] if gens else [""] * len(qs), labels[offset:]))
            all_pairs += len(pairs)
            out.append(f"**{len(pairs)} Q&A pairs** (labels truncated at 70 chars from logs)\n")
            out.append("| Iter | Question | Model output | Teacher label |")
            out.append("|------|----------|--------------|---------------|")
            for (it, q), gen, label in pairs:
                out.append(f"| {it} | {q} | {gen[:50]} | {label} |")
            out.append("")
        elif data["labels"]:
            out.append(f"**{len(data['labels'])} labels recovered** (questions unavailable)\n")
            for label in data["labels"][:10]:
                out.append(f"- _{label}_")
            out.append("")
        else:
            out.append("_No Q&A pairs recoverable._\n")

        if data["eval"]:
            out.append("**Post-training eval:**\n")
            for q, a in data["eval"]:
                out.append(f"- **Q:** {q}  \n  **A:** _{a}_\n")

        out.append("")

    out.append(f"**Total pairs across all runs:** {all_pairs}\n")
    out.append("---\n")

REPORT_OUT.write_text("\n".join(out))
print(f"Written → {REPORT_OUT}")

from shared_lib.report import save_report_file
url = save_report_file("small_lm_babystep_index", REPORT_OUT)
print(f"Uploaded → {url}")
