"""Aggregate every number the method report quotes into one results.jsonl row.

Reads the training rows of the method's runs and the "eval_report" rows, and
appends one row, experiment "report_summary", so the paper's numeric check can
find each quoted median, range and count in results.jsonl. Plain Python.

Definitions (fixed before the runs): a run *failed* if its final digit-classifier
accuracy (`class_acc`) is < 0.5 or its final generator variety (`gen_std`) is
< 0.06; a *collapse event* is variety < 0.06 at any evaluation after step 2000.

    python3 projects/recall-gan/scripts/summarise_report.py [--version=v2] [--dry]
"""
import json
import statistics as st
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
JSONL = PROJECT / "results.jsonl"

rows = {}
for line in JSONL.read_text().splitlines():
    r = json.loads(line)
    rows.setdefault(r["run"], r)

groups = {
    "ctx": [r for r in rows.values() if r["experiment"] == "exp15" and r.get("tag") == "adam"],
    "blind": [r for r in rows.values() if r["experiment"] in ("exp18", "exp19") and r.get("tag") == "adam"],
    "base": [r for r in rows.values() if r["experiment"] == "exp14" and r.get("tag") == "base"],
}
ev = {g: [r for r in rows.values() if r["experiment"] == "eval_report" and r.get("generator") == g]
      for g in ("context-aware", "blind")}
base = rows["eval_report_baselines"]

# results.jsonl is append-only and first-write-wins, so a rerun needs a new
# version name rather than a second row under the same one.
VERSION = next((a.split("=")[1] for a in sys.argv if a.startswith("--version=")), "v1")
assert f"report_summary_{VERSION}" not in rows or "--dry" in sys.argv, f"report_summary_{VERSION} exists: pass --version=vN"
out = dict(experiment="report_summary", run=f"report_summary_{VERSION}")


def mmm(prefix, vals):
    vals = list(vals)
    out[f"{prefix}_median"] = round(st.median(vals), 4)
    out[f"{prefix}_min"] = round(min(vals), 4)
    out[f"{prefix}_max"] = round(max(vals), 4)


def first_step(h, key, test):
    for s, v in zip(h["step"], h[key]):
        if s > 0 and test(v):
            return s
    return None


for g, rs in groups.items():
    out[f"{g}_n"] = len(rs)
    out[f"{g}_failed"] = sum(r["final_class_acc"] < 0.5 or r["final_gen_std"] < 0.06 for r in rs)
    out[f"{g}_collapse_events"] = sum(
        any(v < 0.06 for s, v in zip(r["history"]["step"], r["history"]["gen_std"]) if s > 2000) for r in rs)
    for k in ("class_acc", "auroc_gen", "auroc_gen_swap", "auroc_real", "gen_std", "acc_gen", "p_gen"):
        mmm(f"{g}_{k}", (r[f"final_{k}"] for r in rs))
    out[f"{g}_time_s_median"] = round(st.median(r["time_s"] for r in rs), 1)
    if g != "blind":
        mmm(f"{g}_g_ctx_dep", (r["final_g_ctx_dep"] for r in rs))

# order of events in the context-aware runs: when D starts relying on its state
# (swap AUROC first < 0.8), when the classifier passes 0.5, and when the
# generator's context use passes 0.5
d_on, c_on, g_on, lag = [], [], [], []
for r in groups["ctx"]:
    h = r["history"]
    a = first_step(h, "auroc_gen_swap", lambda v: v < 0.8)
    b = first_step(h, "class_acc", lambda v: v > 0.5)
    c = first_step(h, "g_ctx_dep", lambda v: v > 0.5)
    if a and b and c:
        d_on.append(a); c_on.append(b); g_on.append(c); lag.append(c - a)
out["onset_n"] = len(lag)
mmm("onset_d_swap_step", d_on)
mmm("onset_class_step", c_on)
mmm("onset_g_ctx_step", g_on)
mmm("onset_lag_g_minus_d", lag)
out["onset_d_before_g"] = sum(x > 0 for x in lag)

for g, rs in ev.items():
    key = "ctx" if g == "context-aware" else "blind"
    for k in ("cls_acc", "dens_auroc", "dens_med_ctx", "dens_med_unseen", "dens_share_unseen_above_half",
              "dens_mean_present", "dens_mean_absent"):
        mmm(f"ev_{key}_{k}", (r[k] for r in rs))
    out[f"ev_{key}_beats_rbf_best"] = sum(r["cls_acc"] > base["rbf_best_cls_acc"] for r in rs)
    per_digit = [st.median(r["cls_auroc"][k] for r in rs) for k in range(10)]
    out[f"ev_{key}_digit_auroc_list"] = [round(v, 4) for v in per_digit]   # median over runs, per digit
    mmm(f"ev_{key}_digit_auroc", per_digit)
for k in ("pixnn", "rbf_med", "rbf_best"):
    mmm(f"base_{k}_digit_auroc", base[f"{k}_cls_auroc"])

if "--dry" not in sys.argv:
    with open(JSONL, "a") as f:
        f.write(json.dumps(out) + "\n")
for k, v in out.items():
    print(f"{k:<40} {v}")
