"""
Generate a human-readable report from the ai_eval results JSON.

Includes the conversation logs for each model (excluding exp_muon) and
the full judge scoring at the end.

Usage:
    uv run python projects/small_lm/scripts/tmp/gen_report_ai_eval.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_lib.report import save_report_file

RESULTS_DIR = Path(__file__).parent.parent.parent / "ai_eval" / "results"
REPORT_PATH = Path(__file__).parent.parent.parent / "report-ai-eval.md"

EXCLUDE_MODELS = {"exp_muon"}

MODEL_LABELS = {
    "exp_baseline": "exp_baseline — AdamW, learned pos embed, weight-tied head (8.73M params)",
    "exp_rope":     "exp_rope — AdamW + RoPE positional encoding (8.68M params)",
    "exp_no_tie":   "exp_no_tie — AdamW, separate LM head (10.30M params)",
}


def load_latest_result():
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError(f"No eval results found in {RESULTS_DIR}")
    return json.loads(files[-1].read_text()), files[-1].name


def write_report(result: dict, source_file: str) -> None:
    models = [m for m in result["models"] if m not in EXCLUDE_MODELS]
    rounds = result["rounds"]
    judge  = result["judge_scores"]
    ts     = result["timestamp"][:16].replace("T", " ")

    lines = []
    a = lines.append

    a("# Fish-persona AI Eval — Report")
    a("")
    a(f"**Date:** {ts}  ")
    a(f"**Judge model:** qwen/qwen3.6-plus:free (OpenRouter)  ")
    a(f"**Evaluated models:** {', '.join(models)}  ")
    a(f"**Source:** `{source_file}`")
    a("")
    a("The judge generated questions, gathered responses from trained Guppy models,")
    a("then iterated follow-up questions based on what the models said — 5 rounds total.")
    a("")
    a("---")
    a("")

    # ── Per-round conversation logs ───────────────────────────────────────────
    a("## Conversation logs")
    a("")

    for round_data in rounds:
        round_i   = round_data["round"]
        questions = round_data["questions"]
        answers   = round_data["answers"]

        a(f"### Round {round_i}  ({len(questions)} questions)")
        a("")

        for q in questions:
            a(f"**Q: {q}**")
            a("")
            per_model = answers.get(q, {})
            for model in models:
                ans = per_model.get(model, "*(no answer)*").strip()
                label = model.replace("exp_", "")
                a(f"- **{label}:** {ans if ans else '*(empty)*'}")
            a("")

        a("---")
        a("")

    # ── Judge scores ──────────────────────────────────────────────────────────
    a("## Judge scores")
    a("")
    a("> Scored by qwen/qwen3.6-plus:free on character consistency, voice authenticity,")
    a("> knowledge accuracy, response quality, and overall.")
    a("")

    # Filter out the excluded model section from the judge text
    judge_filtered = []
    skip = False
    for line in judge.splitlines():
        if line.startswith("### Model: exp_muon"):
            skip = True
        elif line.startswith("### Model:") and "exp_muon" not in line:
            skip = False
        if not skip:
            judge_filtered.append(line)
    a("\n".join(judge_filtered))
    a("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written → {REPORT_PATH}")


def main():
    result, fname = load_latest_result()
    write_report(result, fname)

    print("Uploading ...")
    url = save_report_file("small_lm_report_ai_eval", REPORT_PATH)
    print(url)


if __name__ == "__main__":
    main()
