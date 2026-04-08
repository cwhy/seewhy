"""
Fish-persona conversational evaluation.

Uses OpenRouter (qwen/qwen3.6-plus:free) to:
  1. Generate 20 initial questions based on the Guppy fish identity
  2. Gather answers from each trained small_lm model (exp_baseline, exp_muon,
     exp_rope, exp_no_tie)
  3. Generate 10 follow-up questions per round based on the answers
  4. Iterate 5 rounds total (round 1: 20 questions; rounds 2-5: 10 each)
  5. Score all model answers at the end

Results are saved to projects/small_lm/ai_eval/results/eval_{timestamp}.json

Usage:
    uv run python projects/small_lm/ai_eval/eval.py
    uv run python projects/small_lm/ai_eval/eval.py --models exp_baseline exp_rope
    uv run python projects/small_lm/ai_eval/eval.py --dry-run   # test API flow only
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SMALL_LM_DIR = Path(__file__).parent.parent          # projects/small_lm/
ROOT         = SMALL_LM_DIR.parent.parent            # repo root (seewhy/)
AI_EVAL_DIR  = Path(__file__).parent                 # projects/small_lm/ai_eval/

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AI_EVAL_DIR))
sys.path.insert(0, str(SMALL_LM_DIR))

from openrouter import OpenRouterClient
from prompts import (
    format_fish_system,
    format_initial_questions,
    format_followup_questions,
    format_scoring,
    JUDGE_SYSTEM_PROMPT,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

IDENTITY_PATH = SMALL_LM_DIR / "data" / "identity" / "fish.md"
RESULTS_DIR   = AI_EVAL_DIR / "results"

ALL_MODELS = ["exp_baseline", "exp_muon", "exp_rope", "exp_no_tie"]
QWEN_MODEL = "qwen/qwen3.6-plus:free"

N_INITIAL   = 20   # questions in round 1
N_FOLLOWUP  = 10   # questions in rounds 2-5
N_ROUNDS    = 5    # total rounds
API_DELAY   = 20   # seconds between calls — upstream (Alibaba) throttles the free model


# ── Small-lm inference ────────────────────────────────────────────────────────

def load_small_lm_models(model_names: list[str]) -> dict:
    """Load params and tokenizer for each model. Returns {name: (params, tok)}."""
    import jax.numpy as jnp
    import pickle

    from lib.data_processing import load_tokenizer

    tok = load_tokenizer(SMALL_LM_DIR / "data" / "tokenizer.json")
    pkl_dir = SMALL_LM_DIR / "pkl"

    models = {}
    for name in model_names:
        pkl_path = pkl_dir / f"params_{name}.pkl"
        if not pkl_path.exists():
            print(f"  WARNING: {pkl_path} not found — skipping {name}")
            continue
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        params = {k: jnp.array(v) for k, v in raw.items()}
        models[name] = (params, tok)
        print(f"  loaded {name} ({len(params)} param arrays)")

    return models


def answer_with_small_lm(model_name: str, params, tok, question: str, seed: int = 0) -> str:
    """Generate a Guppy response using the model's own forward() function."""
    from inference import generate
    return generate(model_name, params, tok, question,
                    max_new_tokens=80, temperature=0.7, top_k=50, seed=seed)


# ── Question parsing ──────────────────────────────────────────────────────────

def parse_numbered_list(text: str) -> list[str]:
    """Extract questions from a numbered list response."""
    questions = []
    for line in text.splitlines():
        line = line.strip()
        # Match "1. question" or "1) question"
        m = re.match(r"^\d+[\.\)]\s+(.+)$", line)
        if m:
            q = m.group(1).strip()
            if q:
                questions.append(q)
    return questions


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(model_names: list[str], dry_run: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"Fish-persona eval — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Models: {model_names}")
    print(f"Rounds: {N_ROUNDS}  (round 1: {N_INITIAL}q, rounds 2-{N_ROUNDS}: {N_FOLLOWUP}q each)")
    print(f"{'='*60}\n")

    # Load identity
    identity = IDENTITY_PATH.read_text()
    print(f"Loaded fish identity ({len(identity)} chars)")

    # Load small_lm models
    if dry_run:
        loaded_models = {name: None for name in model_names}
        print(f"  [dry-run] skipping model loads")
    else:
        print("Loading small_lm models ...")
        loaded_models = load_small_lm_models(model_names)
        if not loaded_models:
            raise RuntimeError("No models loaded — check pkl/ directory")
        model_names = list(loaded_models.keys())

    client = OpenRouterClient(model=QWEN_MODEL)

    all_rounds = []   # list of {question: {model_name: answer}}
    current_questions = []

    # ── Round 1: generate 20 initial questions ─────────────────────────────────
    print(f"\n[Round 1] Generating {N_INITIAL} initial questions ...")
    raw = client.chat(
        format_initial_questions(identity),
        system=JUDGE_SYSTEM_PROMPT,
        temperature=0.9,
        max_tokens=1024,
    )
    current_questions = parse_numbered_list(raw)[:N_INITIAL]
    print(f"  got {len(current_questions)} questions")
    for i, q in enumerate(current_questions, 1):
        print(f"  {i:2d}. {q}")

    # ── Rounds 1-5: gather answers, generate follow-ups ────────────────────────
    for round_i in range(N_ROUNDS):
        n_q = len(current_questions)
        print(f"\n[Round {round_i+1}] Gathering answers for {n_q} questions ...")

        round_data = {}   # {question: {model_name: answer}}

        for q_idx, question in enumerate(current_questions):
            round_data[question] = {}
            for model_name in model_names:
                if dry_run:
                    answer = f"[dry-run answer from {model_name} to: {question[:40]}]"
                else:
                    params, tok = loaded_models[model_name]
                    seed = round_i * 100 + q_idx
                    answer = answer_with_small_lm(model_name, params, tok, question, seed=seed)
                round_data[question][model_name] = answer
                print(f"  [{model_name}] Q{q_idx+1}: {answer[:70]}")

        all_rounds.append(round_data)

        # Generate follow-up questions for next round (skip after last round)
        if round_i < N_ROUNDS - 1:
            print(f"\n[Round {round_i+1}] Generating follow-up questions ...")
            # Sample answers for context (one per model, max 5 Q/A pairs)
            sample = {}
            for model_name in model_names:
                pairs = []
                for q, answers in list(round_data.items())[:5]:
                    pairs.append((q, answers.get(model_name, "")))
                sample[model_name] = pairs

            time.sleep(API_DELAY)
            raw_followup = client.chat(
                format_followup_questions(identity, current_questions, sample),
                system=JUDGE_SYSTEM_PROMPT,
                temperature=0.85,
                max_tokens=512,
            )
            current_questions = parse_numbered_list(raw_followup)[:N_FOLLOWUP]
            print(f"  got {len(current_questions)} follow-up questions")
            for i, q in enumerate(current_questions, 1):
                print(f"  {i:2d}. {q}")

    # ── Scoring ────────────────────────────────────────────────────────────────
    time.sleep(API_DELAY)
    print(f"\n[Scoring] Asking judge to evaluate all {len(model_names)} models ...")
    scoring_prompt = format_scoring(identity, model_names, all_rounds)
    score_response = client.chat(
        scoring_prompt,
        system=JUDGE_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=3000,
    )
    print("\n" + "="*60)
    print("JUDGE SCORES")
    print("="*60)
    print(score_response)

    client.close()

    # ── Package results ────────────────────────────────────────────────────────
    result = {
        "timestamp": datetime.now().isoformat(),
        "models": model_names,
        "n_rounds": N_ROUNDS,
        "rounds": [
            {
                "round": i + 1,
                "questions": list(r.keys()),
                "answers": r,
            }
            for i, r in enumerate(all_rounds)
        ],
        "judge_scores": score_response,
    }
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fish-persona conversational eval")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Which small_lm models to evaluate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model inference (test API + prompt flow only)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    result = run_eval(args.models, dry_run=args.dry_run)

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
