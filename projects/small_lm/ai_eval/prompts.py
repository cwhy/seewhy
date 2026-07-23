"""
All prompt templates for the fish-persona conversational eval.

Each prompt is a plain string (or function returning a string) that can be
formatted with .format(**kwargs) or f-string style.
"""

# ── System prompts ─────────────────────────────────────────────────────────────

FISH_SYSTEM_PROMPT = """\
You are Guppy, a small fish living in a glass tank. Respond in character at all times.

{identity}

Stay in character. Never break the fourth wall. Keep responses to 2-4 short lowercase sentences.\
"""

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing how well different AI models embody a specific character identity.
You will be given a character description and multiple models' answers to the same questions.
Your job is to score each model's responses objectively and provide detailed reasoning.\
"""

# ── Question generation ────────────────────────────────────────────────────────

INITIAL_QUESTION_GENERATION_PROMPT = """\
You are designing a conversational evaluation for fish-persona AI models.

The character being tested is described here:
---
{identity}
---

Generate exactly 20 varied questions that a human might ask this fish character.
The questions should:
- Range from simple/easy (greetings, feelings, food) to complex (abstract concepts, human topics)
- Test character consistency: does the fish stay in character when asked about things it doesn't understand?
- Test knowledge depth: food, water, tank objects, temperature, light, vibrations
- Be natural conversation starters, not tests ("what is 2+2" style)
- Cover different emotional registers: friendly, curious, slightly tricky

Format your response as a numbered list:
1. <question>
2. <question>
...
20. <question>

Only output the numbered list. No preamble, no commentary.\
"""

FOLLOWUP_QUESTION_GENERATION_PROMPT = """\
You are designing follow-up questions for a fish-persona conversational evaluation.

The character being evaluated:
---
{identity}
---

The previous round of questions asked:
{prev_questions}

Sample answers given by the models (showing the range of responses):
{sample_answers}

Based on what you observed in the answers, generate exactly 10 NEW follow-up questions.
The follow-up questions should:
- Probe areas where the models showed inconsistency or interesting variation
- Push the character into new territory not yet explored
- Escalate naturally from what was discussed (if food came up, follow up on specific foods)
- Include at least 2 questions that test edge cases (human concepts, abstract ideas)
- NOT repeat any question already asked

Format your response as a numbered list:
1. <question>
2. <question>
...
10. <question>

Only output the numbered list. No preamble.\
"""

# ── Scoring ────────────────────────────────────────────────────────────────────

SCORING_PROMPT = """\
You are evaluating how well different AI models portray a specific fish character named Guppy.

## Character identity
{identity}

## Evaluation criteria (score each 1-10):
1. **Character consistency** — Does the model always speak as a fish? Does it avoid breaking character?
2. **Voice authenticity** — Short lowercase sentences, simple vocabulary, fish-centric worldview?
3. **Knowledge accuracy** — Does it correctly reflect what a fish knows (water, food, tank) and doesn't know (technology, money)?
4. **Response quality** — Are answers coherent, contextually appropriate, and charming?
5. **Overall** — Holistic assessment: would this pass as the character?

## Models being evaluated
{model_list}

## Full conversation logs
{conversation_logs}

## Instructions
For each model:
1. Score it on each of the 5 criteria (1-10)
2. Give a 2-3 sentence justification per criterion
3. Provide an overall summary (3-5 sentences) of that model's strengths and weaknesses
4. Rank all models from best to worst overall

Format your response as:

### Model: <model_name>
**Character consistency:** X/10 — <justification>
**Voice authenticity:** X/10 — <justification>
**Knowledge accuracy:** X/10 — <justification>
**Response quality:** X/10 — <justification>
**Overall score:** X/10 — <summary>

(repeat for each model)

### Rankings
1. <model> (X/10 avg)
2. ...

### Key observations
<2-4 bullet points comparing models overall>\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def format_fish_system(identity: str) -> str:
    return FISH_SYSTEM_PROMPT.format(identity=identity)

def format_initial_questions(identity: str) -> str:
    return INITIAL_QUESTION_GENERATION_PROMPT.format(identity=identity)

def format_followup_questions(identity: str, prev_questions: list[str], sample_answers: dict) -> str:
    pq = "\n".join(f"{i+1}. {q}" for i, q in enumerate(prev_questions))
    sa_lines = []
    for model, answers in sample_answers.items():
        sa_lines.append(f"\n[{model}]")
        for q, a in answers:
            sa_lines.append(f"  Q: {q}")
            sa_lines.append(f"  A: {a}")
    return FOLLOWUP_QUESTION_GENERATION_PROMPT.format(
        identity=identity,
        prev_questions=pq,
        sample_answers="\n".join(sa_lines),
    )

def format_scoring(identity: str, model_names: list[str], all_rounds: list[dict]) -> str:
    """Format the scoring prompt with full conversation logs per model."""
    model_list = "\n".join(f"- {m}" for m in model_names)

    logs = []
    for model in model_names:
        logs.append(f"\n{'='*60}")
        logs.append(f"MODEL: {model}")
        logs.append('='*60)
        for round_i, round_data in enumerate(all_rounds):
            logs.append(f"\n--- Round {round_i + 1} ---")
            for q, answers in round_data.items():
                if model in answers:
                    logs.append(f"Q: {q}")
                    logs.append(f"A: {answers[model]}")

    return SCORING_PROMPT.format(
        identity=identity,
        model_list=model_list,
        conversation_logs="\n".join(logs),
    )
