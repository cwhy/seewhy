"""
prompts.py — All LLM prompts for exp_identity_morris.py.

Use fill(TEMPLATE, var=value) to substitute {{var}} placeholders.
"""


def fill(template: str, **kwargs) -> str:
    """Replace {{key}} placeholders with provided values."""
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


# ── Question generation ───────────────────────────────────────────────────────

QGEN_SYSTEM = (
    'You are a curious, natural interviewer talking with a fictional character. '
    'Your tone is warm, conversational, and human, not formal, clinical, or evaluative. '
    'You should make sure the character understands your questions'
    'Return valid JSON only.'
)


QGEN_USER = """\
{{identity}}

Known facts about this character so far:
{{facts_block}}

Generate {{n_questions}} diverse questions to reveal this character's personality, \
opinions, memories, and experiences. Vary between open and specific questions. \
Ask directly to the charactoer. \
Return JSON only: {"questions": ["...", ...]}"""


# ── Single-pair interpretation ────────────────────────────────────────────────

INTERP_SYSTEM = "You are interpreting interview answers on behalf of a fictional character. Return valid JSON only."

INTERP_USER = """\
{{identity}}

Known facts about this character:
{{facts_block}}

Question: {{question}}
Answer (may be noisy, vague or fragmented): {{answer}}

Use the question as your main guide and imagine what this character would genuinely say. \
You are encouraged to use imagination to interpret personality, memory, and voice \
— if the answer is hard to decipher, try simple best-effort guesses based on what you know about the character.

1. Extract any new facts about the character if the inferred response includes, keep it empty if the facts are alreay there or there is no information \
2. Write the exact sentence this character would say — short, in their voice, as it is \
3. If the information is too vague, you can keep the facts empty, but still try to get a guess of the sentence
4. The sentence you provide should be exactly the answer from the character that your best effort guess is

Return JSON only: {"new_facts": ["..."], "exact_sentence": "..."}"""


# ── Impersonation interpretation ─────────────────────────────────────────────

INTERP_IMPERSONATE_SYSTEM = (
    "You are impersonating a fictional character. "
    "Given a raw model answer, restate it in the character's voice. "
    "Return valid JSON only."
)

INTERP_IMPERSONATE_USER = """\
{{identity}}

Known facts about this character:
{{facts_block}}

Question: {{question}}
Raw model answer (may be noisy or fragmented): {{answer}}

Your task:
1. Interpret what the raw answer is trying to say — best-effort, even if garbled
2. Extract any new facts revealed about the character (leave empty if nothing new or already known)
3. Restate the interpreted meaning in this character's voice — short, natural, as they would say it

Key rule: you are voicing the character's version of what was said, not inventing a new answer from scratch.
If the answer is complete gibberish with no recoverable meaning, fall back to a short in-character response to the question.

Return JSON only: {"new_facts": ["..."], "exact_sentence": "..."}"""


# ── Fact consolidation ────────────────────────────────────────────────────────

CONSOLIDATE_SYSTEM = "You are maintaining a concise, accurate character profile. Return valid JSON only."

CONSOLIDATE_USER = """\
{{identity}}

Below is a raw list of facts accumulated about this character through interviews. \
Many are redundant, overlapping, or contradict each other. \
Your job is to merge them into a clean, orthogonal list — each entry should be \
a distinct, non-overlapping fact about who this character is. \
Eliminate duplicates, merge related points into single clear concise statements, \
and remove anything already implied by the identity description above. \
The final fact count should not be larger than 64, remove less important facts if there are more. \

Raw facts:
{{facts_block}}

Return JSON only: {"facts": ["..."]}"""
