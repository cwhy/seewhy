"""
exp_identity_morris.py — Identity-training loop for Old Morris (lighthouse keeper).

Trains a small LLM from the k-Dyck RoPE phase-1 checkpoint to roleplay as a
fictional character using an external model (inception/mercury-2 via OpenRouter)
as teacher.

Loop:
  1. Load pre-sampled questions from next_questions.json
  2. Generate model answers (greedy-ish, temp=0.3, top_k=20)
  3. Batch-interpret via mercury-2 → get exact_sentence + new_facts per pair
  4. Train on (question → exact_sentence) with masked loss (assistant tokens only)
  5. Report to console + report.md
  6. Async: generate more questions + resample next_questions.json

Usage:
    uv run python projects/small_lm/exp_identity_morris.py
"""

import json
import logging
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ── Path setup ────────────────────────────────────────────────────────────────

SMALL_LM  = Path(__file__).parent.parent
REPO_ROOT = SMALL_LM.parent.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SMALL_LM))

from lib.data_processing import load_tokenizer
from lib.prompts import fill, QGEN_SYSTEM, QGEN_USER, INTERP_SYSTEM, INTERP_USER, CONSOLIDATE_SYSTEM, CONSOLIDATE_USER
from ai_eval.openrouter import OpenRouterClient

# ── Import architecture from exp_kdyck_rope ───────────────────────────────────

import importlib.util
_spec = importlib.util.spec_from_file_location("exp_kdyck_rope", SMALL_LM / "exp_kdyck_rope.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

forward     = _mod.forward
VOCAB_SIZE  = _mod.VOCAB_SIZE
MAX_SEQ_LEN = _mod.MAX_SEQ_LEN
PAD_ID      = _mod.PAD_ID

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IDENTITY_FILE  = SMALL_LM / "data" / "identity" / "lighthouse.md"
TOKENIZER_FILE = SMALL_LM / "data" / "tokenizer.json"
CHECKPOINT_IN  = SMALL_LM / "pkl" / "params_exp_kdyck_rope_phase1.pkl"
CHECKPOINT_OUT = SMALL_LM / "pkl" / "params_identity_train.pkl"

OUT_DIR       = SMALL_LM / "babystep" / "experiment-1-morris"
ALL_QUESTIONS = OUT_DIR / "all_questions.jsonl"
NEXT_QUESTIONS= OUT_DIR / "next_questions.json"
INTERP_FACTS  = OUT_DIR / "interpreted_facts.jsonl"
REPORT_FILE   = OUT_DIR / "report.md"

N_BATCHES       = 100
BATCH_SIZE      = 16
ADAM_LR         = 1e-4
GRAD_CLIP       = 1.0
GEN_TEMP        = 0.3
GEN_TOP_K       = 20
MAX_GEN_TOKENS  = 64
N_QUESTIONS_GEN = 20
TEACHER_MODEL   = "openai/gpt-4.1-mini"

# ── Masked loss ───────────────────────────────────────────────────────────────

def masked_loss_fn(params, token_ids, loss_mask):
    """Cross-entropy loss only over positions where loss_mask == 1."""
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    mask    = loss_mask[:, 1:]          # shift mask to match target positions
    logits  = forward(params, inputs)
    B, T1, V = logits.shape
    flat_logits  = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    flat_mask    = mask.reshape(-1).astype(jnp.float32)
    ce = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    return (ce * flat_mask).sum() / (flat_mask.sum() + 1e-8)


def make_train_step(optimizer):
    def train_step(carry, batch_and_mask):
        params, opt_state = carry
        token_ids, loss_mask = batch_and_mask
        loss, grads = jax.value_and_grad(masked_loss_fn)(params, token_ids, loss_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss
    return jax.jit(train_step)

# ── Tokenization helpers ──────────────────────────────────────────────────────

def tokenize_with_mask(tok, question: str, answer: str, max_len: int = MAX_SEQ_LEN):
    """
    Returns (token_ids, loss_mask) as int32 arrays of length max_len.
    loss_mask=1 only for assistant tokens (after <|im_start|>assistant\n).
    """
    prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    full_text   = prompt_text + answer + "<|im_end|>"

    prompt_ids = tok.encode(prompt_text).ids
    full_ids   = tok.encode(full_text).ids

    ids = full_ids[:max_len]
    pad_len = max_len - len(ids)
    ids = ids + [PAD_ID] * pad_len

    prompt_len = min(len(prompt_ids), max_len)
    mask = [0] * prompt_len + [1] * (max_len - prompt_len)
    for i in range(len(ids)):
        if ids[i] == PAD_ID and i >= prompt_len:
            mask[i] = 0

    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)


def batch_tokenize(tok, pairs: list[tuple[str, str]]):
    all_ids, all_masks = [], []
    for q, a in pairs:
        ids, mask = tokenize_with_mask(tok, q, a)
        all_ids.append(ids)
        all_masks.append(mask)
    return np.stack(all_ids, axis=0), np.stack(all_masks, axis=0)

# ── Autoregressive generation ─────────────────────────────────────────────────

def top_k_sample(logits, top_k: int, temperature: float, rng: np.random.Generator):
    logits = logits.astype(np.float32)
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -1e9)
        mask[indices] = logits[indices]
        logits = mask
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate_batch(params, tok, questions: list[str], max_new_tokens: int = MAX_GEN_TOKENS,
                   temperature: float = GEN_TEMP, top_k: int = GEN_TOP_K) -> list[str]:
    """Generate responses for all questions in parallel — one forward pass per token step."""
    rng = np.random.default_rng()
    eos_id = 2  # <|im_end|>
    B = len(questions)

    # Encode all prompts, track where each prompt ends (real length)
    prompts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in questions]
    prompt_ids = [tok.encode(p).ids[:MAX_SEQ_LEN] for p in prompts]

    # seqs: (B, MAX_SEQ_LEN) padded; T_real: current real length per sequence
    seqs    = np.full((B, MAX_SEQ_LEN), PAD_ID, dtype=np.int32)
    T_reals = np.zeros(B, dtype=np.int32)
    for i, ids in enumerate(prompt_ids):
        seqs[i, :len(ids)] = ids
        T_reals[i] = len(ids)

    generated = [[] for _ in range(B)]
    active    = np.ones(B, dtype=bool)   # sequences still generating

    for step in range(max_new_tokens):
        if not active.any():
            break
        # Single batched forward pass over all B sequences
        all_logits = np.array(forward(params, jnp.array(seqs)))  # (B, MAX_SEQ_LEN, V)
        if step == 0:
            logging.info(f"    [gen] first token done (JIT compiled), batch_size={B}")

        for i in range(B):
            if not active[i]:
                continue
            logits   = all_logits[i, T_reals[i] - 1]
            next_tok = top_k_sample(logits, top_k, temperature, rng)
            if next_tok == eos_id:
                active[i] = False
                continue
            generated[i].append(next_tok)
            pos = T_reals[i]
            if pos < MAX_SEQ_LEN:
                seqs[i, pos] = next_tok
                T_reals[i]  += 1

    texts = [tok.decode(g) for g in generated]
    for i, (q, t) in enumerate(zip(questions, texts)):
        logging.info(f"    [gen {i}] {len(generated[i])} tok → \"{t[:60]}\"")
    return texts

# ── External model calls ──────────────────────────────────────────────────────

def call_generate_questions(client: OpenRouterClient, identity: str, facts: list[str]) -> list[str]:
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"
    user_msg    = fill(QGEN_USER, identity=identity, facts_block=facts_block, n_questions=N_QUESTIONS_GEN)

    logging.info(f"  [qgen] requesting {N_QUESTIONS_GEN} questions (facts_known={len(facts)}) ...")
    for attempt in range(3):
        try:
            raw = client.chat(user_msg, system=QGEN_SYSTEM, temperature=0.8, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            qs = data.get("questions", [])
            if isinstance(qs, list) and len(qs) > 0:
                result = [str(q).strip() for q in qs if q]
                logging.info(f"  [qgen] got {len(result)} questions — first: \"{result[0][:60]}\"")
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"  [qgen] JSON parse failed attempt {attempt+1}: {e}")
            user_msg += "\n\nReturn valid JSON only."
    logging.warning("  [qgen] all attempts failed — returning empty list")
    return []


def _interpret_one(client: OpenRouterClient, identity: str, facts_block: str,
                   question: str, answer: str, idx: int) -> dict:
    """Interpret a single (question, answer) pair. Retries up to 3 times."""
    logging.info(f"  [interp {idx}] q: \"{question[:50]}\" | model: \"{answer[:40]}\"")
    user_msg = fill(INTERP_USER, identity=identity, facts_block=facts_block,
                    question=question, answer=answer)
    for attempt in range(3):
        try:
            raw = client.chat(user_msg, system=INTERP_SYSTEM, temperature=0.3, max_tokens=256)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            if "exact_sentence" in data:
                result = {"new_facts": data.get("new_facts", []), "exact_sentence": data["exact_sentence"]}
                logging.info(f"  [interp {idx}] → \"{result['exact_sentence'][:60]}\" | facts: {result['new_facts']}")
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"  [interp {idx}] JSON parse failed attempt {attempt+1}: {e} | raw: {raw[:80]!r}")
            user_msg += "\n\nReturn valid JSON only."
    logging.warning(f"  [interp {idx}] all attempts failed — using placeholder")
    return {"new_facts": [], "exact_sentence": "The sea does not answer."}


def call_batch_interpret(client: OpenRouterClient, identity: str, facts: list[str],
                         qa_pairs: list[tuple[str, str]]) -> list[dict]:
    """Fire one parallel request per (question, answer) pair."""
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"
    logging.info(f"  [interp] firing {len(qa_pairs)} parallel requests ...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(qa_pairs)) as pool:
        futures = [
            pool.submit(_interpret_one, client, identity, facts_block, q, a, i)
            for i, (q, a) in enumerate(qa_pairs)
        ]
        results = [f.result() for f in futures]
    logging.info(f"  [interp] all {len(qa_pairs)} done in {time.perf_counter()-t0:.1f}s")
    return results

def consolidate_facts(client: OpenRouterClient, identity: str, facts: list[str]) -> list[str]:
    """Merge accumulated facts into a clean orthogonal list via the external model."""
    facts_block = "\n".join(f"- {f}" for f in facts)
    user_msg    = fill(CONSOLIDATE_USER, identity=identity, facts_block=facts_block)
    logging.info(f"  [consolidate] merging {len(facts)} facts ...")
    for attempt in range(3):
        try:
            raw = client.chat(user_msg, system=CONSOLIDATE_SYSTEM, temperature=0.2, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            result = [str(f).strip() for f in data.get("facts", []) if f]
            logging.info(f"  [consolidate] {len(facts)} → {len(result)} facts")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"  [consolidate] JSON parse failed attempt {attempt+1}: {e}")
            user_msg += "\n\nReturn valid JSON only."
    logging.warning("  [consolidate] all attempts failed — keeping existing facts")
    return facts


# ── Question bank helpers ─────────────────────────────────────────────────────

def append_questions(questions: list[str], batch_idx: int):
    ts = time.time()
    with open(ALL_QUESTIONS, "a") as f:
        for q in questions:
            f.write(json.dumps({"question": q, "batch_idx": batch_idx, "ts": ts}) + "\n")


def load_all_questions() -> list[str]:
    if not ALL_QUESTIONS.exists():
        return []
    qs = []
    with open(ALL_QUESTIONS) as f:
        for line in f:
            try:
                qs.append(json.loads(line)["question"])
            except Exception:
                pass
    return qs


def sample_questions(all_qs: list[str], n: int = BATCH_SIZE) -> list[str]:
    """Recency-weighted sampling (quadratic) without replacement."""
    if len(all_qs) <= n:
        return list(all_qs)
    weights = np.arange(1, len(all_qs) + 1, dtype=float) ** 2
    weights /= weights.sum()
    indices = np.random.choice(len(all_qs), size=n, replace=False, p=weights)
    return [all_qs[i] for i in indices]


def save_next_questions(questions: list[str]):
    with open(NEXT_QUESTIONS, "w") as f:
        json.dump(questions, f, indent=2)


def load_next_questions() -> list[str]:
    if not NEXT_QUESTIONS.exists():
        return []
    with open(NEXT_QUESTIONS) as f:
        return json.load(f)

# ── Reporting ─────────────────────────────────────────────────────────────────

def report_batch(batch_idx: int, loss: float, facts_total: int,
                 questions: list[str], new_facts_this_batch: list[str],
                 sample_qa: tuple[str, str, str]):
    q_preview = " | ".join(f'"{q[:50]}"' for q in questions[:3])
    sample_q, sample_model, sample_label = sample_qa

    sep = "━" * 50
    print(
        f"\n{sep}\n"
        f"Batch {batch_idx}/{N_BATCHES} | loss={loss:.4f} | facts_total={facts_total}\n"
        f"Questions: {q_preview} ...\n"
        f"New facts: {new_facts_this_batch[:3]}\n"
        f'Sample Q: "{sample_q}"\n'
        f'  Model → "{sample_model}"\n'
        f'  Label → "{sample_label}"\n'
        f"{sep}"
    )

    with open(REPORT_FILE, "a") as f:
        f.write(f"""
## Batch {batch_idx}/{N_BATCHES}

**Loss:** `{loss:.4f}` | **Total facts:** {facts_total}

**Questions (first 3):**
{chr(10).join(f'- {q}' for q in questions[:3])}

**New facts this batch:**
{chr(10).join(f'- {fa}' for fa in new_facts_this_batch) if new_facts_this_batch else '- (none)'}

**Sample exchange:**
- Q: {sample_q}
- Model: *{sample_model}*
- Label: *{sample_label}*

---
""")


def init_report():
    with open(REPORT_FILE, "w") as f:
        f.write("# Identity Training Report — Old Morris\n\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info(f"JAX devices: {jax.devices()}")
    tok      = load_tokenizer(TOKENIZER_FILE)
    identity = IDENTITY_FILE.read_text()

    with open(CHECKPOINT_IN, "rb") as f:
        params_np = pickle.load(f)
    params = {k: jnp.array(v) for k, v in params_np.items()}
    logging.info(f"Params: {sum(v.size for v in jax.tree_util.tree_leaves(params)):,}")

    # Load accumulated facts from prior runs
    all_facts: list[str] = []
    if INTERP_FACTS.exists():
        with open(INTERP_FACTS) as f:
            for line in f:
                try:
                    all_facts.extend(json.loads(line).get("new_facts", []))
                except Exception:
                    pass
        logging.info(f"Loaded {len(all_facts)} existing facts")

    optimizer  = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adamw(ADAM_LR, weight_decay=0.0))
    opt_state  = optimizer.init(params)
    train_step = make_train_step(optimizer)
    client     = OpenRouterClient(model=TEACHER_MODEL)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_report()

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    if not ALL_QUESTIONS.exists() or not load_all_questions():
        logging.info("Bootstrap: generating initial questions ...")
        qs = call_generate_questions(client, identity, all_facts)
        append_questions(qs, batch_idx=0)
        logging.info(f"Bootstrap: saved {len(qs)} questions to {ALL_QUESTIONS}")
    else:
        logging.info(f"Bootstrap: {len(load_all_questions())} questions already in bank")

    if not NEXT_QUESTIONS.exists():
        sampled = sample_questions(load_all_questions())
        save_next_questions(sampled)
        logging.info(f"Bootstrap: sampled {len(sampled)} questions → next_questions.json")

    # ── Background refresh ────────────────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=3)
    bg_future: Future | None = None
    consolidate_future: Future | None = None
    consolidate_snapshot_len: int = 0  # len(all_facts) at time of submission

    def background_refresh(batch_idx: int):
        new_qs = call_generate_questions(client, identity, all_facts)
        if new_qs:
            append_questions(new_qs, batch_idx=batch_idx)
        save_next_questions(sample_questions(load_all_questions()))

    # ── Training loop ─────────────────────────────────────────────────────────
    t_total = time.perf_counter()

    for batch_idx in range(1, N_BATCHES + 1):
        t_batch = time.perf_counter()

        if bg_future is not None and not bg_future.done():
            logging.info(f"  [batch {batch_idx}] waiting for background refresh ...")
            bg_future.result()

        # Apply consolidated facts if ready
        if consolidate_future is not None and consolidate_future.done():
            consolidated = consolidate_future.result()
            # Prepend consolidated facts, keep any new facts added since snapshot
            all_facts[:] = consolidated + all_facts[consolidate_snapshot_len:]
            logging.info(f"  [batch {batch_idx}] consolidated facts applied → {len(all_facts)} entries")
            consolidate_future = None

        questions = load_next_questions()[:BATCH_SIZE]
        if len(questions) < BATCH_SIZE:
            questions = sample_questions(load_all_questions(), min(BATCH_SIZE, len(load_all_questions())))

        logging.info(f"  [batch {batch_idx}] generating {len(questions)} model answers (batched) ...")
        t_gen = time.perf_counter()
        model_answers = generate_batch(params, tok, questions)
        logging.info(f"  [batch {batch_idx}] generation done in {time.perf_counter()-t_gen:.1f}s")

        logging.info(f"  [batch {batch_idx}] calling mercury-2 ...")
        qa_pairs       = list(zip(questions, model_answers))
        interpretations = call_batch_interpret(client, identity, all_facts, qa_pairs)

        new_facts_this_batch = []
        with open(INTERP_FACTS, "a") as f:
            for interp, (q, model_ans) in zip(interpretations, qa_pairs):
                nf = interp.get("new_facts", [])
                new_facts_this_batch.extend(nf)
                all_facts.extend(nf)
                f.write(json.dumps({
                    "batch": batch_idx, "question": q, "llm_answer": model_ans,
                    "exact_sentence": interp.get("exact_sentence", ""), "new_facts": nf,
                }) + "\n")

        pairs_for_train = [(q, interp.get("exact_sentence", "")) for q, interp in zip(questions, interpretations)]
        token_ids_np, loss_mask_np = batch_tokenize(tok, pairs_for_train)
        mask_frac = loss_mask_np.mean()
        logging.info(f"  [batch {batch_idx}] tokenized — mask coverage: {mask_frac:.2%}")
        logging.info(f"  [batch {batch_idx}] running train step ...")
        t_train = time.perf_counter()
        (params, opt_state), loss = train_step(
            (params, opt_state), (jnp.array(token_ids_np), jnp.array(loss_mask_np))
        )
        loss_val = float(loss)
        logging.info(f"  [batch {batch_idx}] train step done in {time.perf_counter()-t_train:.1f}s — loss={loss_val:.4f}")

        report_batch(
            batch_idx, loss_val, len(all_facts), questions, new_facts_this_batch,
            (questions[0], model_answers[0][:80], interpretations[0].get("exact_sentence", "")[:80]),
        )
        logging.info(f"  [batch {batch_idx}] loss={loss_val:.4f}  facts={len(all_facts)}  "
                     f"elapsed={time.perf_counter()-t_total:.0f}s  "
                     f"batch_time={time.perf_counter()-t_batch:.1f}s")

        bg_future = executor.submit(background_refresh, batch_idx)

        # Every 10 batches: kick off async fact consolidation
        if batch_idx % 10 == 0 and consolidate_future is None:
            consolidate_snapshot_len = len(all_facts)
            consolidate_future = executor.submit(consolidate_facts, client, identity, list(all_facts))
            logging.info(f"  [batch {batch_idx}] fact consolidation submitted async ({len(all_facts)} facts)")

    # ── Final save ────────────────────────────────────────────────────────────
    if bg_future is not None:
        bg_future.result()
    if consolidate_future is not None:
        consolidated = consolidate_future.result()
        all_facts[:] = consolidated + all_facts[consolidate_snapshot_len:]
        logging.info(f"Final consolidation applied → {len(all_facts)} facts")
    executor.shutdown(wait=False)

    CHECKPOINT_OUT.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT_OUT, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"Checkpoint saved → {CHECKPOINT_OUT}")

    # ── Post-training eval ────────────────────────────────────────────────────
    eval_questions = [
        "How do you feel when the supply boat is late?",
        "Do you ever get lonely out here?",
        "What does the sea teach you about patience?",
        "Have you ever thought about leaving?",
        "What does the darkness feel like when the light is the only thing burning?",
    ]

    sep = "━" * 50
    print(f"\n{sep}")
    print("POST-TRAINING EVAL — Old Morris after training")
    print(sep)
    eval_answers = generate_batch(params, tok, eval_questions, temperature=0.5, top_k=40)
    eval_lines = ["## Post-Training Eval\n"]
    for q, ans in zip(eval_questions, eval_answers):
        print(f"Q: {q}")
        print(f"A: {ans}")
        print()
        eval_lines.append(f"**Q:** {q}\n**A:** {ans}\n")
    print(sep)

    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Done\n\nTotal facts: {len(all_facts)}\nTotal time: {time.perf_counter()-t_total:.0f}s\n\n")
        f.write("\n".join(eval_lines))

    logging.info(f"Done. facts={len(all_facts)}  time={time.perf_counter()-t_total:.0f}s")
    client.close()


if __name__ == "__main__":
    main()
