"""
exp_identity_kylo.py — Identity-training loop for Kylo (shiba inu).

Architecture:
  - 1 new pair per iteration (decouple from batch size)
  - Main thread: model inference (JAX)
  - Background thread: external model interpretation (HTTP) + question generation
  - pairs.jsonl: growing data file, sampled recency-weighted for training
  - Training runs whenever ≥ TRAIN_BATCH_SIZE pairs are available

Usage:
    uv run python projects/small_lm/babystep/exp_identity_kylo.py
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
from lib.prompts import (fill, QGEN_SYSTEM, QGEN_USER,
                          INTERP_SYSTEM, INTERP_USER,
                          CONSOLIDATE_SYSTEM, CONSOLIDATE_USER)
from ai_eval.openrouter import OpenRouterClient

import importlib.util
_spec = importlib.util.spec_from_file_location("exp_kdyck_rope", SMALL_LM / "exp_kdyck_rope.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

forward     = _mod.forward
MAX_SEQ_LEN = _mod.MAX_SEQ_LEN
PAD_ID      = _mod.PAD_ID

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IDENTITY_FILE  = SMALL_LM / "data" / "identity" / "kylo.md"
TOKENIZER_FILE = SMALL_LM / "data" / "tokenizer.json"
CHECKPOINT_IN  = SMALL_LM / "pkl" / "params_exp_kdyck_rope_phase1.pkl"
CHECKPOINT_OUT = SMALL_LM / "pkl" / "params_identity_kylo.pkl"

OUT_DIR            = SMALL_LM / "babystep" / "experiment-2-kylo"
PAIRS_FILE         = OUT_DIR / "pairs.jsonl"
PAIR_LOSSES_FILE   = OUT_DIR / "pair_losses.json"
PENDING_QUESTION   = OUT_DIR / "pending_question.json"
ALL_QUESTIONS_FILE = OUT_DIR / "all_questions.jsonl"
INTERP_FACTS_FILE  = OUT_DIR / "interpreted_facts.jsonl"
REPORT_FILE        = OUT_DIR / "report.md"

N_ITERATIONS      = 2000       # total iterations (1 pair generated per iter)
TRAIN_BATCH_SIZE  = 16         # pairs sampled for each train step
TRAIN_START_AT    = 16         # start training once this many pairs exist
ADAM_LR           = 1e-4
GRAD_CLIP         = 1.0
GEN_TEMP          = 0.3
GEN_TOP_K         = 20
MAX_GEN_TOKENS    = 64
N_QUESTIONS_GEN   = 20
TEACHER_MODEL     = "google/gemini-2.5-flash-lite-preview-09-2025"
CONSOLIDATE_EVERY = 10

# ── Masked loss ───────────────────────────────────────────────────────────────

def masked_loss_fn(params, token_ids, loss_mask):
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    mask    = loss_mask[:, 1:].astype(jnp.float32)
    logits  = forward(params, inputs)
    flat_logits  = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask    = mask.reshape(-1)
    ce = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    return (ce * flat_mask).sum() / (flat_mask.sum() + 1e-8)


@jax.jit
def per_sample_losses_fn(params, token_ids, loss_mask):
    """Return per-sample mean loss, shape (B,)."""
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    mask    = loss_mask[:, 1:].astype(jnp.float32)
    logits  = forward(params, inputs)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).reshape(token_ids.shape[0], -1)
    return (ce * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-8)


def make_train_step(optimizer):
    def train_step(carry, batch_and_mask):
        params, opt_state = carry
        token_ids, loss_mask = batch_and_mask
        loss, grads = jax.value_and_grad(masked_loss_fn)(params, token_ids, loss_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss
    return jax.jit(train_step)

# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_with_mask(tok, question: str, answer: str):
    prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    full_text   = prompt_text + answer + "<|im_end|>"
    prompt_ids  = tok.encode(prompt_text).ids
    full_ids    = tok.encode(full_text).ids

    ids     = full_ids[:MAX_SEQ_LEN]
    pad_len = MAX_SEQ_LEN - len(ids)
    ids     = ids + [PAD_ID] * pad_len

    prompt_len = min(len(prompt_ids), MAX_SEQ_LEN)
    mask = [0] * prompt_len + [1] * (MAX_SEQ_LEN - prompt_len)
    for i in range(len(ids)):
        if ids[i] == PAD_ID and i >= prompt_len:
            mask[i] = 0

    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)


def batch_tokenize(tok, pairs: list[tuple[str, str]]):
    ids_list, mask_list = [], []
    for q, a in pairs:
        ids, mask = tokenize_with_mask(tok, q, a)
        ids_list.append(ids)
        mask_list.append(mask)
    return np.stack(ids_list), np.stack(mask_list)

# ── Single-sequence generation ────────────────────────────────────────────────

def top_k_sample(logits, top_k, temperature, rng):
    logits = logits.astype(np.float32)
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        idx = np.argpartition(logits, -top_k)[-top_k:]
        m = np.full_like(logits, -1e9)
        m[idx] = logits[idx]
        logits = m
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate_one(params, tok, question: str) -> str:
    rng = np.random.default_rng()
    prompt    = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tok.encode(prompt).ids[:MAX_SEQ_LEN]
    eos_id    = 2

    generated = []
    for step in range(MAX_GEN_TOKENS):
        seq    = (input_ids + generated)[-MAX_SEQ_LEN:]
        T_real = len(seq)
        padded = seq + [PAD_ID] * (MAX_SEQ_LEN - T_real)
        logits = np.array(forward(params, jnp.array(np.array(padded, dtype=np.int32)[None]))[0, T_real - 1])
        tok_id = top_k_sample(logits, GEN_TOP_K, GEN_TEMP, rng)
        if step == 0:
            logging.info("    [gen] JIT compiled")
        if tok_id == eos_id:
            break
        generated.append(tok_id)

    return tok.decode(generated)

# ── External model ────────────────────────────────────────────────────────────

def call_generate_questions(client, identity, facts):
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"
    user_msg    = fill(QGEN_USER, identity=identity, facts_block=facts_block, n_questions=N_QUESTIONS_GEN)
    logging.info(f"  [qgen] requesting {N_QUESTIONS_GEN} questions ...")
    for attempt in range(3):
        try:
            raw  = client.chat(user_msg, system=QGEN_SYSTEM, temperature=0.8, max_tokens=1024)
            data = json.loads(_strip_fences(raw))
            qs   = [str(q).strip() for q in data.get("questions", []) if q]
            if qs:
                logging.info(f"  [qgen] got {len(qs)} — first: \"{qs[0][:60]}\"")
                return qs
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"  [qgen] attempt {attempt+1} failed: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return []


def interpret_one(client, identity, facts_block, question, answer):
    user_msg = fill(INTERP_USER, identity=identity, facts_block=facts_block,
                    question=question, answer=answer)
    for attempt in range(3):
        try:
            raw  = client.chat(user_msg, system=INTERP_SYSTEM, temperature=0.3, max_tokens=256, retries=0)
            data = json.loads(_strip_fences(raw))
            if "exact_sentence" in data:
                result = {"new_facts": data.get("new_facts", []), "exact_sentence": data["exact_sentence"]}
                logging.info(f"  [interp] → \"{result['exact_sentence'][:70]}\"")
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"  [interp] attempt {attempt+1} failed: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return {"new_facts": [], "exact_sentence": "woof."}


def consolidate_facts(client, identity, facts):
    facts_block = "\n".join(f"- {f}" for f in facts)
    user_msg    = fill(CONSOLIDATE_USER, identity=identity, facts_block=facts_block)
    logging.info(f"  [consolidate] merging {len(facts)} facts ...")
    for attempt in range(3):
        try:
            raw    = client.chat(user_msg, system=CONSOLIDATE_SYSTEM, temperature=0.2, max_tokens=1024)
            data   = json.loads(_strip_fences(raw))
            result = [str(f).strip() for f in data.get("facts", []) if f]
            logging.info(f"  [consolidate] {len(facts)} → {len(result)} facts")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"  [consolidate] attempt {attempt+1} failed: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return facts


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw

# ── Data helpers ──────────────────────────────────────────────────────────────

def append_pair(question: str, exact_sentence: str, iter_idx: int):
    with open(PAIRS_FILE, "a") as f:
        f.write(json.dumps({"question": question, "exact_sentence": exact_sentence,
                            "iter": iter_idx, "ts": time.time()}) + "\n")


def load_pairs() -> list[tuple[str, str]]:
    if not PAIRS_FILE.exists():
        return []
    pairs = []
    with open(PAIRS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                pairs.append((r["question"], r["exact_sentence"]))
            except Exception:
                pass
    return pairs


def load_pair_losses() -> dict[int, float]:
    if not PAIR_LOSSES_FILE.exists():
        return {}
    with open(PAIR_LOSSES_FILE) as f:
        return {int(k): v for k, v in json.load(f).items()}


def save_pair_losses(pair_losses: dict[int, float]):
    with open(PAIR_LOSSES_FILE, "w") as f:
        json.dump({str(k): v for k, v in pair_losses.items()}, f)


def sample_pairs(all_pairs: list[tuple[str, str]], n: int,
                 pair_losses: dict[int, float]) -> tuple[list[tuple[str, str]], list[int]]:
    """Sample n pairs; returns (pairs, global_indices).
    Weight = 0.5 * recency + 0.5 * loss  (smaller loss → higher weight).
    """
    N = len(all_pairs)
    if N <= n:
        return list(all_pairs), list(range(N))

    # recency: quadratic ramp, index 0 = oldest = lowest weight
    w_rec = np.arange(1, N + 1, dtype=float) ** 2

    # loss: smaller loss → higher weight; unknown pairs use mean of known losses
    known_losses = [pair_losses[i] for i in range(N) if i in pair_losses]
    fallback = float(np.mean(known_losses)) if known_losses else 5.0
    losses = np.array([pair_losses.get(i, fallback) for i in range(N)], dtype=float)
    w_loss = (losses.max() + 1.0) - losses  # invert: smaller loss → larger weight

    w = 0.5 * (w_rec / w_rec.sum()) + 0.5 * (w_loss / w_loss.sum())
    w /= w.sum()

    idx = np.random.choice(N, size=n, replace=False, p=w)
    return [all_pairs[i] for i in idx], list(idx)


def append_questions(questions, iter_idx):
    with open(ALL_QUESTIONS_FILE, "a") as f:
        for q in questions:
            f.write(json.dumps({"question": q, "iter": iter_idx, "ts": time.time()}) + "\n")


def load_all_questions() -> list[str]:
    if not ALL_QUESTIONS_FILE.exists():
        return []
    qs = []
    with open(ALL_QUESTIONS_FILE) as f:
        for line in f:
            try:
                qs.append(json.loads(line)["question"])
            except Exception:
                pass
    return qs


def pop_pending_question() -> str | None:
    if not PENDING_QUESTION.exists():
        return None
    try:
        with open(PENDING_QUESTION) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    q      = data.get("questions", [])
    next_q = q.pop(0) if q else None
    try:
        with open(PENDING_QUESTION, "w") as f:
            json.dump({"questions": q}, f)
    except OSError:
        pass
    return next_q


def refill_pending(all_qs: list[str], n: int = 20):
    if len(all_qs) == 0:
        return
    weights = np.arange(1, len(all_qs) + 1, dtype=float) ** 2
    weights /= weights.sum()
    chosen = list(np.random.choice(all_qs, size=min(n, len(all_qs)), replace=False, p=weights))
    existing = []
    if PENDING_QUESTION.exists():
        try:
            with open(PENDING_QUESTION) as f:
                existing = json.load(f).get("questions", [])
        except (json.JSONDecodeError, OSError):
            existing = []
    with open(PENDING_QUESTION, "w") as f:
        json.dump({"questions": existing + chosen}, f)

# ── Reporting ─────────────────────────────────────────────────────────────────

def init_report():
    with open(REPORT_FILE, "w") as f:
        f.write("# Identity Training Report — Kylo\n\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")


def report_iter(iter_idx, loss, n_pairs, question, model_ans, label, new_facts):
    sep = "━" * 50
    print(
        f"\n{sep}\n"
        f"Iter {iter_idx}/{N_ITERATIONS} | loss={loss:.4f} | pairs={n_pairs}\n"
        f'Q: "{question}"\n'
        f'  Model → "{model_ans[:70]}"\n'
        f'  Label → "{label[:70]}"\n'
        f"  New facts: {new_facts[:2]}\n"
        f"{sep}"
    )
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Iter {iter_idx}/{N_ITERATIONS}\n\n"
                f"**Loss:** `{loss:.4f}` | **Pairs:** {n_pairs}\n\n"
                f"**Q:** {question}\n"
                f"- Model: *{model_ans[:80]}*\n"
                f"- Label: *{label[:80]}*\n"
                f"- New facts: {new_facts}\n\n---\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info(f"JAX devices: {jax.devices()}")
    tok      = load_tokenizer(TOKENIZER_FILE)
    identity = IDENTITY_FILE.read_text()

    with open(CHECKPOINT_IN, "rb") as f:
        params = {k: jnp.array(v) for k, v in pickle.load(f).items()}
    logging.info(f"Params: {sum(v.size for v in jax.tree_util.tree_leaves(params)):,}")

    pair_losses: dict[int, float] = load_pair_losses()
    all_facts: list[str] = []
    if INTERP_FACTS_FILE.exists():
        with open(INTERP_FACTS_FILE) as f:
            for line in f:
                try:
                    all_facts.extend(json.loads(line).get("new_facts", []))
                except Exception:
                    pass

    optimizer  = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adamw(ADAM_LR, weight_decay=0.0))
    opt_state  = optimizer.init(params)
    train_step = make_train_step(optimizer)
    client     = OpenRouterClient(model=TEACHER_MODEL)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_report()

    # ── Bootstrap question bank ────────────────────────────────────────────────
    if not ALL_QUESTIONS_FILE.exists() or not load_all_questions():
        logging.info("Bootstrap: generating initial questions ...")
        qs = call_generate_questions(client, identity, all_facts)
        append_questions(qs, iter_idx=0)
    refill_pending(load_all_questions())

    # ── Thread pool ────────────────────────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=3)
    interp_future: Future | None = None        # pending interpretation
    interp_question: str         = ""
    interp_model_ans: str        = ""

    # previous iter's resolved pair — used for reporting (label is known one iter later)
    prev_question:   str = ""
    prev_model_ans:  str = ""
    prev_label:      str = ""
    prev_new_facts:  list = []

    consolidate_future: Future | None = None
    consolidate_snapshot_len: int     = 0

    t_total = time.perf_counter()

    for iter_idx in range(1, N_ITERATIONS + 1):
        t_iter = time.perf_counter()

        # ── Apply consolidated facts if ready ──────────────────────────────
        if consolidate_future is not None and consolidate_future.done():
            all_facts[:] = consolidate_future.result() + all_facts[consolidate_snapshot_len:]
            logging.info(f"  [iter {iter_idx}] consolidated facts applied → {len(all_facts)}")
            consolidate_future = None

        # ── Wait for pending interpretation ────────────────────────────────
        prev_question = prev_model_ans = prev_label = ""
        prev_new_facts = []
        if interp_future is not None:
            interp = interp_future.result()
            nf     = interp.get("new_facts", [])
            label  = interp.get("exact_sentence", "")
            all_facts.extend(nf)
            append_pair(interp_question, label, iter_idx - 1)
            with open(INTERP_FACTS_FILE, "a") as f:
                f.write(json.dumps({"iter": iter_idx - 1, "question": interp_question,
                                    "llm_answer": interp_model_ans, "exact_sentence": label,
                                    "new_facts": nf}) + "\n")
            prev_question, prev_model_ans, prev_label, prev_new_facts = (
                interp_question, interp_model_ans, label, nf)
            interp_future = None

        # ── Pick next question ─────────────────────────────────────────────
        question = pop_pending_question()
        if not question:
            all_qs = load_all_questions()
            if not all_qs:
                logging.warning("  No questions available — regenerating ...")
                all_qs = call_generate_questions(client, identity, all_facts)
                append_questions(all_qs, iter_idx)
            question = all_qs[-1] if all_qs else "What do you like?"
        logging.info(f"  [iter {iter_idx}] q: \"{question[:70]}\"")

        # ── Refill pending if low ──────────────────────────────────────────
        pending_len = 99
        if PENDING_QUESTION.exists():
            try:
                with open(PENDING_QUESTION) as fp:
                    pending_len = len(json.load(fp).get("questions", []))
            except (json.JSONDecodeError, OSError):
                pending_len = 0
        if pending_len < 5:
                executor.submit(lambda: (
                    append_questions(call_generate_questions(client, identity, all_facts), iter_idx),
                    refill_pending(load_all_questions()),
                ))

        # ── Generate model answer (main thread / JAX) ──────────────────────
        t_gen    = time.perf_counter()
        model_ans = generate_one(params, tok, question)
        logging.info(f"  [iter {iter_idx}] gen done in {time.perf_counter()-t_gen:.1f}s → \"{model_ans[:50]}\"")

        # ── Submit interpretation async ────────────────────────────────────
        facts_block    = "\n".join(f"- {f}" for f in all_facts) if all_facts else "(none yet)"
        interp_question  = question
        interp_model_ans = model_ans
        interp_future    = executor.submit(interpret_one, client, identity, facts_block, question, model_ans)

        # ── Train step (if enough pairs) ───────────────────────────────────
        all_pairs = load_pairs()
        if len(all_pairs) >= TRAIN_START_AT:
            sampled, sampled_idx = sample_pairs(all_pairs, TRAIN_BATCH_SIZE, pair_losses)
            token_ids_np, loss_mask_np = batch_tokenize(tok, sampled)
            ids_jnp, mask_jnp = jnp.array(token_ids_np), jnp.array(loss_mask_np)
            t_train = time.perf_counter()
            (params, opt_state), loss = train_step((params, opt_state), (ids_jnp, mask_jnp))
            loss_val = float(loss)
            # record per-sample losses for future sampling
            sample_losses = np.array(per_sample_losses_fn(params, ids_jnp, mask_jnp))
            for gi, sl in zip(sampled_idx, sample_losses):
                pair_losses[gi] = float(sl)
            save_pair_losses(pair_losses)
            logging.info(f"  [iter {iter_idx}] train done in {time.perf_counter()-t_train:.1f}s — loss={loss_val:.4f}")

            if prev_question:
                report_iter(iter_idx, loss_val, len(all_pairs),
                            prev_question, prev_model_ans, prev_label, prev_new_facts)
        else:
            loss_val = float("nan")
            logging.info(f"  [iter {iter_idx}] skipping train ({len(all_pairs)}/{TRAIN_START_AT} pairs)")

        logging.info(f"  [iter {iter_idx}] done in {time.perf_counter()-t_iter:.1f}s  "
                     f"elapsed={time.perf_counter()-t_total:.0f}s  facts={len(all_facts)}")

        # ── Async fact consolidation every N iters ─────────────────────────
        if iter_idx % CONSOLIDATE_EVERY == 0 and consolidate_future is None and all_facts:
            consolidate_snapshot_len = len(all_facts)
            consolidate_future = executor.submit(consolidate_facts, client, identity, list(all_facts))
            logging.info(f"  [iter {iter_idx}] consolidation submitted ({len(all_facts)} facts)")

    # ── Flush final pending interpretation ────────────────────────────────────
    if interp_future is not None:
        interp = interp_future.result()
        append_pair(interp_question, interp.get("exact_sentence", ""), N_ITERATIONS)
    if consolidate_future is not None:
        all_facts[:] = consolidate_future.result() + all_facts[consolidate_snapshot_len:]

    executor.shutdown(wait=False)

    # ── Save checkpoint ────────────────────────────────────────────────────────
    CHECKPOINT_OUT.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT_OUT, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"Checkpoint saved → {CHECKPOINT_OUT}")

    # ── Post-training eval ─────────────────────────────────────────────────────
    eval_qs = [
        "What do you do when you're bored?",
        "Do you like cats?",
        "What's your favourite thing to do outside?",
        "Are you a good boy?",
        "What do you do when your owner comes home?",
    ]
    sep = "━" * 50
    print(f"\n{sep}\nPOST-TRAINING EVAL — Kylo\n{sep}")
    eval_lines = ["## Post-Training Eval\n"]
    for q in eval_qs:
        ans = generate_one(params, tok, q)
        print(f"Q: {q}\nA: {ans}\n")
        eval_lines.append(f"**Q:** {q}\n**A:** {ans}\n")
    print(sep)

    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Done\n\nTotal pairs: {len(load_pairs())}\n"
                f"Total facts: {len(all_facts)}\n"
                f"Total time: {time.perf_counter()-t_total:.0f}s\n\n")
        f.write("\n".join(eval_lines))

    logging.info(f"Done. pairs={len(load_pairs())}  facts={len(all_facts)}  "
                 f"time={time.perf_counter()-t_total:.0f}s")
    client.close()

    # ── Upload report ──────────────────────────────────────────────────────────
    try:
        from shared_lib.report import save_report_file
        url = save_report_file("small_lm_babystep_kylo", REPORT_FILE)
        logging.info(f"Report uploaded → {url}")
        print(f"\nReport: {url}")
    except Exception as e:
        logging.warning(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
