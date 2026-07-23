"""
exp_kylo_continual.py — Batched continual identity training for Kylo.

Runs N batches starting from any checkpoint, saves checkpoint.pkl for the next run.
All data accumulates in experiment-3-kylo-continual/ across runs.

Usage:
    # First run (from k-Dyck pretraining checkpoint):
    uv run python projects/small_lm/babystep/exp_kylo_continual.py

    # Continue from saved checkpoint:
    uv run python projects/small_lm/babystep/exp_kylo_continual.py \\
        --from projects/small_lm/babystep/experiment-3-kylo-continual/checkpoint.pkl

    # Specify batch count:
    uv run python projects/small_lm/babystep/exp_kylo_continual.py --batches 200

Each batch:
  - Generate BATCH_SIZE answers in parallel (batched JAX forward pass)
  - Fire BATCH_SIZE parallel interpretation requests
  - Train on recency+loss-weighted sample from pairs history
  - If a question yields no new facts → requeue it for future use
"""

import argparse, json, logging, pickle, sys, time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax

# ── Paths ─────────────────────────────────────────────────────────────────────

SMALL_LM  = Path(__file__).parent.parent
REPO_ROOT = SMALL_LM.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SMALL_LM))

from lib.data_processing import load_tokenizer
from lib.prompts import (fill, QGEN_SYSTEM, QGEN_USER,
                         INTERP_SYSTEM, INTERP_USER,
                         INTERP_IMPERSONATE_SYSTEM, INTERP_IMPERSONATE_USER,
                         CONSOLIDATE_SYSTEM, CONSOLIDATE_USER)
from ai_eval.openrouter import OpenRouterClient

import importlib.util
_spec = importlib.util.spec_from_file_location("exp_kdyck_rope", SMALL_LM / "exp_kdyck_rope.py")
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
forward     = _mod.forward
init_params = _mod.init_params
MAX_SEQ_LEN = _mod.MAX_SEQ_LEN
PAD_ID      = _mod.PAD_ID

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

parser = argparse.ArgumentParser(description="Kylo continual identity training")
parser.add_argument("--from", dest="checkpoint_in", default=None,
                    help="Input checkpoint (.pkl). Defaults to k-Dyck phase-1 checkpoint.")
parser.add_argument("--batches", type=int, default=100, help="Number of batches to run.")
parser.add_argument("--out-dir", dest="out_dir", default=None,
                    help="Experiment output directory. Defaults to experiment-3-kylo-continual.")
parser.add_argument("--random-init", dest="random_init", action="store_true",
                    help="Initialize params randomly instead of loading a checkpoint.")
parser.add_argument("--impersonate", action="store_true",
                    help="Use impersonation teacher mode: voice model's answer as character.")
args = parser.parse_args()

IDENTITY_FILE  = SMALL_LM / "data" / "identity" / "kylo.md"
TOKENIZER_FILE = SMALL_LM / "data" / "tokenizer.json"
KDYCK_CKPT     = SMALL_LM / "pkl" / "params_exp_kdyck_rope_phase1.pkl"

OUT_DIR         = Path(args.out_dir) if args.out_dir else SMALL_LM / "babystep" / "experiment-3-kylo-continual"
CHECKPOINT_IN   = Path(args.checkpoint_in) if args.checkpoint_in else KDYCK_CKPT
CHECKPOINT_OUT  = OUT_DIR / "checkpoint.pkl"
PAIRS_FILE      = OUT_DIR / "pairs.jsonl"
PAIR_LOSSES_FILE= OUT_DIR / "pair_losses.json"
ALL_Q_FILE      = OUT_DIR / "all_questions.jsonl"
PENDING_FILE    = OUT_DIR / "pending_questions.json"
INTERP_FILE     = OUT_DIR / "interpreted_facts.jsonl"
FACTS_FILE      = OUT_DIR / "facts.json"
RUN_TAG         = time.strftime("%Y%m%d_%H%M%S")
REPORT_FILE     = OUT_DIR / f"report_{RUN_TAG}.md"

N_BATCHES         = args.batches
BATCH_SIZE        = 16
TRAIN_BATCH_SIZE  = 16
ADAM_LR           = 1e-4
GRAD_CLIP         = 1.0
GEN_TEMP          = 0.3
GEN_TOP_K         = 20
MAX_GEN_TOKENS    = 64
N_QUESTIONS_GEN   = 20
TEACHER_MODEL     = "google/gemini-2.5-flash-lite-preview-09-2025"
CONSOLIDATE_EVERY = 10
QGEN_EVERY        = 50   # generate new questions every N batches

# ── Loss functions ────────────────────────────────────────────────────────────

def masked_loss_fn(params, token_ids, loss_mask):
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    mask    = loss_mask[:, 1:].astype(jnp.float32)
    logits  = forward(params, inputs)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return (ce * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)


@jax.jit
def per_sample_losses_fn(params, token_ids, loss_mask):
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    mask   = loss_mask[:, 1:].astype(jnp.float32)
    logits = forward(params, inputs)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).reshape(token_ids.shape[0], -1)
    return (ce * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-8)


def make_train_step(optimizer):
    @jax.jit
    def train_step(carry, batch):
        params, opt_state = carry
        token_ids, loss_mask = batch
        loss, grads = jax.value_and_grad(masked_loss_fn)(params, token_ids, loss_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss
    return train_step

# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_pair(tok, question, answer):
    prompt   = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    full     = prompt + answer.lower() + "<|im_end|>"
    p_ids    = tok.encode(prompt).ids
    f_ids    = tok.encode(full).ids[:MAX_SEQ_LEN]
    pad      = MAX_SEQ_LEN - len(f_ids)
    ids      = f_ids + [PAD_ID] * pad
    plen     = min(len(p_ids), MAX_SEQ_LEN)
    mask     = [0] * plen + [1] * (MAX_SEQ_LEN - plen)
    for i in range(len(ids)):
        if ids[i] == PAD_ID and i >= plen: mask[i] = 0
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)


def batch_tokenize(tok, pairs):
    ids, masks = zip(*[tokenize_pair(tok, q, a) for q, a in pairs])
    return np.stack(ids), np.stack(masks)

# ── Batched generation ────────────────────────────────────────────────────────

def top_k_sample(logits, top_k, temperature, rng):
    logits = logits.astype(np.float32)
    if temperature > 0: logits = logits / temperature
    if top_k > 0:
        idx = np.argpartition(logits, -top_k)[-top_k:]
        m = np.full_like(logits, -1e9); m[idx] = logits[idx]; logits = m
    logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate_batch(params, tok, questions):
    B   = len(questions)
    rng = np.random.default_rng()
    eos_id = 2

    prompts   = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in questions]
    seqs      = np.full((B, MAX_SEQ_LEN), PAD_ID, dtype=np.int32)
    T_reals   = np.zeros(B, dtype=np.int32)
    generated = [[] for _ in range(B)]
    active    = np.ones(B, dtype=bool)

    for i, p in enumerate(prompts):
        ids = tok.encode(p).ids[:MAX_SEQ_LEN]
        seqs[i, :len(ids)] = ids
        T_reals[i] = len(ids)

    for step in range(MAX_GEN_TOKENS):
        if not active.any(): break
        all_logits = np.array(forward(params, jnp.array(seqs)))
        if step == 0: logging.info(f"    [gen] JIT compiled, batch_size={B}")
        for i in range(B):
            if not active[i]: continue
            tok_id = top_k_sample(all_logits[i, T_reals[i] - 1], GEN_TOP_K, GEN_TEMP, rng)
            if tok_id == eos_id or T_reals[i] >= MAX_SEQ_LEN - 1:
                active[i] = False; continue
            generated[i].append(tok_id)
            seqs[i, T_reals[i]] = tok_id
            T_reals[i] += 1

    return [tok.decode(g) for g in generated]

# ── External model ────────────────────────────────────────────────────────────

def _strip_fences(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return raw


def call_generate_questions(client, identity, facts):
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"
    user_msg    = fill(QGEN_USER, identity=identity, facts_block=facts_block, n_questions=N_QUESTIONS_GEN)
    for attempt in range(3):
        try:
            raw  = client.chat(user_msg, system=QGEN_SYSTEM, temperature=0.8, max_tokens=1024)
            data = json.loads(_strip_fences(raw))
            qs   = [str(q).strip() for q in data.get("questions", []) if q]
            if qs:
                logging.info(f"  [qgen] got {len(qs)} — first: \"{qs[0][:60]}\"")
                return qs
        except Exception as e:
            logging.warning(f"  [qgen] attempt {attempt+1}: {type(e).__name__}: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return []


def _interp_one(client, identity, facts_block, question, answer, idx, impersonate=False):
    if impersonate:
        system  = INTERP_IMPERSONATE_SYSTEM
        user_msg = fill(INTERP_IMPERSONATE_USER, identity=identity, facts_block=facts_block,
                        question=question, answer=answer)
    else:
        system  = INTERP_SYSTEM
        user_msg = fill(INTERP_USER, identity=identity, facts_block=facts_block,
                        question=question, answer=answer)
    for attempt in range(3):
        try:
            raw  = client.chat(user_msg, system=system, temperature=0.3, max_tokens=256, retries=0)
            data = json.loads(_strip_fences(raw))
            if "exact_sentence" in data:
                r = {"new_facts": data.get("new_facts", []), "exact_sentence": data["exact_sentence"]}
                logging.info(f"  [interp {idx}] → \"{r['exact_sentence'][:70]}\" | facts: {r['new_facts'][:1]}")
                return idx, r
        except Exception as e:
            logging.warning(f"  [interp {idx}] attempt {attempt+1}: {type(e).__name__}: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return idx, {"new_facts": [], "exact_sentence": "woof."}


def call_parallel_interp(executor, client, identity, facts_block, questions, answers, impersonate=False):
    futures = {
        executor.submit(_interp_one, client, identity, facts_block, q, a, i, impersonate): i
        for i, (q, a) in enumerate(zip(questions, answers))
    }
    results = [None] * len(questions)
    for fut in as_completed(futures):
        idx, r = fut.result()
        results[idx] = r
    return results


def consolidate_facts(client, identity, facts):
    facts_block = "\n".join(f"- {f}" for f in facts)
    user_msg    = fill(CONSOLIDATE_USER, identity=identity, facts_block=facts_block)
    for attempt in range(3):
        try:
            raw    = client.chat(user_msg, system=CONSOLIDATE_SYSTEM, temperature=0.2, max_tokens=2048)
            data   = json.loads(_strip_fences(raw))
            result = [str(f).strip() for f in data.get("facts", []) if f]
            logging.info(f"  [consolidate] {len(facts)} → {len(result)} facts")
            return result
        except Exception as e:
            logging.warning(f"  [consolidate] attempt {attempt+1}: {type(e).__name__}: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return facts

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_pairs():
    if not PAIRS_FILE.exists(): return []
    pairs = []
    with open(PAIRS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line); pairs.append((r["question"], r["exact_sentence"]))
            except Exception: pass
    return pairs


def append_pair(question, exact_sentence, batch_idx):
    with open(PAIRS_FILE, "a") as f:
        f.write(json.dumps({"question": question, "exact_sentence": exact_sentence,
                            "batch": batch_idx, "ts": time.time()}) + "\n")


def load_pair_losses():
    if not PAIR_LOSSES_FILE.exists(): return {}
    with open(PAIR_LOSSES_FILE) as f:
        return {int(k): v for k, v in json.load(f).items()}


def save_pair_losses(pair_losses):
    with open(PAIR_LOSSES_FILE, "w") as f:
        json.dump({str(k): v for k, v in pair_losses.items()}, f)


def sample_pairs(all_pairs, n, pair_losses):
    N = len(all_pairs)
    if N <= n: return list(all_pairs), list(range(N))
    # 1 lowest-loss pair (rehearsal) + n-1 most recent pairs
    known = {i: pair_losses[i] for i in range(N) if i in pair_losses}
    low_loss_idx = min(known, key=known.get) if known else 0
    recent = [i for i in range(max(0, N - (n - 1)), N) if i != low_loss_idx]
    idx = [low_loss_idx] + recent[-(n - 1):]
    return [all_pairs[i] for i in idx], idx


def load_all_questions():
    if not ALL_Q_FILE.exists(): return []
    qs = []
    with open(ALL_Q_FILE) as f:
        for line in f:
            try: qs.append(json.loads(line)["question"])
            except Exception: pass
    return qs


def append_questions(questions, batch_idx):
    with open(ALL_Q_FILE, "a") as f:
        for q in questions:
            f.write(json.dumps({"question": q, "batch": batch_idx, "ts": time.time()}) + "\n")


def pop_questions(n):
    """Pop up to n questions from pending queue."""
    try:
        with open(PENDING_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"questions": []}
    q = data.get("questions", [])
    chosen, remaining = q[:n], q[n:]
    with open(PENDING_FILE, "w") as f:
        json.dump({"questions": remaining}, f)
    return chosen


def push_questions(questions):
    """Push questions back to FRONT of pending (requeue unanswered ones)."""
    try:
        with open(PENDING_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"questions": []}
    existing = data.get("questions", [])
    with open(PENDING_FILE, "w") as f:
        json.dump({"questions": questions + existing}, f)


def refill_pending(n=20):
    all_qs = load_all_questions()
    if not all_qs: return
    weights = np.arange(1, len(all_qs) + 1, dtype=float) ** 2
    weights /= weights.sum()
    chosen = list(np.random.choice(all_qs, size=min(n, len(all_qs)), replace=False, p=weights))
    try:
        with open(PENDING_FILE) as f:
            existing = json.load(f).get("questions", [])
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    with open(PENDING_FILE, "w") as f:
        json.dump({"questions": existing + chosen}, f)


def pending_len():
    try:
        with open(PENDING_FILE) as f:
            return len(json.load(f).get("questions", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

# ── Reporting ─────────────────────────────────────────────────────────────────

def init_report():
    with open(REPORT_FILE, "w") as f:
        f.write(f"# Kylo Continual Learning — {RUN_TAG}\n\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"Teacher: `{TEACHER_MODEL}`  \n")
        f.write(f"Teacher mode: `{'impersonate' if args.impersonate else 'standard'}`  \n")
        f.write(f"Checkpoint in: `{CHECKPOINT_IN.name}`  \n")
        f.write(f"Batches: {N_BATCHES}\n\n---\n")


def report_batch(batch_idx, loss, n_pairs, questions, gen_answers, labels, new_facts_counts):
    sep = "━" * 50
    sample_q   = questions[0] if questions else ""
    sample_gen = gen_answers[0] if gen_answers else ""
    sample_lbl = labels[0] if labels else ""
    print(
        f"\n{sep}\n"
        f"Batch {batch_idx}/{N_BATCHES} | loss={loss:.4f} | pairs={n_pairs}\n"
        f'Q: "{sample_q[:70]}"\n'
        f'  Model → "{sample_gen[:60]}"\n'
        f'  Label → "{sample_lbl[:70]}"\n'
        f"  New facts this batch: {sum(new_facts_counts)}\n"
        f"{sep}"
    )
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Batch {batch_idx}/{N_BATCHES}\n\n")
        f.write(f"**Loss:** `{loss:.4f}` | **Pairs:** {n_pairs}\n\n")
        f.write("| Question | Model output | Teacher label | New facts |\n")
        f.write("|----------|--------------|---------------|-----------|\n")
        for q, gen, lbl, nf in zip(questions, gen_answers, labels, new_facts_counts):
            f.write(f"| {q[:70]} | {gen[:40]} | {lbl[:70]} | {nf} |\n")
        f.write("\n---\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info(f"JAX devices: {jax.devices()} | checkpoint_in={CHECKPOINT_IN.name} | batches={N_BATCHES}")
    tok      = load_tokenizer(TOKENIZER_FILE)
    identity = IDENTITY_FILE.read_text()

    if args.random_init:
        from shared_lib.random_utils import infinite_safe_keys_from_key
        key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(42))
        params  = init_params(key_gen)
        logging.info(f"Random init | params: {sum(v.size for v in jax.tree_util.tree_leaves(params)):,}")
    else:
        with open(CHECKPOINT_IN, "rb") as f:
            params = {k: jnp.array(v) for k, v in pickle.load(f).items()}
        logging.info(f"Loaded {CHECKPOINT_IN.name} | params: {sum(v.size for v in jax.tree_util.tree_leaves(params)):,}")

    # Load existing facts — consolidated_facts is the last compacted set, raw_facts accumulates since then
    consolidated_facts: list[str] = []
    if FACTS_FILE.exists():
        with open(FACTS_FILE) as f:
            consolidated_facts = json.load(f)
        logging.info(f"Loaded {len(consolidated_facts)} consolidated facts from {FACTS_FILE.name}")
    elif INTERP_FILE.exists():
        with open(INTERP_FILE) as f:
            for line in f:
                try: consolidated_facts.extend(json.loads(line).get("new_facts", []))
                except Exception: pass
        logging.info(f"Loaded {len(consolidated_facts)} facts from raw interp log")
    raw_facts: list[str] = []  # new facts since last consolidation

    pair_losses = load_pair_losses()
    optimizer   = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adamw(ADAM_LR, weight_decay=0.0))
    opt_state   = optimizer.init(params)
    train_step  = make_train_step(optimizer)
    client      = OpenRouterClient(model=TEACHER_MODEL)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_report()

    # Bootstrap questions if none exist yet
    if not load_all_questions():
        logging.info("Bootstrap: generating initial questions ...")
        qs = call_generate_questions(client, identity, consolidated_facts + raw_facts)
        append_questions(qs, batch_idx=0)
    if pending_len() < BATCH_SIZE:
        refill_pending(n=40)

    executor = ThreadPoolExecutor(max_workers=BATCH_SIZE + 2)
    consolidate_future: Future | None = None
    consolidate_snap: int = 0

    t_total = time.perf_counter()

    for batch_idx in range(1, N_BATCHES + 1):
        t_batch = time.perf_counter()

        # Apply consolidated facts if ready
        if consolidate_future is not None and consolidate_future.done():
            consolidated_facts = consolidate_future.result()
            raw_facts          = raw_facts[consolidate_snap:]  # keep facts added during consolidation
            consolidate_future = None
            logging.info(f"  [batch {batch_idx}] consolidated facts applied → {len(consolidated_facts)} + {len(raw_facts)} raw")
            with open(FACTS_FILE, "w") as f:
                json.dump(consolidated_facts, f)

        all_facts   = consolidated_facts + raw_facts
        facts_block = "\n".join(f"- {f}" for f in all_facts) if all_facts else "(none yet)"

        cold_start = batch_idx == 1 and not load_pairs()

        if cold_start:
            # No existing pairs: parallel bootstrap to collect BATCH_SIZE pairs at once
            questions = pop_questions(BATCH_SIZE)
            if len(questions) < BATCH_SIZE:
                refill_pending(n=40)
                questions += pop_questions(BATCH_SIZE - len(questions))
            if not questions:
                questions = load_all_questions()[-BATCH_SIZE:] or ["What do you like?"] * BATCH_SIZE
            logging.info(f"  [batch 1] cold-start parallel bootstrap: {len(questions)} pairs ...")

            t_gen = time.perf_counter()
            gen_answers = generate_batch(params, tok, questions)
            logging.info(f"  [batch 1] gen done in {time.perf_counter()-t_gen:.1f}s")

            t_interp = time.perf_counter()
            results  = call_parallel_interp(executor, client, identity, facts_block, questions, gen_answers, impersonate=args.impersonate)
            logging.info(f"  [batch 1] interp done in {time.perf_counter()-t_interp:.1f}s")

            labels, new_facts_counts = [], []
            for q, r in zip(questions, results):
                nf = r["new_facts"]
                labels.append(r["exact_sentence"])
                new_facts_counts.append(len(nf))
                raw_facts.extend(nf)
                append_pair(q, r["exact_sentence"], batch_idx)
                with open(INTERP_FILE, "a") as f:
                    f.write(json.dumps({"batch": batch_idx, "question": q,
                                        "exact_sentence": r["exact_sentence"],
                                        "new_facts": nf}) + "\n")
        else:
            # Batch 2+: 1 new pair per batch, same as original kylo pattern
            qs = pop_questions(1)
            if not qs:
                refill_pending(n=40)
                qs = pop_questions(1)
            question = qs[0] if qs else (load_all_questions() or ["What do you like?"])[-1]
            questions = [question]

            t_gen = time.perf_counter()
            gen_answers = generate_batch(params, tok, questions)
            logging.info(f"  [batch {batch_idx}] gen done in {time.perf_counter()-t_gen:.1f}s")

            idx, r   = _interp_one(client, identity, facts_block, question, gen_answers[0], 0, impersonate=args.impersonate)
            nf       = r["new_facts"]
            labels   = [r["exact_sentence"]]
            new_facts_counts = [len(nf)]
            all_facts.extend(nf)
            append_pair(question, r["exact_sentence"], batch_idx)
            with open(INTERP_FILE, "a") as f:
                f.write(json.dumps({"batch": batch_idx, "question": question,
                                    "exact_sentence": r["exact_sentence"],
                                    "new_facts": nf}) + "\n")
        # Refill pending in background if low
        if pending_len() < BATCH_SIZE * 2:
            executor.submit(refill_pending, 40)

        # Train step
        all_pairs = load_pairs()
        if all_pairs:
            sampled, sampled_idx = sample_pairs(all_pairs, TRAIN_BATCH_SIZE, pair_losses)
            ids_np, mask_np      = batch_tokenize(tok, sampled)
            ids_jnp, mask_jnp   = jnp.array(ids_np), jnp.array(mask_np)
            t_train = time.perf_counter()
            (params, opt_state), loss = train_step((params, opt_state), (ids_jnp, mask_jnp))
            loss_val = float(loss)
            sample_losses = np.array(per_sample_losses_fn(params, ids_jnp, mask_jnp))
            for gi, sl in zip(sampled_idx, sample_losses):
                pair_losses[gi] = float(sl)
            save_pair_losses(pair_losses)
            logging.info(f"  [batch {batch_idx}] train in {time.perf_counter()-t_train:.1f}s — loss={loss_val:.4f}")
        else:
            loss_val = float("nan")

        report_batch(batch_idx, loss_val, len(all_pairs),
                     questions, gen_answers, labels, new_facts_counts)

        logging.info(f"  [batch {batch_idx}] done in {time.perf_counter()-t_batch:.1f}s  "
                     f"elapsed={time.perf_counter()-t_total:.0f}s  facts={len(all_facts)}")

        # Periodic question generation
        if batch_idx % QGEN_EVERY == 0:
            executor.submit(lambda: (
                append_questions(call_generate_questions(client, identity, consolidated_facts + raw_facts), batch_idx),
                refill_pending(40),
            ))
            logging.info(f"  [batch {batch_idx}] qgen submitted")

        # Async fact consolidation
        if batch_idx % CONSOLIDATE_EVERY == 0 and consolidate_future is None and raw_facts:
            consolidate_snap   = len(raw_facts)
            consolidate_future = executor.submit(consolidate_facts, client, identity, consolidated_facts + raw_facts[:consolidate_snap])
            logging.info(f"  [batch {batch_idx}] consolidation submitted ({len(consolidated_facts)} consolidated + {consolidate_snap} new)")

    # Flush consolidation
    if consolidate_future is not None:
        consolidated_facts = consolidate_future.result()
        raw_facts          = raw_facts[consolidate_snap:]
        with open(FACTS_FILE, "w") as f:
            json.dump(consolidated_facts, f)
    all_facts = consolidated_facts + raw_facts

    executor.shutdown(wait=False)

    # Save checkpoint
    with open(CHECKPOINT_OUT, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"Checkpoint saved → {CHECKPOINT_OUT}")

    # Post-training eval
    eval_qs = [
        "What do you do when you're bored?",
        "Do you like cats?",
        "What's your favourite thing to do outside?",
        "Are you a good boy?",
        "What do you do when your owner comes home?",
    ]
    sep = "━" * 50
    print(f"\n{sep}\nPOST-TRAINING EVAL\n{sep}")
    eval_lines = ["\n## Post-Training Eval\n"]
    for q in eval_qs:
        ans = generate_batch(params, tok, [q])[0]
        print(f"Q: {q}\nA: {ans}\n")
        eval_lines.append(f"**Q:** {q}  \n**A:** _{ans}_\n")

    total_time = time.perf_counter() - t_total
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Summary\n\n"
                f"Checkpoint in: `{CHECKPOINT_IN.name}`  \n"
                f"Checkpoint out: `{CHECKPOINT_OUT.name}`  \n"
                f"Total pairs: {len(load_pairs())}  \nTotal facts: {len(all_facts)}  \n"
                f"Total time: {total_time:.0f}s\n\n")
        f.write("\n".join(eval_lines))

    logging.info(f"Done. pairs={len(load_pairs())}  facts={len(all_facts)}  time={total_time:.0f}s")
    logging.info(f"Next run: uv run python projects/small_lm/babystep/exp_kylo_continual.py --from {CHECKPOINT_OUT}")

    # Upload
    try:
        from shared_lib.report import save_report_file
        url = save_report_file(f"small_lm_babystep_kylo_continual_{RUN_TAG}", REPORT_FILE)
        logging.info(f"Report uploaded → {url}")
        print(f"\nReport: {url}")
    except Exception as e:
        logging.warning(f"Upload failed: {e}")

    client.close()


if __name__ == "__main__":
    main()
