"""
exp_kylo_factsbased.py — Experiment-10: saturation metric + batched consolidation.

Key changes from exp-9:
- Saturation metric: track β (logit inverse temperature) per batch; fit rolling
  power-law exponent.  β > c₀ ≈ 4.08 (vocab=4096) → peaked/saturated regime.
  Based on arxiv:2602.03685: in that regime loss ∝ τ^(−1/3).
- Batched facts consolidation: _extract_facts returns list[str] of new facts;
  when multiple futures resolve in the same batch, ONE consolidation call handles
  all new facts instead of N separate calls.

Usage:
    mkdir -p projects/small_lm/babystep/experiment-10-kylo-saturation
    uv run python projects/small_lm/babystep/exp_kylo_factsbased.py \\
      --random-init \\
      --batches 3000 \\
      --out-dir projects/small_lm/babystep/experiment-10-kylo-saturation \\
      > projects/small_lm/babystep/experiment-10-kylo-saturation/run_$(date +%Y%m%d_%H%M%S).log 2>&1

Pipeline per batch N:
  1. Harvest done facts futures (list[str]) → batch-consolidate once if any new
  2. Submit teacher call + gen for batch N+1  (hides ~15s latency behind train step)
  3. Block on batch N's teacher label
  4. append_pair → submit _extract_facts async (no consolidation inside)
  5. train_step (~2s) → unpack (loss, beta)
  6. Compute tau, rolling power-law exponent; log to gen_log.jsonl
  7. Every QGEN_EVERY batches: submit qgen async
"""

import argparse, functools, json, logging, math, pickle, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax

# ── Paths ─────────────────────────────────────────────────────────────────────

SMALL_LM  = next(p for p in Path(__file__).resolve().parents if p.name == "small_lm")
REPO_ROOT = SMALL_LM.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SMALL_LM))

from lib.data_processing import load_tokenizer
from lib.prompts import fill
from ai_eval.openrouter import OpenRouterClient

import importlib.util

# ── Prompts (inline) ──────────────────────────────────────────────────────────

# Label generation: teacher voices character from identity + facts only (no model output)
LABEL_SYSTEM = (
    "You are voicing a fictional character exactly as they would speak. "
    "Stay strictly in-character: use their vocabulary, their emotional register, "
    "their world-knowledge. Return valid JSON only."
)
LABEL_USER = """\
{{identity}}

Known facts about this character:
{{facts_block}}

Question asked to the character: {{question}}

Write the exact sentence this character would say in response. \
Keep it short (1-2 sentences), natural, and entirely in their voice. \
Do not add explanation or narration.

Return JSON only: {"exact_sentence": "..."}"""

# Facts: two-step update (extract then consolidate)
FACTS_SYSTEM = "You are maintaining a concise, accurate character profile. Return valid JSON only."

FACTS_EXTRACT_USER = """\
{{identity}}

Current known facts about this character:
{{facts_block}}

The character had these recent exchanges (where given, confidence level \
1=high / 4=low reflects how well the model has learned that response):
{{exchanges_block}}

Extract up to 5 new facts revealed in these exchanges that are NOT already \
covered by the current facts above. Write each fact so it can be understood \
at any education level. If nothing new, return an empty list.

Return JSON only: {"new_facts": ["..."]}"""

FACTS_CONSOLIDATE_USER = """\
{{identity}}

Current known facts about this character:
{{facts_block}}

New facts to incorporate:
{{new_facts_block}}

Merge the new facts into the existing list: remove duplicates, resolve \
contradictions in favour of the newer fact, keep at most 64 facts. \
Write all facts so they can be understood at any education level.

Return JSON only: {"facts": ["..."]}"""

# Question generation
QGEN_SYSTEM = (
    'You are a curious interviewer talking with a fictional character. '
    'Return valid JSON only.'
)
QGEN_USER = """\
{{identity}}

Known facts about this character:
{{facts_block}}

Generate {{n_questions}} natural, conversational questions to ask this character directly. \
Vary the topics and stay grounded in everyday situations.

Return JSON only: {"questions": ["...", ...]}"""

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

parser = argparse.ArgumentParser(description="Kylo facts-based teacher experiment (exp-10)")
parser.add_argument("--from", dest="checkpoint_in", default=None,
                    help="Input checkpoint (.pkl). Defaults to k-Dyck phase-1 checkpoint.")
parser.add_argument("--batches", type=int, default=5000, help="Number of batches to run.")
parser.add_argument("--out-dir", dest="out_dir", default=None,
                    help="Experiment output directory.")
parser.add_argument("--random-init", dest="random_init", action="store_true",
                    help="Initialize params randomly instead of loading a checkpoint.")
parser.add_argument("--tokenizer", choices=["custom", "gpt2"], default="custom",
                    help="Tokenizer to use: 'custom' (2375-token BPE) or 'gpt2' (50260-token).")
args = parser.parse_args()

# ── Dynamic model + tokenizer selection ───────────────────────────────────────

if args.tokenizer == "gpt2":
    _model_file    = SMALL_LM / "exp_kdyck_rope_50k.py"
    TOKENIZER_FILE = SMALL_LM / "data" / "tokenizer_gpt2.json"
else:
    _model_file    = SMALL_LM / "exp_kdyck_rope.py"
    TOKENIZER_FILE = SMALL_LM / "data" / "tokenizer.json"

_spec = importlib.util.spec_from_file_location("_model", _model_file)
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
forward     = _mod.forward
init_params = _mod.init_params
MAX_SEQ_LEN = _mod.MAX_SEQ_LEN
PAD_ID      = _mod.PAD_ID
EOS_ID      = _mod.EOS_ID
VOCAB_SIZE  = _mod.VOCAB_SIZE

# ── Saturation constants (arxiv:2602.03685) ───────────────────────────────────
BETA_C0 = math.sqrt(2 * math.log(VOCAB_SIZE))  # ≈ 4.08 for 4096-vocab, ≈ 4.65 for 50260

IDENTITY_FILE  = SMALL_LM / "data" / "identity" / "kylo.md"
CHARACTER_NAME = IDENTITY_FILE.stem  # e.g. "kylo"
KDYCK_CKPT    = SMALL_LM / "pkl" / "params_exp_kdyck_rope_phase1.pkl"

OUT_DIR          = Path(args.out_dir) if args.out_dir else SMALL_LM / "babystep" / "experiment-10-kylo-saturation"
CHECKPOINT_IN    = Path(args.checkpoint_in) if args.checkpoint_in else KDYCK_CKPT
CHECKPOINT_OUT   = OUT_DIR / "checkpoint.pkl"
PAIRS_FILE       = OUT_DIR / "pairs.jsonl"
PAIR_LOSSES_FILE = OUT_DIR / "pair_losses.json"
ALL_Q_FILE       = OUT_DIR / "all_questions.jsonl"
PENDING_FILE     = OUT_DIR / "pending_questions.json"
FACTS_FILE       = OUT_DIR / "facts.json"           # consolidated facts (source of truth)
FACTS_RAW_FILE   = OUT_DIR / "facts_raw.jsonl"      # append-only raw extracted facts log
GEN_LOG_FILE     = OUT_DIR / "gen_log.jsonl"
RUN_TAG          = time.strftime("%Y%m%d_%H%M%S")
REPORT_FILE      = OUT_DIR / f"report_{RUN_TAG}.md"

N_BATCHES        = args.batches
BATCH_SIZE       = 16          # used for cold-start refill sizing
TRAIN_BATCH_SIZE = 16
ADAM_LR          = 1e-4
GRAD_CLIP        = 1.0
GEN_TEMP         = 0.3
GEN_TOP_K        = 20
MAX_GEN_TOKENS   = 64
N_QUESTIONS_GEN  = 20
TEACHER_MODEL    = "google/gemini-2.5-flash-lite-preview-09-2025"
QGEN_EVERY       = 50
FACTS_CAP        = 64          # max facts in consolidated list (enforced by prompt)
FACTS_UPDATE_EVERY = 10        # submit a facts update call once per N batches
CKPT_EVERY       = 100         # save checkpoint every N batches

# ── Loss functions ────────────────────────────────────────────────────────────

def masked_loss_fn(params, token_ids, loss_mask):
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    mask   = loss_mask[:, 1:].astype(jnp.float32)
    logits = forward(params, inputs)             # (B, T-1, V)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss = (ce * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)
    # Beta: masked mean std over vocab (response positions only)
    beta_pos = jnp.std(logits, axis=-1)          # (B, T-1)
    beta = (beta_pos * mask).sum() / (mask.sum() + 1e-8)
    return loss, beta                            # has_aux=True


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
        (loss, beta), grads = jax.value_and_grad(masked_loss_fn, has_aux=True)(
            params, token_ids, loss_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), (loss, beta)
    return train_step

# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_pair(tok, question, answer):
    prompt = f"<|im_start|>interviewer\n{question}<|im_end|>\n<|im_start|>{CHARACTER_NAME}\n"
    full   = prompt + answer + "<|im_end|>"
    p_ids  = tok.encode(prompt).ids[:MAX_SEQ_LEN]
    f_ids  = tok.encode(full).ids[:MAX_SEQ_LEN]
    pad    = MAX_SEQ_LEN - len(f_ids)
    ids    = f_ids + [PAD_ID] * pad
    plen   = min(len(p_ids), MAX_SEQ_LEN)
    mask   = [0] * plen + [1] * (MAX_SEQ_LEN - plen)
    for i in range(len(ids)):
        if ids[i] == PAD_ID and i >= plen: mask[i] = 0
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)


def batch_tokenize(tok, pairs):
    ids, masks = zip(*[tokenize_pair(tok, q, a) for q, a in pairs])
    return np.stack(ids), np.stack(masks)

# ── JAX Generation (lax.scan) ─────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def generate_one_jit(max_gen_tokens, top_k, temperature, eos_id, pad_id,
                     params, prompt_ids, prompt_len, rng_key):
    """Single-sequence generation via lax.scan — eliminates Python loop over tokens."""
    def body(carry, _):
        seq, t, rng, done = carry
        logits = forward(params, seq[None])[0, t - 1] / temperature
        kth = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits >= kth, logits, -1e9)
        rng, sub = jax.random.split(rng)
        tok_id = jax.random.categorical(sub, logits)
        done = done | (tok_id == eos_id)
        tok_id = jnp.where(done, pad_id, tok_id)
        seq = seq.at[t].set(tok_id)
        return (seq, t + 1, rng, done), None

    (seq, _, _, _), _ = jax.lax.scan(
        body, (prompt_ids, prompt_len, rng_key, jnp.bool_(False)), None,
        length=max_gen_tokens)
    return seq


def _gen_one_seq(params, tok, question, rng_key):
    prompt = f"<|im_start|>interviewer\n{question}<|im_end|>\n<|im_start|>{CHARACTER_NAME}\n"
    p_ids  = tok.encode(prompt).ids[:MAX_SEQ_LEN]
    seq_np = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.int32)
    seq_np[:len(p_ids)] = p_ids
    result = generate_one_jit(
        MAX_GEN_TOKENS, GEN_TOP_K, GEN_TEMP, EOS_ID, PAD_ID,
        params, jnp.array(seq_np), jnp.int32(len(p_ids)), rng_key,
    )
    gen_ids = []
    for t in range(len(p_ids), MAX_SEQ_LEN):
        tid = int(result[t])
        if tid == PAD_ID or tid == EOS_ID: break
        gen_ids.append(tid)
    return tok.decode(gen_ids)


def generate_batch(params, tok, questions, rng_key=None):
    """Generate for a list of questions. Calls generate_one_jit per sequence."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(int(time.time()) & 0xFFFFFFFF)
    keys = jax.random.split(rng_key, len(questions))
    results = []
    for i, q in enumerate(questions):
        results.append(_gen_one_seq(params, tok, q, keys[i]))
        if i == 0:
            logging.info(f"    [gen] first seq done (lax.scan JIT compiled)")
    return results

# ── External model calls ──────────────────────────────────────────────────────

def _strip_fences(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return raw


def _label_one(client, identity, facts_block, question, idx):
    """Teacher: generate a clean in-character label (no model output)."""
    user_msg = fill(LABEL_USER, identity=identity, facts_block=facts_block, question=question)
    for attempt in range(3):
        try:
            raw  = client.chat(user_msg, system=LABEL_SYSTEM, temperature=0.3, max_tokens=256, retries=0)
            data = json.loads(_strip_fences(raw))
            if "exact_sentence" in data:
                sentence = data["exact_sentence"].strip()
                logging.info(f"  [label {idx}] \"{sentence[:70]}\"")
                return idx, sentence
        except Exception as e:
            logging.warning(f"  [label {idx}] attempt {attempt+1}: {type(e).__name__}: {e}")
            user_msg += "\n\nReturn valid JSON only."
    return idx, "..."


def _update_facts_from_buffer(client, identity, current_facts, pairs_buffer, pair_losses_snap, idx):
    """Two-step facts update from a buffer of (pair_idx, question, label) tuples.

    Step 1: extract up to 5 new facts from the buffer exchanges.
    Step 2: consolidate new facts into the existing list.
    Returns the new consolidated list. Falls back to current_facts on failure.
    """
    facts_block = "\n".join(f"- {f}" for f in current_facts) if current_facts else "(none yet)"
    lines = []
    for pair_idx, q, a in pairs_buffer:
        tier = get_confidence_tier(pair_idx, pair_losses_snap)
        tier_str = f"  [confidence: {tier}/4]" if tier is not None else ""
        lines.append(f'Q: "{q}"\nA: "{a}"{tier_str}')
    exchanges_block = "\n\n".join(lines)

    # Step 1: extract new facts
    new_facts = []
    msg1 = fill(FACTS_EXTRACT_USER, identity=identity, facts_block=facts_block,
                exchanges_block=exchanges_block)
    for attempt in range(3):
        try:
            raw  = client.chat(msg1, system=FACTS_SYSTEM, temperature=0.2, max_tokens=512, retries=0)
            data = json.loads(_strip_fences(raw))
            new_facts = [str(f).strip() for f in data.get("new_facts", []) if f]
            logging.info(f"  [facts {idx}] extracted {len(new_facts)} new facts from {len(pairs_buffer)} pairs")
            break
        except Exception as e:
            logging.warning(f"  [facts {idx}] extract attempt {attempt+1}: {type(e).__name__}: {e}")
            msg1 += "\n\nReturn valid JSON only."

    if not new_facts:
        return current_facts

    # Step 2: consolidate
    new_facts_block = "\n".join(f"- {f}" for f in new_facts)
    msg2 = fill(FACTS_CONSOLIDATE_USER, identity=identity, facts_block=facts_block,
                new_facts_block=new_facts_block)
    for attempt in range(3):
        try:
            raw    = client.chat(msg2, system=FACTS_SYSTEM, temperature=0.2, max_tokens=8192, retries=0)
            data   = json.loads(_strip_fences(raw))
            result = [str(f).strip() for f in data.get("facts", []) if f]
            logging.info(f"  [facts {idx}] consolidated {len(current_facts)} + {len(new_facts)} → {len(result)} facts")
            return result
        except Exception as e:
            logging.warning(f"  [facts {idx}] consolidate attempt {attempt+1}: {type(e).__name__}: {e}")
            msg2 += "\n\nReturn valid JSON only."

    return current_facts + new_facts  # best-effort fallback


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


def get_confidence_tier(pair_idx, pair_losses):
    """1=highest confidence (lowest loss), 4=lowest confidence. None if unrated."""
    if pair_idx not in pair_losses or len(pair_losses) < 4:
        return None
    losses = list(pair_losses.values())
    q25, q50, q75 = float(np.percentile(losses, 25)), float(np.percentile(losses, 50)), float(np.percentile(losses, 75))
    v = pair_losses[pair_idx]
    if v <= q25: return 1
    elif v <= q50: return 2
    elif v <= q75: return 3
    else: return 4


def sample_pairs(all_pairs, n, pair_losses):
    N = len(all_pairs)
    if N == 0:
        return [], []
    if N <= n:
        indices = list(range(N))
        while len(indices) < n:
            indices = indices + indices
        return [all_pairs[i] for i in indices[:n]], indices[:n]

    half = n // 2

    # Half: newest pairs
    newest_idx = list(range(N - half, N))

    # Half: lowest-loss pairs (exclude unrated; fallback to newest if insufficient)
    rated = sorted(
        [(i, pair_losses[i]) for i in range(N) if i in pair_losses],
        key=lambda x: x[1]
    )
    lowest_idx = [i for i, _ in rated[:half]]
    if len(lowest_idx) < half:
        # Not enough rated pairs yet — fill with newest
        lowest_idx = list(range(max(0, N - n), N - half))

    # Deduplicate while preserving order, fill to exactly n
    seen = set()
    idx = []
    for i in lowest_idx + newest_idx:
        if i not in seen:
            seen.add(i)
            idx.append(i)
    # Pad if dedup reduced count
    while len(idx) < n:
        idx.append(idx[-1])

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


def load_facts():
    """Load consolidated facts from facts.json."""
    if not FACTS_FILE.exists(): return []
    with open(FACTS_FILE) as f:
        return json.load(f)


def save_facts(facts):
    with open(FACTS_FILE, "w") as f:
        json.dump(facts, f)

# ── Reporting ─────────────────────────────────────────────────────────────────

def init_report():
    with open(REPORT_FILE, "w") as f:
        f.write(f"# Kylo Saturation Metric (exp-10) — {RUN_TAG}\n\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"Teacher: `{TEACHER_MODEL}`  \n")
        f.write(f"Teacher mode: facts-based (no model output)  \n")
        f.write(f"Checkpoint in: `{CHECKPOINT_IN.name}`  \n")
        f.write(f"Batches: {N_BATCHES}\n\n---\n")


def report_batch(batch_idx, elapsed, loss, n_pairs, question, label, gen_out,
                 n_facts_total, beta_val=float("nan"), pl_exponent=float("nan")):
    sat_str = "sat" if beta_val > BETA_C0 else "not saturated"
    pl_str  = f"{pl_exponent:.3f}" if not math.isnan(pl_exponent) else "n/a"
    sep = "━" * 50
    print(
        f"\n{sep}\n"
        f"Batch {batch_idx}/{N_BATCHES} | {elapsed:.1f}s | loss={loss:.4f} | "
        f"pairs={n_pairs} | facts={n_facts_total}\n"
        f"β={beta_val:.3f} (c₀={BETA_C0:.2f}, {sat_str}) | τ^exp={pl_str}\n"
        f'Q:     "{question[:70]}"\n'
        f'Label: "{label[:70]}"\n'
        f'Model: "{gen_out[:60]}"\n'
        f"{sep}"
    )
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Batch {batch_idx}/{N_BATCHES}\n\n")
        f.write(f"**Loss:** `{loss:.4f}` | **β:** `{beta_val:.3f}` ({sat_str}) | "
                f"**τ^exp:** `{pl_str}` | **Pairs:** {n_pairs} | "
                f"**Facts:** {n_facts_total} | **Time:** {elapsed:.1f}s\n\n")
        f.write(f"**Q:** {question[:80]}  \n")
        f.write(f"**Label:** {label[:100]}  \n")
        f.write(f"**Model:** {gen_out[:80]}  \n\n---\n")

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

    # Load consolidated facts
    all_facts: list[str] = load_facts()
    logging.info(f"Loaded {len(all_facts)} consolidated facts from {FACTS_FILE.name}")

    pair_losses = load_pair_losses()
    optimizer   = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adamw(ADAM_LR, weight_decay=0.0))
    opt_state   = optimizer.init(params)
    train_step  = make_train_step(optimizer)
    client      = OpenRouterClient(model=TEACHER_MODEL)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot this script into the experiment folder for reproducibility.
    # Skip if we are already running as the snapshot (i.e. launched from OUT_DIR).
    import shutil
    _this = Path(__file__).resolve()
    if _this.parent.resolve() != OUT_DIR.resolve():
        shutil.copy2(_this, OUT_DIR / "exp_snapshot.py")

    init_report()

    # Bootstrap questions if none exist
    if not load_all_questions():
        logging.info("Bootstrap: generating initial questions ...")
        qs = call_generate_questions(client, identity, all_facts)
        append_questions(qs, batch_idx=0)
    if pending_len() < BATCH_SIZE:
        refill_pending(n=40)

    # Load existing pairs into memory (avoid re-reading file every batch)
    all_pairs = load_pairs()
    logging.info(f"Loaded {len(all_pairs)} existing pairs")

    executor = ThreadPoolExecutor(max_workers=8)

    # Facts futures return list[str] — the full updated consolidated facts list
    pending_facts_futures: list[Future] = []

    # Buffer of (pair_idx, question, label) tuples accumulated since last facts update submission
    pairs_buffer: list[tuple[int, str, str]] = []

    # Rolling window for power-law exponent fit: stores (log_tau, log_loss)
    loss_history: deque = deque(maxlen=200)

    # All batch losses for end-of-run plot
    batch_losses: list[float] = []

    def _pop_or_fallback():
        qs = pop_questions(1)
        if not qs:
            refill_pending(n=40)
            qs = pop_questions(1)
        return qs[0] if qs else (load_all_questions() or ["What do you like?"])[-1]

    def _facts_block_from(facts):
        return "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"

    # ── Pipeline startup: submit teacher call + gen for batch 1 ───────────────
    question_cur  = _pop_or_fallback()
    facts_blk_cur = _facts_block_from(all_facts)
    base_rng      = jax.random.PRNGKey(int(time.time()) & 0xFFFFFFFF)

    teacher_future_cur: Future | None = executor.submit(
        _label_one, client, identity, facts_blk_cur, question_cur, 1)
    params_snap = jax.tree_util.tree_map(np.array, params)
    gen_future_cur: Future | None = executor.submit(
        generate_batch, params_snap, tok, [question_cur],
        jax.random.fold_in(base_rng, 1))

    t_total = time.perf_counter()

    for batch_idx in range(1, N_BATCHES + 1):
        t_batch = time.perf_counter()

        # ── Step 1: Harvest completed facts futures ────────────────────────────
        for fut in list(pending_facts_futures):
            if fut.done():
                try:
                    all_facts = fut.result()          # full updated consolidated list
                    save_facts(all_facts)
                except Exception as e:
                    logging.warning(f"  [batch {batch_idx}] facts future error: {e}")
                pending_facts_futures.remove(fut)

        facts_blk_cur = _facts_block_from(all_facts)

        # ── Step 2: Submit pipeline for batch N+1 ─────────────────────────────
        teacher_future_next: Future | None = None
        gen_future_next: Future | None     = None
        question_next: str | None          = None

        if batch_idx < N_BATCHES:
            question_next = _pop_or_fallback()
            if pending_len() < BATCH_SIZE * 2:
                executor.submit(refill_pending, 40)

            teacher_future_next = executor.submit(
                _label_one, client, identity, facts_blk_cur, question_next, batch_idx + 1)

            # Snapshot params before this batch's train step
            params_snap = jax.tree_util.tree_map(np.array, params)
            gen_future_next = executor.submit(
                generate_batch, params_snap, tok, [question_next],
                jax.random.fold_in(base_rng, batch_idx + 1))

        # ── Step 3: Block on current batch's teacher label ────────────────────
        _, label = teacher_future_cur.result()

        # ── Step 4: Record pair, accumulate buffer, submit facts update async ──
        append_pair(question_cur, label, batch_idx)
        all_pairs.append((question_cur, label))
        pair_idx = len(all_pairs) - 1
        pairs_buffer.append((pair_idx, question_cur, label))
        if batch_idx % FACTS_UPDATE_EVERY == 0:
            facts_snap  = list(all_facts)
            buffer_snap = list(pairs_buffer)
            losses_snap = dict(pair_losses)
            pairs_buffer.clear()
            facts_fut = executor.submit(
                _update_facts_from_buffer, client, identity, facts_snap, buffer_snap, losses_snap, batch_idx)
            pending_facts_futures.append(facts_fut)

        # ── Step 5: Train step ─────────────────────────────────────────────────
        loss_val    = float("nan")
        beta_val    = float("nan")
        pl_exponent = float("nan")
        if all_pairs:
            sampled, sampled_idx = sample_pairs(all_pairs, TRAIN_BATCH_SIZE, pair_losses)
            ids_np, mask_np     = batch_tokenize(tok, sampled)
            ids_jnp, mask_jnp  = jnp.array(ids_np), jnp.array(mask_np)
            t_train = time.perf_counter()
            (params, opt_state), (loss, beta_jax) = train_step(
                (params, opt_state), (ids_jnp, mask_jnp))
            loss_val = float(loss)
            beta_val = float(beta_jax)
            sample_losses = np.array(per_sample_losses_fn(params, ids_jnp, mask_jnp))
            for gi, sl in zip(sampled_idx, sample_losses):
                pair_losses[gi] = float(sl)
            save_pair_losses(pair_losses)

            # Saturation: rolling power-law exponent fit
            tau = batch_idx * ADAM_LR
            if loss_val > 0 and tau > 0:
                loss_history.append((math.log(tau), math.log(loss_val)))
            if len(loss_history) >= 10:
                log_taus, log_losses = zip(*loss_history)
                pl_exponent, _ = np.polyfit(log_taus, log_losses, 1)

            if not math.isnan(loss_val):
                batch_losses.append(loss_val)

            sat_str = "sat" if beta_val > BETA_C0 else "not sat"
            pl_str  = f"{pl_exponent:.3f}" if not math.isnan(pl_exponent) else "n/a"
            logging.info(f"  [batch {batch_idx}] train {time.perf_counter()-t_train:.1f}s "
                         f"loss={loss_val:.4f} β={beta_val:.3f} ({sat_str}) τ^exp={pl_str}")

        # ── Step 6: Log gen output ─────────────────────────────────────────────
        gen_out = ""
        if gen_future_cur is not None:
            try:
                gen_result = gen_future_cur.result()
                gen_out    = gen_result[0] if gen_result else ""
                with open(GEN_LOG_FILE, "a") as f:
                    f.write(json.dumps({
                        "batch": batch_idx, "question": question_cur,
                        "gen_out": gen_out, "label": label,
                        "beta": beta_val, "tau": batch_idx * ADAM_LR,
                        "pl_exponent": pl_exponent,
                        "ts": time.time(),
                    }) + "\n")
            except Exception as e:
                logging.warning(f"  [batch {batch_idx}] gen future error: {e}")

        elapsed = time.perf_counter() - t_batch
        report_batch(batch_idx, elapsed, loss_val, len(all_pairs),
                     question_cur, label, gen_out, len(all_facts),
                     beta_val=beta_val, pl_exponent=pl_exponent)
        logging.info(f"  [batch {batch_idx}] done in {elapsed:.1f}s  "
                     f"elapsed={time.perf_counter()-t_total:.0f}s  facts={len(all_facts)}")

        # ── Step 7: Periodic qgen ──────────────────────────────────────────────
        if batch_idx % QGEN_EVERY == 0:
            _facts_snap = list(all_facts)
            executor.submit(lambda b=batch_idx, fs=_facts_snap: (
                append_questions(call_generate_questions(client, identity, fs), b),
                refill_pending(40),
            ))
            logging.info(f"  [batch {batch_idx}] qgen submitted")

        # Periodic checkpoint
        if batch_idx % CKPT_EVERY == 0:
            with open(CHECKPOINT_OUT, "wb") as f:
                pickle.dump({k: np.array(v) for k, v in params.items()}, f)
            logging.info(f"  [batch {batch_idx}] checkpoint saved → {CHECKPOINT_OUT.name}")

        # ── Advance pipeline ───────────────────────────────────────────────────
        question_cur       = question_next
        teacher_future_cur = teacher_future_next
        gen_future_cur     = gen_future_next

    # ── Flush remaining facts futures ─────────────────────────────────────────
    for fut in pending_facts_futures:
        try:
            all_facts = fut.result(timeout=120)
        except Exception as e:
            logging.warning(f"  [flush] facts future error: {e}")
    save_facts(all_facts)

    executor.shutdown(wait=False)

    # ── Save final checkpoint ──────────────────────────────────────────────────
    with open(CHECKPOINT_OUT, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"Checkpoint saved → {CHECKPOINT_OUT}")

    # ── Post-training eval ────────────────────────────────────────────────────
    eval_qs = [
        "What do you do when you're bored?",
        "Do you like cats?",
        "What's your favourite thing to do outside?",
        "Are you a good boy?",
        "What do you do when your owner comes home?",
    ]
    sep = "━" * 50
    print(f"\n{sep}\nPOST-TRAINING EVAL\n{sep}")
    eval_rng = jax.random.PRNGKey(0)
    eval_lines = ["\n## Post-Training Eval\n"]
    for q in eval_qs:
        ans = generate_batch(params, tok, [q], eval_rng)[0]
        print(f"Q: {q}\nA: {ans}\n")
        eval_lines.append(f"**Q:** {q}  \n**A:** _{ans}_\n")

    total_time = time.perf_counter() - t_total
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n## Summary\n\n"
                f"Checkpoint in: `{CHECKPOINT_IN.name}`  \n"
                f"Checkpoint out: `{CHECKPOINT_OUT.name}`  \n"
                f"Total pairs: {len(all_pairs)}  \n"
                f"Total facts: {len(all_facts)}  \n"
                f"Total time: {total_time:.0f}s\n\n")
        f.write("\n".join(eval_lines))

    logging.info(f"Done. pairs={len(all_pairs)}  facts={len(all_facts)}  time={total_time:.0f}s")
    logging.info(f"Next run: uv run python projects/small_lm/babystep/exp_kylo_factsbased.py "
                 f"--from {CHECKPOINT_OUT} --out-dir {OUT_DIR} --batches 3000")

    # ── Saturation plot ───────────────────────────────────────────────────────
    try:
        import base64, io, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sat_entries = [e for e in (
            json.loads(l) for l in open(GEN_LOG_FILE)
        ) if "beta" in e and not math.isnan(e.get("beta", float("nan")))]

        if sat_entries:
            batches   = [e["batch"] for e in sat_entries]
            betas     = [e["beta"] for e in sat_entries]
            pl_exps   = [e.get("pl_exponent", float("nan")) for e in sat_entries]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.suptitle("Saturation Metric over Training")

            ax1.plot(batches, betas, lw=0.8, color="steelblue", label="β (logit std)")
            ax1.axhline(BETA_C0, color="red", lw=1, ls="--", label=f"c₀={BETA_C0:.2f}")
            ax1.set_ylabel("β")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            valid_pl = [(b, p) for b, p in zip(batches, pl_exps) if not math.isnan(p)]
            if valid_pl:
                pb, pp = zip(*valid_pl)
                ax2.plot(pb, pp, lw=0.8, color="darkorange", label="pl_exponent (rolling)")
                ax2.axhline(-1/3, color="red", lw=1, ls="--", label="−1/3 (theory)")
                ax2.set_ylabel("power-law exponent")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

            ax2.set_xlabel("batch")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120)
            plt.close()
            b64 = base64.b64encode(buf.getvalue()).decode()
            with open(REPORT_FILE, "a") as f:
                f.write(f"\n## Saturation Plot\n\n"
                        f"![saturation](data:image/png;base64,{b64})\n")
            logging.info("Saturation plot embedded in report")
    except Exception as e:
        logging.warning(f"Saturation plot failed: {e}")

    # ── Loss curve plot ───────────────────────────────────────────────────────
    if batch_losses:
        try:
            import base64, io, matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(range(1, len(batch_losses) + 1), batch_losses, lw=0.7, color="steelblue")
            ax.set_xlabel("batch"); ax.set_ylabel("loss"); ax.set_title("Training Loss Curve")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=120); plt.close()
            b64 = base64.b64encode(buf.getvalue()).decode()
            with open(REPORT_FILE, "a") as f:
                f.write(f"\n## Loss Curve\n\n![loss](data:image/png;base64,{b64})\n")
            logging.info("Loss curve embedded in report")
        except Exception as e:
            logging.warning(f"Loss curve plot failed: {e}")

    # Upload report
    try:
        from shared_lib.report import save_report_file
        url = save_report_file(f"small_lm_babystep_kylo_factsbased_{RUN_TAG}", REPORT_FILE)
        logging.info(f"Report uploaded → {url}")
        print(f"\nReport: {url}")
    except Exception as e:
        logging.warning(f"Upload failed: {e}")

    client.close()


if __name__ == "__main__":
    main()
