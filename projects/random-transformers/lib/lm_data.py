"""TinyStories: tokenizer, packed token stream, and sampling.

The paper trains a 10,000-token BPE tokenizer on the TinyStories training split
and uses a 512-token context. Both are reproduced here. Everything is cached to
disk under `lm_cache/` (gitignored) because tokenising is far slower than the
training runs it feeds, and every architecture in the sweep must see *the same*
token stream — otherwise the scaling curve compares data as well as models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

VOCAB = 10_000
CTX = 512
CACHE = Path(__file__).parent.parent / "lm_cache"
DATASET = "roneneldan/TinyStories"

# How much of the training split to tokenise. The paper does 5 epochs over all
# of it (~2.3B tokens); this is the declared budget deviation. Held fixed across
# every architecture so the sweep compares models, not data.
TRAIN_TOKENS = 100_000_000
EVAL_TOKENS = 2_000_000
TOKENIZER_SAMPLE = 200_000        # stories used to fit the BPE merges


def _load_split(split: str):
    from datasets import load_dataset
    return load_dataset(DATASET, split=split)


def tokenizer_path() -> Path:
    return CACHE / f"bpe_{VOCAB}.json"


def build_tokenizer():
    """Fit a byte-level BPE on the training split, or load the cached one."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    path = tokenizer_path()
    if path.exists():
        return Tokenizer.from_file(str(path))

    CACHE.mkdir(parents=True, exist_ok=True)
    logging.info(f"training a {VOCAB}-token BPE tokenizer")
    ds = _load_split("train")
    n = min(TOKENIZER_SAMPLE, len(ds))

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def it():
        for i in range(0, n, 1000):
            yield [t for t in ds[i:i + 1000]["text"]]

    tok.train_from_iterator(it(), trainer=trainer, length=n // 1000)
    tok.save(str(path))
    logging.info(f"tokenizer -> {path}")
    return tok


def _tokenize_split(tok, split: str, budget: int, out: Path) -> np.ndarray:
    if out.exists():
        return np.load(out, mmap_mode="r")

    logging.info(f"tokenising {split} to {budget:,} tokens")
    ds = _load_split(split)
    eot = tok.token_to_id("<|endoftext|>")
    chunks, total = [], 0
    for i in range(0, len(ds), 2000):
        texts = ds[i:i + 2000]["text"]
        for enc in tok.encode_batch(texts):
            chunks.append(np.asarray(enc.ids + [eot], dtype=np.uint16))
            total += len(enc.ids) + 1
        if total >= budget:
            break
    arr = np.concatenate(chunks)[:budget]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    logging.info(f"{split}: {arr.size:,} tokens -> {out}")
    return np.load(out, mmap_mode="r")


def load_tokens() -> tuple[np.ndarray, np.ndarray, object]:
    """``(train_tokens, eval_tokens, tokenizer)`` — flat uint16 streams."""
    tok = build_tokenizer()
    tr = _tokenize_split(tok, "train", TRAIN_TOKENS, CACHE / f"train_{TRAIN_TOKENS}.npy")
    ev = _tokenize_split(tok, "validation", EVAL_TOKENS, CACHE / f"eval_{EVAL_TOKENS}.npy")
    return tr, ev, tok


def to_contexts(stream: np.ndarray, ctx: int = CTX) -> np.ndarray:
    """Reshape a flat token stream into non-overlapping (n, ctx) contexts."""
    n = stream.size // ctx
    return np.asarray(stream[: n * ctx], dtype=np.int32).reshape(n, ctx)
