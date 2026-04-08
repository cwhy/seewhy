"""
Data processing for small_lm: tokenizer training/loading and token-array loading.

Public API:
    train_tokenizer(texts, save_path, vocab_size=4096) -> Tokenizer
    load_tokenizer(path) -> Tokenizer
    load_tokens(jsonl, tok, max_seq_len=128) -> np.ndarray  # (N, T) int32
"""

import json
import numpy as np
from pathlib import Path

# ── Special token IDs (must match tokenizer training order) ──────────────────

SPECIAL_TOKENS = ["<pad>", "<|im_start|>", "<|im_end|>"]
PAD_ID   = 0
BOS_ID   = 1   # <|im_start|>
EOS_ID   = 2   # <|im_end|>


def train_tokenizer(texts: list[str], save_path: Path, vocab_size: int = 4096):
    """Train a BPE tokenizer on the provided texts and save to save_path.

    Args:
        texts:      List of raw conversation strings (format_sample output).
        save_path:  Where to write tokenizer.json.
        vocab_size: Target vocabulary size (default 4096, matching GuppyLM).

    Returns:
        Trained tokenizers.Tokenizer object.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(texts):,} texts ...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved → {save_path}  ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def load_tokenizer(path: Path):
    """Load a saved tokenizer from a JSON file."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def load_tokens(
    jsonl: Path,
    tok,
    max_seq_len: int = 128,
) -> np.ndarray:
    """Tokenize all samples in a JSONL file.

    Each line must have a "text" field (the full conversation string).
    Sequences are truncated to max_seq_len and right-padded with PAD_ID.

    Returns:
        int32 array of shape (N, max_seq_len).
    """
    texts = []
    with open(jsonl) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    encodings = tok.encode_batch(texts)

    out = np.full((len(texts), max_seq_len), PAD_ID, dtype=np.int32)
    for i, enc in enumerate(encodings):
        ids = enc.ids[:max_seq_len]
        out[i, :len(ids)] = ids

    n_truncated = sum(len(enc.ids) > max_seq_len for enc in encodings)
    if n_truncated:
        print(f"  {n_truncated}/{len(texts)} sequences truncated to {max_seq_len} tokens")

    return out
