"""
One-shot setup: generate synthetic conversations + train BPE tokenizer.

Run once before any experiments:
    uv run python projects/small_lm/scripts/tmp/setup_data.py

Output:
    projects/small_lm/data/train.jsonl      (gitignored)
    projects/small_lm/data/eval.jsonl       (gitignored)
    projects/small_lm/data/tokenizer.json   (committed)
"""

import json
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.dataset         import save_data
from lib.data_processing import train_tokenizer

N_SAMPLES  = 60_000
EVAL_FRAC  = 0.05
VOCAB_SIZE = 4096
SEED       = 42

if __name__ == "__main__":
    print(f"Step 1/2 — Generating {N_SAMPLES:,} conversations ...")
    save_data(DATA_DIR, n=N_SAMPLES, eval_frac=EVAL_FRAC, seed=SEED)

    print(f"\nStep 2/2 — Training BPE tokenizer (vocab={VOCAB_SIZE}) ...")
    texts = []
    for split in ["train.jsonl", "eval.jsonl"]:
        with open(DATA_DIR / split) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
    print(f"  Loaded {len(texts):,} texts for tokenizer training")

    tok = train_tokenizer(texts, DATA_DIR / "tokenizer.json", vocab_size=VOCAB_SIZE)

    # Quick smoke test
    test = "<|im_start|>user\nhi guppy<|im_end|>\n<|im_start|>assistant\nhello.<|im_end|>"
    ids     = tok.encode(test).ids
    decoded = tok.decode(ids)
    print(f"\nSmoke test:")
    print(f"  input:   {test!r}")
    print(f"  n_tokens:{len(ids)}")
    print(f"  decoded: {decoded!r}")
    print(f"\nDone. Data at {DATA_DIR}")
