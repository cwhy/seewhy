"""A 100M-token training stream for backprop checkpoints, disjoint from val and test.

The paper measures gradient alignment on backprop-trained checkpoints at 10M, 100M and
1B tokens (Section 4.2, Table 8). 10M is already in `data/train_10M.npy`; this builds the
100M stream. It replays `prepare_data.py`'s document stream to the same place --
200,000 skipped documents, the ~20.5M-token pool, the validation and test splits -- and
keeps reading, so every document here comes *after* val and test and cannot overlap them.

    data/train_100M.npy   [~48,800, 2049]

Usage (on the box, from the repo root):
    ~/.local/bin/uv run --no-sync python projects/dust/scripts/prepare_extra.py
"""

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
SEQ = 2049


def main() -> None:
    from datasets import load_dataset

    corpus = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    tok = Tokenizer.from_file(str(PROJECT / "tokenizer" / "tokenizer.json"))
    end = tok.token_to_id("<|endoftext|>")
    docs = iter(corpus)
    for _ in range(200_000):
        next(docs)

    def take(min_tokens: int) -> int:
        n = 0
        while n < min_tokens:
            n += 1 + len(tok.encode(next(docs)["text"]).ids)
        return n

    # replay the pool and both held-out splits exactly as prepare_data.py consumed them
    for target in (20_500_000, 1_150_000, 1_150_000):
        take(target)
        print("skipped", target, flush=True)

    tokens: list[int] = []
    while len(tokens) < 100_000_000 + SEQ:
        tokens.append(end)
        tokens.extend(tok.encode(next(docs)["text"]).ids)
        if len(tokens) % 5_000_000 < 600:
            print(len(tokens), flush=True)
    usable = len(tokens) // SEQ * SEQ
    np.save(PROJECT / "data" / "train_100M.npy", np.asarray(tokens[:usable], dtype=np.uint16).reshape(-1, SEQ))
    print("done", usable // SEQ, "sequences", flush=True)


if __name__ == "__main__":
    main()
