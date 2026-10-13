"""Build the FineWeb token files the DUST paper trains on, as plain numpy.

The authors' `prepare_data.py` writes torch tensors; this project has no torch,
so the recipe is reproduced here and writes `.npy` instead. Nothing else is
changed: same corpus slice (HuggingFaceFW/fineweb, sample-10BT, first 200,000
documents skipped), same BPE-4096 tokenizer (`tokenizer/tokenizer.json`, from
github.com/qlabs-eng/dust, MIT), same split sizes. Training prefixes are cut
from one ~20.5M-token pool, so a smaller budget is always a prefix of a larger.

    data/train_100K.npy   48 sequences     (6 steps of 8)
    data/train_1M.npy    488 sequences     (61 steps of 8)
    data/train_10M.npy  4880 sequences     (610 steps of 8)
    data/val.npy, data/test.npy   544 sequences each

Every row is 2049 token ids: 2048 inputs plus the shifted target.

Usage (on the GPU box, from the repo root):
    ~/.local/bin/uv run --no-sync python projects/dust/scripts/prepare_data.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
SEQ = 2049
TRAIN_SEQUENCES = {"100K": 48, "1M": 488, "10M": 4880}
HELD_OUT = 544


def encode_split(documents, tokenizer, min_tokens: int) -> np.ndarray:
    tokens: list[int] = []
    end = tokenizer.token_to_id("<|endoftext|>")
    while len(tokens) < min_tokens:
        tokens.append(end)
        tokens.extend(tokenizer.encode(next(documents)["text"]).ids)
    usable = len(tokens) // SEQ * SEQ
    return np.asarray(tokens[:usable], dtype=np.uint16).reshape(-1, SEQ)


def main() -> None:
    from datasets import load_dataset   # not shared_lib.datasets: see gpu-box-stray-datasets-py

    DATA.mkdir(exist_ok=True)
    corpus = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                          split="train", streaming=True)
    tokenizer = Tokenizer.from_file(str(PROJECT / "tokenizer" / "tokenizer.json"))
    documents = iter(corpus)
    for _ in range(200_000):
        next(documents)

    pool = encode_split(documents, tokenizer, 20_500_000)
    for name, count in TRAIN_SEQUENCES.items():
        np.save(DATA / f"train_{name}.npy", pool[:count])
        print(f"train_{name}: {count} sequences", flush=True)
    del pool
    for split in ("val", "test"):
        rows = encode_split(documents, tokenizer, 1_150_000)
        np.save(DATA / f"{split}.npy", rows[:HELD_OUT])
        print(f"{split}: {HELD_OUT} sequences", flush=True)

    files = {}
    for p in sorted(DATA.glob("*.npy")):
        arr = np.load(p)
        files[p.name] = dict(sequences=len(arr), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (DATA / "manifest.json").write_text(json.dumps(dict(
        corpus="HuggingFaceFW/fineweb", subset="sample-10BT", skipped_documents=200_000,
        tokenizer_sha256=hashlib.sha256((PROJECT / "tokenizer" / "tokenizer.json").read_bytes()).hexdigest(),
        files=files), indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
