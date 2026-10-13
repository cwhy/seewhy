# Tokenizer

This folder contains everything needed to use or retrain the tokenizer:

- `tokenizer.json`: the saved tokenizer used by the experiments.
- `train.py`: the original BPE training recipe, extracted into a standalone script.

**Use the included JSON for normal training.** From the repository root, run
`python prepare_data.py`; it loads this file automatically. You do not need to
train or download a tokenizer separately.

## How it was trained

For the paper experiments, we trained a byte-level BPE tokenizer with Hugging
Face's `tokenizers` library on the **first 200,000 documents** of FineWeb's
`sample-10BT` subset (`HuggingFaceFW/fineweb`). The configuration is:

| Setting | Value |
| --- | --- |
| Vocabulary size | 4,096, including the special token |
| Initial alphabet | All 256 byte-level symbols |
| Pre-tokenization | Byte-level, with `add_prefix_space=False` |
| Special token | `<\|endoftext\|>` (ID 0 in the included artifact) |

BPE learns frequent byte-pair merges; it does not use backpropagation. The JSON
stores the resulting vocabulary and merge rules. The tokenizer stays fixed and
is shared by DUST and the backpropagation baseline.

Data preparation skips those 200,000 documents before collecting transformer
training data. The 1M/10M budgets count transformer training tokens; they do not
include the separate corpus used to train the tokenizer.

## Optional retraining

After following the installation instructions in the main README, run from the
repository root:

```bash
python tokenizer/train.py
```

This trains on the original 200,000-document recipe and writes
`tokenizer/retrained.json`, leaving the included tokenizer unchanged. It runs on
CPU and streams documents from Hugging Face. The script collects the training
texts in memory, as the original recipe did.

To select an output location or use local FineWeb shards:

```bash
python tokenizer/train.py --output tokenizer/retrained.json \
  --parquet /path/to/000_00000.parquet
```

`--documents` can be changed for exploratory runs. Changing it changes the
training recipe; data preparation still reserves the original 200,000-document
prefix. Retraining is not guaranteed to reproduce the exact saved JSON if the
source stream, its order, or library behavior changes. Keep the included artifact
when comparing with the paper. Normal data preparation always reads
`tokenizer/tokenizer.json`; it does not automatically use a retrained file.
