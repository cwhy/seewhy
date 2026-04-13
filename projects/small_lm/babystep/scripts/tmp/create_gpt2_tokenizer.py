"""
create_gpt2_tokenizer.py — Fetch GPT-2 tokenizer from HuggingFace and add special tokens.

Final vocab size: 50260
  50257: <pad>
  50258: <|im_start|>
  50259: <|im_end|>

Run from repo root:
    uv run python projects/small_lm/babystep/scripts/tmp/create_gpt2_tokenizer.py
"""

from pathlib import Path
from tokenizers import Tokenizer

out_path = Path(__file__).parent.parent.parent.parent / "data" / "tokenizer_gpt2.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

tok = Tokenizer.from_pretrained("gpt2")
tok.add_special_tokens(["<pad>", "<|im_start|>", "<|im_end|>"])

tok.save(str(out_path))
print(f"Saved to {out_path}")
print(f"Vocab size: {tok.get_vocab_size()}")  # should be 50260
assert tok.get_vocab_size() == 50260, f"Unexpected vocab size: {tok.get_vocab_size()}"
print("OK — vocab size is 50260")
