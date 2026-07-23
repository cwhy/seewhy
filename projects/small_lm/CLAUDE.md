# small_lm — Project Rules

## Python Execution
Always use `uv run python` — never `python3` or `python` directly. The system Python lacks all project deps.

## Scripts
Write throwaway/diagnostic scripts to `scripts/tmp/` and run them with `uv run python scripts/tmp/<name>.py`. Never use inline `python3 -c "..."` or `uv run python -c "..."`.

## Experiment Data — NEVER DELETE
Never delete experiment data files:
- `pairs.jsonl`, `pair_losses.json`
- `interpreted_facts.jsonl`, `facts.json`
- `all_questions.jsonl`, `pending_questions.json`
- Any `.pkl` checkpoint

These are irreplaceable training records that cost real money to generate. If a fresh start is needed, create a new experiment folder — never overwrite or clear an existing one.

## Running Experiments — NEVER RESTART WITHOUT EXPLICIT INSTRUCTION
Never kill or restart a running experiment (`pkill`, process kill, relaunch) unless the user explicitly says to do so.

If code changes are needed while an experiment is running, make the edits to the file and wait for the current run to finish naturally. Ask the user before killing any running process.

## Experiment Structure
Experiments live in `babystep/experiment-N-name/`. Each experiment folder is self-contained with its own data files and checkpoints. New variants always get a new folder — never reuse an existing one.

## Resuming Experiments
```bash
uv run python projects/small_lm/babystep/exp_kylo_continual.py \
  --from projects/small_lm/babystep/experiment-N-name/checkpoint.pkl \
  --out-dir projects/small_lm/babystep/experiment-N-name \
  --batches 100 \
  > projects/small_lm/babystep/experiment-N-name/run_$(date +%Y%m%d_%H%M%S).log 2>&1
```
