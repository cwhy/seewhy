# Project Start Guide

## 1. Create the directory

```bash
mkdir -p projects/{project-name}/scripts/tmp
mkdir -p projects/{project-name}/logs
mkdir -p projects/{project-name}/lib
touch projects/{project-name}/lib/__init__.py
touch projects/{project-name}/results.jsonl
```

## 2. Copy templates

```bash
# Project-specific templates (will be customized)
cp projects/TEMPLATES/v1/workflow.md projects/{project-name}/workflow.md

# Shared infrastructure (copy as-is, update docstring only)
cp shared_lib/templates/v1/run_experiments.py projects/{project-name}/scripts/run_experiments.py
cp shared_lib/templates/v1/poll_result.py     projects/{project-name}/scripts/poll_result.py
```

## 3. Customize

**`workflow.md`** — fill in every `{PROJECT_NAME}` / `{project-name}` placeholder,
write the one-sentence project description, and update the directory layout to
reflect actual files.

**`run_experiments.py`** — replace `{PROJECT_NAME}` / `{project-name}` in the
docstring. If the project has well-known metrics, override `print_results_table()`
with project-specific columns (see existing projects for examples).

**`poll_result.py`** — replace `{project-name}` in the docstring. The logic
is generic and usually needs no further changes.

## 4. Write the first experiment

Follow the file structure in `workflow.md`:

```
module docstring    — hypothesis, what changed, usage line
hyperparameters     — UPPER_SNAKE_CASE constants at module top
precomputed consts
utilities           — n_params, append_result
init_params()
forward() / encode() / decode()
loss_fn()
make_epoch_fn()     — returns jax.jit'd epoch fn using lax.scan
eval_metrics()
train()
__main__            — load data → init → train → visualize → append_result
```

## 5. Gitignore large files

Add to `.gitignore` (or the project's local ignore):

```
projects/{project-name}/logs/
projects/{project-name}/*.pkl
projects/{project-name}/*.npy
```

`results.jsonl` is the **only** committed output.

---

## Template version used

This project was set up with template **v1**.
Record this in `workflow.md` if you deviate from the template significantly.
