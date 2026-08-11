# Project Start Guide

Start-guide **v3**. Versioned artifacts (`workflow.md`, the runner scripts) stay
on **v1** — nothing here changes the workflow contract, so existing projects need
no migration. See "What changed in v3" at the bottom.

Set these once and paste the rest:

```bash
NAME=my-project                 # directory name, kebab-case
TITLE="My Project"              # human-readable, used in docstrings/headings
```

## 1. Scaffold

```bash
mkdir -p projects/$NAME/scripts/tmp projects/$NAME/logs projects/$NAME/lib projects/$NAME/reports
touch projects/$NAME/lib/__init__.py projects/$NAME/results.jsonl
cat > projects/$NAME/.gitignore <<'EOF'
logs/
scripts/tmp/
*.pkl
*.npy
*.json
!results.jsonl
EOF
```

Do **not** add `paper/` exclusions here — `scaffold_paper()` writes a nested
`paper/.gitignore` that re-includes the figure data this one would otherwise
swallow. A rule here would win over it for whole directories.

The local `.gitignore` is deliberate: `*.pkl` and `*.log` are already covered
globally, but a per-project file keeps the project self-contained and stops the
root `.gitignore` from growing a hand-written block per project (it has two
already). **`*.json` is the one that matters** — intermediate sweep and probe
dumps are what actually leak: `universal-ar` has 7 untracked JSON dumps in
`git status` right now.

## 2. Copy templates

```bash
cp projects/TEMPLATES/v1/workflow.md          projects/$NAME/workflow.md
cp shared_lib/templates/v1/run_experiments.py projects/$NAME/scripts/run_experiments.py
cp shared_lib/templates/v1/poll_result.py     projects/$NAME/scripts/poll_result.py
cp shared_lib/templates/v1/gen_report.py      projects/$NAME/scripts/gen_report.py
cp shared_lib/templates/v1/figures.py         projects/$NAME/lib/figures.py
```

`gen_report.py` is generic — it loads `results.jsonl` and hands the rows to
`build_figures()`. `lib/figures.py` is where the project-specific work goes; it
ships with two worked examples to show the shape, both meant to be rewritten.

## 3. Substitute placeholders

```bash
sed -i '' "s|{project-name}|$NAME|g; s|{PROJECT_NAME}|$TITLE|g" \
  projects/$NAME/workflow.md projects/$NAME/scripts/*.py projects/$NAME/lib/*.py   # macOS
# GNU/Linux (on the GPU box): same but `sed -i` with no '' argument
```

Verify — this must print nothing:

```bash
grep -rn "{project-name}\|{PROJECT_NAME}" projects/$NAME
```

Then, if the project has well-known metrics, override `print_results_table()` in
`run_experiments.py` with project-specific columns (see existing projects).
`poll_result.py` and `gen_report.py` are generic and usually need nothing beyond
the docstring; `lib/figures.py` is meant to be rewritten as soon as you know
what the project measures.

## 4. Write the two docs

**`concepts.md` — create it now, even as a stub.** `workflow.md` links to it in
its header line and directory layout, so skipping it leaves a dangling link:
`ema-viz`, `ssl`, and `universal-ar` each have one today. Sections: task/data
definition, model + loss, metric definitions (state the exact threshold behind
any "solved" / "emerged" style metric), findings appended as they land.

```bash
cat > projects/$NAME/concepts.md <<EOF
# $TITLE — Concepts

## Task / data
## Model & loss
## Metrics
## Findings
EOF
```

**`workflow.md`** — fill in the one-sentence project description and update the
directory layout to the files that actually exist. Delete rows for files you
don't have rather than leaving aspirational ones.

**`proposal.md` (optional)** — worth it for paper replications and anything
multi-stage: premise, claims-under-test table, staged experiment list with
expected outcomes, deviations from the source. `universal-ar` and
`sparse-attn-emergence` use one. Skip it for a single-experiment probe.

**`CLAUDE.md` (optional)** — only for rules that override repo defaults or guard
irreplaceable data. `small_lm` has one because its LLM-generated data cost real
money to produce. Don't restate `workflow.md` in it.

## 5. Scaffold the paper — now, not at the end

```bash
uv run python -c "
from shared_lib.typst_report import scaffold_paper
scaffold_paper('projects/$NAME/paper', title='$TITLE')
"
```

This writes `main.typ`, `template.typ`, `refs.bib`, a nested `.gitignore`, and
nine section stubs, each carrying its obligations as comments and its unwritten
passages as `#todo`. The tree is the checklist.

It goes in at the start for one reason: **the methodology section is written
while you still remember the decisions.** Reconstructed six weeks later from
old code, it is the section that most reliably comes out wrong, and it is the
one an outside reader most depends on.

The paper is a living document — republished as work proceeds, not written once
at the end. It stays a draft (`status: "draft"` shows a banner and renders each
`#todo` in red) until the project ends. Publish it whenever it says something:

```bash
uv run python -m shared_lib.publish projects/$NAME/paper --check    # what's still open
uv run python -m shared_lib.publish projects/$NAME/paper --stable   # publish
```

Always `--stable`: the link gets shared before the paper is finished, and a
date-prefixed key would leave every already-shared link showing a stale
version. The audience contract and per-section obligations are in
[`paper_checklist.md`](paper_checklist.md).

Intermediate findings do **not** go in the paper. They stay as markdown in
`reports/*.md`, published with `save_report()` — see `workflow.md`.

## 6. Write the first experiment

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
__main__            — skip-if-done → load data → init → train → visualize → append_result
```

Include the **skip-if-done** guard from `workflow.md` in `__main__` from the
first experiment — it's cheap up front and retrofitting it always happens after
a crash has already cost a rerun.

## 7. Smoke-test before the real run

The first real launch should never be the first execution. Run once with the
step/epoch budget cut to ~10 and a small batch.

What this catches, roughly in order of how often it bites: shape errors that
only appear after the first `lax.scan` compile, a missing `append_result` field,
viz code that only runs at the end of training, and R2 upload failures (missing
`.env`). Every one of them otherwise surfaces *after* the full run.

## 8. Running on the GPU box

Experiments run on the remote 2× RTX 4090 host, not the Mac — the Mac has no
`.venv`. Verified 2026-08-04:

| | |
|---|---|
| host | `195.133.135.186` (`owner-CROV`) |
| GPUs | 2× RTX 4090, 24 GB each |
| checkout | `/home/newuser/Projects/seewhy` |
| env | JAX 0.8.1, both CUDA devices visible; `.env` with R2 creds present |

**The box has no `.git`.** It is an execution environment, not a checkout — the
Mac is the single source of truth for code. Never run git there. Code goes up
by rsync; `results.jsonl` only ever comes down.

Three gotchas that cost time if you don't know them:

- **`uv` is not on the non-interactive-ssh `PATH`** — call `~/.local/bin/uv`.
  A bare `uv` inside `ssh host 'cmd'` fails with `command not found`.
- **Use `uv run --no-sync`** so a launch can never silently upgrade packages and
  break CUDA (see the GPU warning in the root `CLAUDE.md`).

```bash
# push code — results.jsonl MUST be excluded, see below
rsync -az --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pkl' \
  --exclude 'results.jsonl' \
  projects/$NAME/ 195.133.135.186:Projects/seewhy/projects/$NAME/
ssh 195.133.135.186 "cd Projects/seewhy && ~/.local/bin/uv run --no-sync python \
  projects/$NAME/scripts/run_experiments.py --bg exp1"
# read results back — the only direction results.jsonl ever moves
rsync -az 195.133.135.186:Projects/seewhy/projects/$NAME/results.jsonl projects/$NAME/
```

**Never push `results.jsonl`.** It is appended to on the box, so a stale local copy
overwrites rows a running experiment has already written — silently, because rsync
considers a differing file simply out of date. This destroyed 10 completed sweep cells in
sparse-attn-emergence before the exclude was added. Results flow remote → local only.

`results.jsonl` is committed from the Mac. Never launch two JAX processes on one
GPU — use sequential mode or `--parallel` (which sets `CUDA_VISIBLE_DEVICES`).

## 9. Register the project

When the first report is published, add a section to
[`projects/index.md`](../index.md): a two-line project description plus a table
of report links. It's the repo's only cross-project map — a project that isn't
there is invisible.

List the paper first and mark it as the entry point; the markdown reports below
it are supporting notes. Because the paper is published with `--stable`, its row
never needs updating — the URL is the same every time.

Full workflow in `workflow.md`; the `report-writer` agent automates the
markdown tier.

## 10. Definition of done for setup

- [ ] placeholder `grep` returns nothing
- [ ] `concepts.md` exists (stub is fine) — no dangling link from `workflow.md`
- [ ] `workflow.md` layout matches the real files
- [ ] `.gitignore` in place; `git status` clean apart from intended files
- [ ] `paper/` scaffolded; `--check` runs and reports only `#todo` warnings
- [ ] smoke run passed end-to-end, including one `results.jsonl` row and one
      uploaded figure URL
- [ ] `results.jsonl` row schema has hyperparameters, `n_params`, `time_s`, and
      per-epoch curves
- [ ] runner launched from the repo root on the GPU box; log landed in `logs/`

`results.jsonl` is the only committed **experiment** output. The paper tree also
commits its `figures/`, `assets/` and `.build.json` — see `workflow.md` for why.

Hyperparameters belong in every `results.jsonl` row, not only in the code. The
paper's numeric check reads that file: a value that is never logged can only be
silenced by an allow-list entry, which then hides it changing later.

---

## Template version

Workflow + scripts: **v1**. Start guide: **v3**. Record in `workflow.md` if you
deviate from the template significantly. When the project ends, follow
[`project_end.md`](project_end.md) to push improvements back.

## What changed in v3

The report system became two tiers with different contracts, and the final one
is a paper written for an outside reader.

1. **`paper/` is scaffolded at project start** (step 5) and republished as work
   proceeds, rather than written at the end. `shared_lib.typst_report.scaffold_paper()`
   ships the template, so it is no longer retyped per project — it existed only
   as a one-off inside `omniglot-ar`.
2. **`--stable` publishing.** Uploads used to be date-prefixed, so a document
   republished across ten days lived at ten URLs and every previously shared
   link silently showed a stale version. Stable keys are overwritten in place
   with a short cache lifetime.
3. **`python -m shared_lib.publish`** replaces the per-project
   `publish_report.py` / `publish_paper.py`, which were the same eighty lines
   copied per tree.
4. **`--check`** — structural lint plus a numeric cross-check of prose against
   `results.jsonl`, with a `.lint-allow` escape hatch.
5. **[`paper_checklist.md`](paper_checklist.md)** — the audience contract, also
   carried as comments in each scaffolded section.
6. **Shared figure layer.** `lib/figures.py` returns `Figure` objects that feed
   both tiers; `save_figure()` renders one standalone for a markdown report.
7. **Figure data and `.build.json` are committed.** Projects disagreed on this;
   without them a fresh clone cannot rebuild the PDF, and `--status` on the Mac
   sees no prior publish.

Non-breaking: the runner CLI, directory contract and `results.jsonl` schema are
unchanged, so v1/v2 projects keep working. Adopting the paper tier is opt-in —
run `scaffold_paper()` in an existing project and it drops in beside `report/`.

## What changed in v2

From auditing the 14 template-following projects against the v1 guide:

1. **`concepts.md` is now a step** — v1 never created it, yet the `workflow.md`
   it tells you to copy links to it; 3 projects have dangling links.
2. **`projects/index.md` registration** — a live convention v1 never mentioned.
3. **Remote GPU execution** — v1 implied local runs; the `.venv` is on another
   machine, and the `uv` PATH / `--no-sync` gotchas were undocumented anywhere.
4. **Smoke-test step** — new.
5. **Placeholder substitution scripted** — v1 said replace by hand; the `grep`
   check makes it verifiable.
6. **Per-project `.gitignore`, including `*.json`** — v1 pointed at the root
   `.gitignore`, which is why that file carries per-project blocks; `*.json`
   dumps weren't covered at all and are the ones that leak in practice.
7. **`proposal.md` / `CLAUDE.md` guidance** — both are in use; v1 mentioned
   neither.
8. **Setup definition-of-done checklist** — new.

Non-breaking: nothing here alters `workflow.md`, the runner CLI, or the directory
contract, so v1 projects are unaffected.
