# Project End Guide — Finish the Paper, Improve the Templates

Two jobs when a project wraps up: finish the deliverable, then push what you
learned back into the templates.

The guiding principle for template changes: **evolve forward, never break
backward**. Existing projects should still work unchanged after any update.

---

## Step 0 — Finish the paper

The paper was scaffolded at project start and has been republished throughout.
Ending the project means closing it out.

```bash
uv run python projects/{project-name}/scripts/gen_report.py       # figures from final results
uv run python -m shared_lib.publish projects/{project-name}/paper --check
```

Work through [`paper_checklist.md`](paper_checklist.md) — in particular the
acceptance test on §3 and §4: **could a competent stranger reimplement this
from the paper alone, without opening the repo?**

Then flip `main.typ` from `status: "draft"` to `status: "final"`. Every
remaining `#todo` becomes a compile error, so the build failing is the
checklist telling you what is still open, not a problem to route around.

```bash
uv run python -m shared_lib.publish projects/{project-name}/paper --preview
uv run python -m shared_lib.publish projects/{project-name}/paper --stable
```

Commit the paper tree including `figures/`, `assets/` and `.build.json`, and
make sure the project's row in [`projects/index.md`](../index.md) leads with
the paper.

A project whose paper is still a draft is not finished, however complete the
experiments are. The experiments are what you did; the paper is what anyone
else gets.

---

## What goes where

| Improvement type | Destination |
|---|---|
| New invariant rule or workflow convention discovered | `projects/TEMPLATES/v1/workflow.md` |
| Better `run_experiments.py` infrastructure (args, GPU handling, logging) | `shared_lib/templates/v1/run_experiments.py` |
| Better `poll_result.py` display or polling logic | `shared_lib/templates/v1/poll_result.py` |
| Project-specific metric columns in results table | stays in the project |
| Project-specific viz patterns worth generalizing | `projects/TEMPLATES/v1/workflow.md` (document the pattern) |
| New shared utility (`shared_lib/*.py`) | `shared_lib/` directly — not a template concern |
| Paper template or section-stub improvements | `shared_lib/templates/paper/` |
| A writing obligation worth imposing on every paper | `projects/TEMPLATES/paper_checklist.md` **and** the matching section stub |

---

## Step 1 — Audit the project's script improvements

Compare the project's scripts against the templates:

```bash
diff shared_lib/templates/v1/run_experiments.py projects/{project-name}/scripts/run_experiments.py
diff shared_lib/templates/v1/poll_result.py     projects/{project-name}/scripts/poll_result.py
```

For each diff chunk, ask:
- Is this a bug fix or robustness improvement? → back-port to template
- Is this a project-specific metric/column? → leave it in the project
- Is this a new feature that future projects would want? → add to template

---

## Step 2 — Update `shared_lib/templates/v1/` scripts

Edit the templates directly. Keep changes additive where possible:
- Add new CLI flags with sensible defaults (old invocations still work)
- Improve error messages, logging, heartbeat output
- Never remove existing flags or change their semantics

After editing, verify the template still works as a standalone script
(no project-specific imports, no hardcoded paths).

---

## Step 3 — Update `projects/TEMPLATES/v1/workflow.md`

Look at what was added to the project's `workflow.md` during the project:
- New invariant rules → add to the **Invariant Rules** section
- New JAX patterns → add to **JAX Performance**
- New script types → add an example row to **Diagnostic Scripts**
- Structural changes to experiment files → update **Experiment File Structure**

Keep the template generic: use `{project-name}` placeholders, not the
actual project name. Remove algorithm-specific sections that don't generalize.

---

## Step 4 — Bump the template version (only for breaking changes)

If a change is incompatible with how v1 projects work (e.g. a renamed flag,
a new required file, a changed directory layout), create a `v2/` directory
instead of modifying `v1/`:

```bash
cp -r projects/TEMPLATES/v1  projects/TEMPLATES/v2
cp -r shared_lib/templates/v1 shared_lib/templates/v2
# then edit v2/ with the breaking changes
```

Update `project_start.md` to point new projects at `v2/`.
Old projects remain on `v1/` — no migration needed unless they want new features.

---

## Step 5 — Update the ema-feature (or current) project's workflow.md

After back-porting to templates, note in the project's `workflow.md` if it
introduced a pattern that was upstreamed, so future readers know where to
look for the canonical version.

---

## Checklist

- [ ] Paper passes `--check` with no errors and compiles at `status: "final"`
- [ ] `paper_checklist.md` worked through, including the §3/§4 reimplementation test
- [ ] Paper published with `--stable`; `projects/index.md` leads with it
- [ ] Diffed project scripts against `shared_lib/templates/v1/`
- [ ] Back-ported generic improvements to `shared_lib/templates/v1/`
- [ ] Paper template improvements back-ported to `shared_lib/templates/paper/`
- [ ] Updated `projects/TEMPLATES/v1/workflow.md` with new conventions
- [ ] No project-specific metric names or paths leaked into shared templates
- [ ] Breaking changes went into a new version directory, not v1
- [ ] `results.jsonl` committed, large pkl/npy files gitignored
- [ ] Paper tree committed **with** `figures/`, `assets/` and `.build.json`
- [ ] `test-scripts/test_report_tooling.py` still passes if you touched
      `shared_lib/{paper_lint,typst_report,publish}.py`
