# Project End Guide — Improving the Templates

After a project wraps up, improvements made during it should flow back into
the templates so future projects start from a better baseline.

The guiding principle: **evolve forward, never break backward**.
Existing projects should still work unchanged after any template update.

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

- [ ] Diffed project scripts against `shared_lib/templates/v1/`
- [ ] Back-ported generic improvements to `shared_lib/templates/v1/`
- [ ] Updated `projects/TEMPLATES/v1/workflow.md` with new conventions
- [ ] No project-specific metric names or paths leaked into shared templates
- [ ] Breaking changes went into a new version directory, not v1
- [ ] `results.jsonl` committed, large pkl/npy files gitignored
