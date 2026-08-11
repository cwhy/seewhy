# Paper Checklist

The contract a project's `paper/` tree has to meet before it counts as done.

The paper is **not** a longer version of the markdown reports. Those are lab
notes: written for you, next week, with all the project context in your head.
The paper is written for someone who has never heard of the project and never
will hear about it from anyone but this document.

## The audience

**Technical, but not an ML specialist.** They write code and read statistics.
They have never trained a neural network and do not know what attention is.

Two consequences that are easy to get wrong:

- Every piece of ML vocabulary is defined at first use — attention, in-context
  learning, embedding, logit, epoch, chance level. `template.typ` provides
  `#gloss[term][one-line definition]` so this costs one line, not a paragraph.
- "Standard" choices are not standard to this reader. Adam, cosine decay,
  layer norm: name them, say in one clause what they do, move on.

The acceptance test, applied to §3 and §4: **could a competent stranger
reimplement this from the paper alone, without opening the repo?** If any
answer is "they'd have to read `experiments7.py`", the section is not finished.

## Per-section obligations

The scaffolded sections carry these as comments at the top of each file. This
is the same list, in one place, for reviewing a finished draft.

**1 Introduction** — the question in plain terms, before the method. Explicit
contribution list; each item a claim the paper actually supports.

**2 Background** — mandatory. Defines everything §3–§7 rely on. Related work
only insofar as it frames the question.

**3 Task and data** — exact shapes, splits, sizes, preprocessing. The chance
level **derived**, not asserted: show the arithmetic. What makes the task hard,
and the shortcut you had to rule out.

**4 Methodology** — model as equations or pseudocode, not prose describing
code. Loss written out. Complete hyperparameter table. Seeds and repeats, and
whether reported numbers are single runs or aggregates. Hardware, wall-clock,
parameter count. `#notation` for the symbol table.

**5 Experiments** — a table of run → question → source file. The controls,
especially the positive control: without one, a negative result cannot be
distinguished from a broken harness.

**6 Results** — every metric with its chance level or baseline **in the same
row**. Seed variance, or an explicit note that a number is one run. One figure
per claim. Report what happened; interpretation is §7.

**7 Analysis** — mechanism, not restatement. What would falsify the
explanation, and whether that test was run.

**8 Limitations and negative results** — its own section. What was tried and
failed, including things abandoned before they reached `results.jsonl` — that
is the part a reader cannot reconstruct from the repo. Where the result does
not generalise, specifically.

**9 Conclusion + reproduction appendix** — commit hash, exact commands, and
which `results.jsonl` rows back each figure and headline number.

## Mechanical checks

```bash
uv run python -m shared_lib.publish projects/$NAME/paper --check
```

Reports structural problems (orphan sections, missing includes, missing figure
assets, unresolved citations, remaining `#todo`) and cross-references every
number in the prose against `results.jsonl`.

**On the numeric check.** A literal matches if some value in `results.jsonl`
rounds to it at the precision written — `0.228` matches a stored `0.22814`.
Numbers that legitimately are not results go in `paper/.lint-allow`, one per
line with a reason:

```
1623   # Omniglot character count, from the dataset not a run
3.38   # n_params written in millions
```

Measured on a real finished paper, the check catches about **four out of five**
stale numbers. It is a net, not a proof: a coarse literal like `0.3` is not
discriminating against any realistic pool of values. A clean report means
"nothing obvious", never "the numbers are verified". Treat a growing
`.lint-allow` as a smell — it usually means results that should be logged are
not being logged.

## Definition of done

- [ ] `--check` reports no errors
- [ ] every `#todo` gone; `status: "final"` in `main.typ` compiles (it will
      refuse while any remain)
- [ ] §3 and §4 pass the reimplementation test above
- [ ] every metric in §6 sits beside its chance level or baseline
- [ ] §8 exists and is not a single hedging sentence
- [ ] reproduction appendix filled in with a real commit hash
- [ ] published, and the URL added to [`projects/index.md`](../index.md)
