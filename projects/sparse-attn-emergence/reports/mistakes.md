# Mistakes

Every error made building this replication, what it cost, and — the part actually worth
reading — **how it surfaced**. Kept because a replication whose failures are hidden is worth
less than one whose failures are listed: you cannot calibrate the results without knowing what
nearly went wrong.

Ordered by consequence.

| # | Mistake | Cost | How it surfaced |
|---|---|---|---|
| 1 | Asserted H5 refuted after testing the wrong config | one wrong published claim | **a reader pushed back** |
| 2 | `rsync` push overwrote the remote `results.jsonl` | 10 completed sweep cells, ~25 min GPU | `pick_config` found no rows |
| 3 | Re-running an experiment truncated its log | the only on-disk record of the lost cells | noticed the file was short |
| 4 | Emergence threshold admitted non-solutions | distorted headline numbers in two experiments | a loss value was *too exactly* `ln 2/S` |
| 5 | Explained a metric gap with a story that was wrong | a wrong sentence on a published page | exp4's per-head numbers |
| 6 | Proposed XOR arity as the difficulty driver | a wrong hypothesis, stated twice | the `s=32` cell refuted it |
| 7 | Gave both architectures the same learning rate "for fairness" | made the mixer look hopeless | the LR sweep that followed |
| 8 | Logged the wrong attention-alignment aggregation | understated the model for one experiment | reading final values against final loss |
| 9 | No per-config error handling in a sweep | 7 queued configs discarded by one OOM | the sweep ended early |
| 10 | "Fixed" that OOM by disabling CUDA command buffers | 2h20m wedged compile, twice | 1.7% CPU with an idle GPU |
| 11 | Smoke rows reused real config names | none — caught pre-launch | reading the smoke output |
| 12 | Ranked "best learning rate" by median loss | briefly reported the worse of two runs | the printed table disagreed with the log |
| 13 | Wrote page links to pages that did not exist | none — publish aborted | a validator written for the purpose |
| 14 | Scaffolding gaps against my own new checklist | none | reviewing the staged diff |
| 15 | Misread elapsed time, declared a healthy run stalled | a wrong paragraph | checking the clock |

## The one that mattered

**Testing the wrong configuration, then asserting a contradiction.** exp6 compared mixer
against transformer at cells my *own* sweep flagged as hardest, and reported that H5 "does not
replicate — it inverts". The paper's mixer claim is at `S=16, s=7`. I had auto-selected cells
from my results instead of reading their config, so the comparison never touched the claim.

When challenged, re-reading turned up three differences at once: the config, an unstated
causal-masking choice worth the entire result, and the absence of any published
hyperparameters. Re-run properly ([exp6/7](sparse_attn_emergence_exp67.html)), the paper's
claim holds in direction at its own config — the opposite of what I had said.

Two things about this are uncomfortable and worth naming. The auto-selection *felt* rigorous —
picking the config from measured data rather than by hand — which is exactly why it went
unexamined. And nothing internal would have caught it: the code was correct, the runs were
clean, the numbers were real. Only the mismatch between my config and theirs was wrong, and no
amount of re-running my own setup surfaces that.

## Data loss

**The `rsync` clobber.** Code is pushed to the GPU box; `results.jsonl` is written *on* the box
and pulled back. Pushing the project directory sent a stale local copy over the live file,
destroying 10 completed sweep rows mid-run. Earlier pushes had been harmless only because
nothing had been appended remotely since the previous pull.

The trap is that rsync has no concept of one file in a tree being remote-authoritative. Every
other file flows one way; that one flows the other; mixing both directions in a single command
makes a routine code push destructive. Fixed by making the safe path the only path — a
`sync.sh` with `push`/`pull` and the exclude baked in — rather than by remembering to type a
flag.

**Then losing the record too.** Re-running the sweep truncated `logs/exp2.log`, which held the
summary lines for the very cells that had just been destroyed. Two independent single-copy
failures on the same data. The runner now appends with a per-run header.

## Metric design errors

Three of the fifteen were measurement, not code, and they were the most insidious because the
runs looked perfect.

**A threshold that accepted non-solutions.** `acc2 > 0.95` is passed by (a) a model that copies
its own previous output at maximum density, and (b) a model that learns 15 of 16 rows. So
"solved" meant different things in different cells, and [H4](sparse_attn_emergence_exp3.html)
looked noisy and saturating under it while being cleanly monotone under a strict criterion.

**The wrong aggregation.** exp1 logged best-*single-head* alignment, which understates a model
whose heads divide the work — final values sat at 0.49–0.97 while loss was already ~0.

**Then the wrong explanation for it.** I attributed that gap to heads specialising by row.
exp4's per-head numbers (`0.27 0.33 0.53 0.82 0.11 0.15 0.38 0.21`) show one dominant head plus
partial helpers — a different structure. The first published explanation was a guess presented
with more confidence than it had earned.

## Infrastructure errors

**One crash taking a sweep with it.** An OOM at `H=128` discarded the seven head-dimension
configs queued behind it. A `try`/`except` per config would have cost one line up front.

**Flag-flipping instead of diagnosing.** The OOM error message suggested disabling CUDA command
buffers, so I did — trading an immediate failure for a 2h20m compile at 1.7% CPU, twice, before
looking at *why* the graph was enormous. The actual fix (a shorter scan chunk) addressed graph
size. The remaining `d_head=1` failures are honestly reported as unmeasured.

## What actually caught things

Worth separating, because the mechanisms are not equally available:

- **A reader pushing back** — caught #1, the only error that produced a wrong conclusion. Nothing
  internal to the project could have.
- **Numbers that were too exact to be coincidence** — 0.0433 and 0.0217 matching `ln 2/16` and
  `ln 2/32` to four decimals is what exposed the copying artifact (#4). Suspicion of *clean*
  numbers, not messy ones.
- **Validators written on purpose** — the dead-link check (#13) aborted a publish; it existed
  because relative links across a date-foldered store are fragile.
- **Reading my own output** rather than skimming for success — caught #11 and #12.
- **Re-reading the source material** — the entire resolution of #1, #7 and the masking question.

And what caught nothing: the code running without errors. Every one of #1, #4, #5, #6, #8 and
#12 occurred in code that ran cleanly and produced plausible numbers.

## What generalises

1. **Match the source's configuration before comparing to its claim.** A config chosen from
   your own data is not a neutral choice.
2. **Distrust suspiciously round numbers.** Exact arithmetic matches are where artifacts hide.
3. **State a threshold's failure modes when you define it**, not after it distorts a result.
4. **Make the destructive path unavailable**, not merely documented.
5. **Never let one config's failure end a sweep.**
6. **Separate the finding from the explanation.** Every wrong sentence here was an
   interpretation attached to a correct measurement.
