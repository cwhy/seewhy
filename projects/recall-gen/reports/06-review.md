# Review of Report 6

[`06-plan-phase-a-b1.md`](06-plan-phase-a-b1.md), checked against
`results.jsonl`. The experiments are good. The report is unreadable, and two of
its numbers mislead.

Nit-level issues (precision conventions, table formatting, wording) are
deliberately not listed — they are not why it fails.

---

## Why it is unreadable

**It is a work log, not a report.** Three plan steps were done in one sitting, so
three plan steps got written up in the order they were executed. That single
decision causes everything else.

The consequence is that **the finding is at line 143 of 213**. Recall training
*does* produce in-context generalisation once the context is worth reading —
0.505, below ridge (0.631) and below the best achievable pure look-up (0.552),
from an objective that only ever asked for retrieval. That is the result the
project was built to find, and it is more than halfway down, under a heading
phrased as a question.

Three things follow, and they compound:

**The title sells the housekeeping.** "The noise floor removed, memorisation
blocked, and a context worth using" — three activities, weakest first, no
finding. Report 2 was titled "generalisation appears exactly when retrieval
fails". That is what a title is for.

**Three unrelated stories share one document.** A1 is a normalisation fix that
changed no conclusion. A2 is a negative result about memorisation. B1 is the
headline. They are connected only by having happened on the same day. A reader
has to hold three threads and is given no reason why.

**There are no figures — the only report in the project without any.**

| report | 01 | 02 | 03 | 04 | 05 | **06** |
|---|---|---|---|---|---|---|
| image references | 2 | 5 | 4 | 8 | 6 | **0** |

So the whole argument is carried by dense prose and eight tables of bare numbers,
with nothing to rest the eye on and no way to check anything by looking. This is
the report that most needs pictures: nobody can picture a "16 nearest neighbours
context" from a sentence, and the 0.505 / 0.552 / 0.631 comparison is exactly the
bar chart Report 4 already uses.

It also buries its second-best result. The section called "What the degradation
actually is" shows that recall training does not stop reading the context — it
**narrows what it reads it for**, from "what do these sixteen images say about the
answer" to "which of these is the answer", with the final model *more*
context-dependent than the best one (0.693 against 0.281). That is an independent
finding under a heading that names nothing.

## The two numbers that mislead

**A single table row mixes two evaluation protocols.** Line 132 gives exp1 as
`C=0.869  D=0.843  bestD=0.635`. C and D come from the re-scored `exp1_sharedq`
row; best-D comes from the original `exp1` row, because the re-scored rows carry
no history. Nothing says so. This is the worst error in the report, because A1 —
the step being reported — exists precisely to stop this class of comparison.

**The stated units are wrong.** The preamble says "everything below is normalised
MSE"; the A1 table is raw MSE (`0.0488622` is about 0.69 normalised). A reader
who takes the preamble at its word misreads the table by a factor of fourteen.

## What is actually good

The experiments are better than the writing, and the rewrite should not lose
this.

* **The ablation is the right control and it is decisive.** Swapping in another
  episode's context moves exp20's best checkpoint from 0.505 to 0.785. Without
  it, "beats ridge" proves nothing — exp18 clears that bar on a task whose
  context is worthless. Anticipating the objection and measuring it is the
  strongest thing in the report.
* **A2's design is careful where it is easy to be sloppy.** The warp is applied
  *before* the target is chosen, so recall stays exactly as solvable and only the
  pool's finiteness changes; evaluation is left un-warped so the numbers stay
  comparable to the runs they pair with.
* **The gate ran before training and was stricter than its written criterion.**
  Same-class at 0.743 passes the `< 0.8` gate but is still worse than ridge, so
  an objective could reach it without reading the context at all. Catching that
  the written gate was too loose is what a gate is for.

## The rewrite

1. Retitle around the B1 finding.
2. Open with the headline and its reference points, then the scope correction it
   forces on the published paper.
3. Give the specialisation result its own named section.
4. Split A1 out — it is a methods note, not a result.
5. Fix the two misleading numbers and label the provenance of the rest.
6. Add two figures: a knn context beside an i.i.d. one, and exp20 against its
   ceilings.

---

## The rules this suggests

Six, for this project's reports. Deliberately short — a long checklist is how the
nits creep back in.

**1. Lead with the finding.** The title states what was found, not what was done:
"generalisation appears exactly when retrieval fails", never "three plan steps".
The first paragraph gives the headline number *with the thing it should be
compared against*, because a number alone is not a finding — 0.505 means nothing
until it sits beside ridge at 0.631 and the look-up ceiling at 0.552.

**2. One report, one question.** If three things were run in a day, that is three
reports or one question they all bear on. "These happened on Tuesday" is not a
question. A2 and B1 answer different questions and should never have shared a
document.

**3. Figures carry the argument, not the decoration.** A report with no figures is
unreadable no matter how good the numbers are. Any claim about *quality* needs
pixels — "the completions are good" is not checkable in prose. Any comparison of
three or more quantities wants a chart. And say how the samples were picked:
file order flatters, percentiles of per-sample error do not.

**4. Raw numbers, side by side, with their provenance.** No derived statistic
where two measured numbers adjacent say the same thing — the reader should make
the comparison, not receive it pre-made. And every number says which run and
which protocol it came from; two protocols must never share a table row.

**5. Every number carries its reference point in the same place it appears.** A
chance level, a baseline, or a ceiling — in the same row or the same sentence,
not in a paragraph three sections earlier. If the reference moves (chance = 1/M),
it cannot be a column at all.

**6. End with what it changes.** What does this do to what we already published,
and what does it mean for the next run? A report that does not support a decision
is a diary entry.

The single test that catches most of it: **read only the title and the first
paragraph, and ask what you now know.** For Report 6 the answer is "a
normalisation floor was removed", which is the least important thing in it.
