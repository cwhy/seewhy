#import "/template.typ": *

// OBLIGATIONS
//  - The exact task and data: shapes, dtypes, splits, sizes, preprocessing.
//    Enough that a reader could construct the same inputs.
//  - DERIVE the chance level, do not assert it. "5-way classification, so
//    chance is 0.200" is a derivation; "chance is 0.200" is a number the
//    reader has to trust. Every metric in section 6 is read against it.
//  - State what makes the task hard, and what would make it trivially easy —
//    the shortcut you had to rule out.
//  - If train and test differ structurally (disjoint classes, held-out
//    alphabets), say so here; it is usually the load-bearing design choice.

= Task and data

#todo[
  Exact definition. Shapes, splits, sizes.
]

== Chance level

#todo[
  Derive it. Show the arithmetic.
]
