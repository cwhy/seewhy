// The method, drawn in Typst with cetz (as recall-gen's architecture figure):
// real Typst text and maths, sizes in pt at the size they reach the page.
// Authored at text-block width — do not scale.

#import "@preview/cetz:0.5.2"

#let c-ctx = rgb("#2a78d6")      // context images / real
#let c-gen = rgb("#e8590c")      // generator / fake
#let c-ink = rgb("#111111")
#let c-mut = rgb("#55534e")
#let c-pan = rgb("#f2f1ee")
#let c-ok = rgb("#1a9850")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let sm(body) = text(7.5pt, fill: c-mut, body)
  let lb(body) = text(8.5pt, fill: c-ink, body)
  let hd(body) = text(10pt, weight: "bold", fill: c-ink, body)

  // an image: a framed tile with one pen stroke
  let tile(cx, cy, col, k: 0) = {
    rect((cx - 0.42, cy - 0.42), (cx + 0.42, cy + 0.42), radius: 0.04, fill: c-pan, stroke: 1pt + col)
    let s = ((( -0.15, 0.25), (0.2, 0.05), (-0.05, -0.28)),
             ((0.0, 0.28), (0.02, 0.0), (0.0, -0.28)),
             ((-0.18, 0.2), (0.2, 0.25), (0.05, -0.28)),
             ((-0.12, -0.25), (0.25, 0.1), (-0.15, 0.22))).at(calc.rem(k, 4))
    bezier((cx + s.at(0).at(0), cy + s.at(0).at(1)), (cx + s.at(2).at(0), cy + s.at(2).at(1)),
           (cx + s.at(1).at(0), cy + s.at(1).at(1)),
           stroke: (paint: rgb("#3a3936"), thickness: 1.4pt, cap: "round"))
  }
  let arrow(a, b, col: c-mut, w: 1pt, dash: none) = line(a, b, stroke: (paint: col, thickness: w, dash: dash),
                                                        mark: (end: ">", scale: 0.7))
  let box(a, b, title, sub, col: c-ink, fill: white) = {
    rect(a, b, radius: 0.12, fill: fill, stroke: 1.1pt + col)
    let cx = (a.at(0) + b.at(0)) / 2
    content((cx, b.at(1) - 0.38), text(9.5pt, weight: "bold", fill: col, title))
    content((cx, a.at(1) + 0.36), sm(sub))
  }

  // ── (a) the game ────────────────────────────────────────────────────────────
  content((0, 9.6), anchor: "west", hd[(a) one training episode])

  // context row
  for i in range(5) { tile(0.6 + i * 1.0, 8.3, c-ctx, k: i) }
  content((5.55, 8.3), text(11pt, fill: c-mut)[$dots.c$])
  content((3.0, 7.45), lb[16 context images $c_1 dots c_16$ (real)])

  // discriminator state
  box((7.0, 7.45), (11.9, 9.15), [discriminator's KDA state $S$], [$4 times 64 times 64 = 16 space 384$ numbers, fixed size])
  arrow((6.05, 8.3), (6.9, 8.3), col: c-ctx, w: 1.2pt)
  content((6.47, 8.65), text(7.5pt, fill: c-ctx)[write])

  // generator
  box((0.2, 4.6), (4.6, 6.15), [generator $G$], [(context-aware: its own KDA state)], col: c-gen)
  arrow((2.4, 7.0), (2.4, 6.25), col: c-gen, dash: "dashed")
  content((3.55, 6.62), text(7.5pt, fill: c-gen)[context-aware only])
  content((-0.35, 5.38), lb[$z$])
  arrow((-0.15, 5.38), (0.12, 5.38))

  // queries
  tile(6.2, 5.9, c-ctx, k: 2)
  content((6.2, 5.2), sm[copy of a $c_i$])
  content((6.2, 6.62), text(8pt, fill: c-ctx)[real query])
  tile(6.2, 3.9, c-gen, k: 3)
  content((6.2, 4.62), text(8pt, fill: c-gen)[fake query])
  arrow((4.7, 5.0), (5.7, 4.1), col: c-gen, w: 1.2pt)

  // discriminator read-out
  box((8.0, 3.75), (11.4, 6.2), [read-out], [$o = S q$, then a small MLP])
  arrow((9.45, 7.35), (9.45, 6.3), w: 1.2pt)
  content((10.0, 6.83), sm[read])
  arrow((6.7, 5.9), (7.9, 5.3))
  arrow((6.7, 3.9), (7.9, 4.6))

  // outputs
  arrow((11.5, 5.0), (12.3, 5.0), w: 1.2pt)
  content((12.4, 5.55), anchor: "west", lb[$D = P("in context")$])
  content((12.4, 4.95), anchor: "west", text(8.5pt, fill: c-ctx)[real → 1])
  content((12.4, 4.45), anchor: "west", text(8.5pt, fill: c-gen)[fake → 0])
  content((12.4, 3.85), anchor: "west", text(8pt, fill: c-gen)[$G$ wants fake → 1])

  // ── (b) the two probes ──────────────────────────────────────────────────────
  content((0, 2.55), anchor: "west", hd[(b) how the trained discriminator is tested (no training involved)])

  // swap test
  content((0.1, 1.75), anchor: "west", lb[*swap test* — same queries, another episode's context:])
  for i in range(3) { tile(0.55 + i * 0.95, 0.75, rgb("#9a9893"), k: i + 1) }
  arrow((3.05, 0.75), (3.75, 0.75), col: c-mut)
  rect((3.85, 0.3), (5.5, 1.2), radius: 0.1, fill: white, stroke: 0.9pt + c-mut)
  content((4.67, 0.75), sm[wrong $S$])
  content((0.1, -0.1), anchor: "west", sm[score unchanged → $S$ ignored; score worse → $S$ relied on])

  // classifier
  content((7.6, 1.75), anchor: "west", lb[*digit classifier* — one context per digit:])
  for k in range(4) {
    let x = 8.3 + k * 1.6
    rect((x - 0.72, 0.3), (x + 0.72, 1.2), radius: 0.1, fill: c-pan, stroke: 0.9pt + c-ctx)
    content((x, 0.75), text(7.5pt, fill: c-ctx)[16 × "#k"])
  }
  content((14.85, 0.75), text(11pt, fill: c-mut)[$dots.c$])
  content((7.6, -0.1), anchor: "west", sm[a test image is assigned the digit whose context scores it highest])
})
