// The architecture figure, drawn in Typst with cetz rather than as an SVG.
//
// Why not SVG: an SVG's text is resolved against Typst's own font set (so
// `sans-serif` silently becomes the document serif), its sizes are viewBox units
// that have to be converted by hand to end up above the print floor, and its
// maths has to be faked with `tspan` shifts. Here `$H d_k^2$` is real Typst
// maths, sizes are pt, and the face is the document's by construction.
//
// The canvas is authored at roughly text-block width, so the pt sizes below are
// the sizes that reach the page — do not scale this figure.

#import "@preview/cetz:0.5.2"

#let c-ctx = rgb("#2a78d6")      // context tokens — they write
#let c-qry = rgb("#eb6834")      // query tokens — they only read
#let c-ink = rgb("#0b0b0b")
#let c-mut = rgb("#52514e")
#let c-lin = rgb("#7d7b76")
#let c-bad = rgb("#c1121f")
#let c-pan = rgb("#f2f1ee")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let sm(body) = text(7.5pt, fill: c-mut, body)
  let lb(body) = text(8.5pt, fill: c-ink, body)
  let hd(body) = text(10.5pt, weight: "bold", fill: c-ink, body)

  // an image token: a framed tile with a couple of pen strokes in it
  let tile(cx, cy, strokes, col: c-ctx, masked: false) = {
    rect((cx - 0.55, cy - 0.55), (cx + 0.55, cy + 0.55),
         radius: 0.04, fill: c-pan, stroke: 1pt + col)
    for s in strokes {
      bezier((cx + s.at(0).at(0), cy + s.at(0).at(1)),
             (cx + s.at(1).at(0), cy + s.at(1).at(1)),
             (cx + s.at(2).at(0), cy + s.at(2).at(1)),
             stroke: (paint: rgb("#3a3936"), thickness: 1.5pt, cap: "round"))
    }
    if masked {
      rect((cx - 0.55, cy - 0.55), (cx + 0.55, cy - 0.02),
           fill: rgb("#b9b7b1"), stroke: 1pt + col)
    }
  }

  // ── panel (a) ──────────────────────────────────────────────────────────────
  content((0, 17.15), anchor: "west",
          hd[(a) an episode — the state is the only path from context to query])

  tile(0.75, 15.2, (((-0.25, 0.2), (0.25, 0.3), (0.05, 0.45)),
                    ((0.25, 0.3), (-0.1, -0.35), (0.3, -0.05))))
  tile(2.05, 15.2, (((-0.25, 0.3), (0.25, 0.3), (0, 0.34)),
                    ((0.25, 0.3), (-0.05, -0.35), (0.2, -0.05))))
  tile(3.35, 15.2, (((0.0, 0.32), (0.0, -0.35), (0.03, 0)),
                    ((-0.22, 0.14), (0.0, 0.32), (-0.14, 0.28))))
  content((4.35, 15.2), text(11pt, fill: c-mut)[$dots.c$])
  tile(5.35, 15.2, (((-0.25, -0.3), (0.28, -0.28), (0.02, 0.5)),
                    ((-0.16, 0.02), (0.16, 0.02), (0, 0.02))))

  content((3.05, 14.15), lb[M context images, complete])
  content((3.05, 13.72), text(8pt, fill: c-ctx)[these WRITE to the state])

  rect((6.9, 14.42), (10.3, 15.98), radius: 0.12, fill: white, stroke: 1.2pt + rgb("#3a3936"))
  content((8.6, 15.62), text(11.5pt, weight: "bold", fill: c-ink)[state $S$])
  content((8.6, 15.20), text(8.5pt, fill: c-mut)[$H times d_k times d_k$])
  content((8.6, 14.78), text(9pt, fill: c-ink)[$= 16 space 384$ numbers])
  content((8.6, 14.05), sm[fixed — independent of $M$])

  tile(11.85, 15.2, (((-0.25, 0.28), (0.25, 0.3), (0, 0.36)),),
       col: c-qry, masked: true)
  tile(13.15, 15.2, (((-0.22, 0.3), (0.22, 0.3), (0, 0.3)),),
       col: c-qry, masked: true)
  content((14.15, 15.2), text(11pt, fill: c-mut)[$dots.c$])

  content((12.9, 14.15), lb[Q queries, bottom half hidden])
  content((12.9, 13.72), text(8pt, fill: c-qry)[these only READ])

  line((6.02, 15.2), (6.78, 15.2), stroke: 1.2pt + c-ctx, mark: (end: ">", scale: 0.8))
  content((6.40, 15.62), text(7.5pt, fill: c-ctx)[write])
  content((6.40, 14.80), text(7.5pt, fill: c-mut)[$beta > 0$])

  line((10.42, 15.2), (11.18, 15.2), stroke: 1.2pt + c-qry, mark: (end: ">", scale: 0.8))
  content((10.80, 15.62), text(7.5pt, fill: c-qry)[read])
  content((10.80, 14.80), text(7.5pt, fill: c-mut)[$o = S q$])

  bezier((11.45, 14.58), (10.36, 14.44), (11.35, 13.6), (10.7, 13.7),
         stroke: (paint: rgb("#9c9a94"), thickness: 1pt, dash: "dashed"),
         mark: (end: ">", scale: 0.7, fill: rgb("#9c9a94")))
  line((10.80, 13.78), (11.08, 14.06), stroke: (paint: c-bad, thickness: 1.6pt, cap: "round"))
  line((11.08, 13.78), (10.80, 14.06), stroke: (paint: c-bad, thickness: 1.6pt, cap: "round"))
  content((10.6, 13.25), text(8pt, fill: c-bad)[$beta = 0$ — a query never writes])

  // ── panel (b) ──────────────────────────────────────────────────────────────
  line((0, 12.55), (15.65, 12.55), stroke: 0.6pt + rgb("#d8d6d1"))
  content((0, 12.05), anchor: "west", hd[(b) what each token passes through])

  let stage(x0, x1, cy, body, fill: c-pan, col: c-lin) = {
    rect((x0, cy - 0.45), (x1, cy + 0.45), radius: 0.08, fill: fill, stroke: 1pt + col)
    content(((x0 + x1) / 2, cy), body)
  }
  let flow = 10.6
  stage(0.15, 1.85, flow, [#lb[embed] \ #sm[$784 -> 256$]])
  stage(2.35, 3.75, flow, lb[layer 1], fill: rgb("#eaf1fb"), col: c-ctx)
  stage(4.15, 5.55, flow, lb[layer 2], fill: rgb("#eaf1fb"), col: c-ctx)
  stage(5.95, 7.35, flow, lb[layer 3], fill: rgb("#eaf1fb"), col: c-ctx)
  stage(7.75, 9.15, flow, lb[layer 4], fill: rgb("#eaf1fb"), col: c-ctx)
  stage(9.65, 11.35, flow, [#lb[head] \ #sm[$256 -> 784$]])

  for (a, b) in ((1.85, 2.35), (3.75, 4.15), (5.55, 5.95), (7.35, 7.75),
                 (9.15, 9.65), (11.35, 11.95)) {
    line((a, flow), (b, flow), stroke: 1pt + c-mut, mark: (end: ">", scale: 0.65))
  }

  rect((11.95, flow - 0.55), (13.05, flow + 0.55), radius: 0.04,
       fill: c-pan, stroke: 1pt + c-lin)
  bezier((12.2, flow + 0.2), (12.7, flow + 0.3), (12.45, flow + 0.36),
         stroke: (paint: rgb("#3a3936"), thickness: 1.5pt, cap: "round"))
  rect((11.95, flow - 0.55), (13.05, flow - 0.02), fill: rgb("#cfe0f6"), stroke: 1pt + c-ctx)
  content((13.3, flow), anchor: "west", lb[only the shaded \ half is scored])

  // the blow-up
  for p in ((2.6, 8.7), (13.6, 8.7)) {
    line((4.85, flow - 0.5), p, stroke: (paint: rgb("#9c9a94"), thickness: 0.7pt, dash: "dashed"))
  }
  rect((0.15, 2.55), (15.65, 8.7), radius: 0.14, fill: rgb("#fbfbfa"), stroke: 0.8pt + rgb("#c9c7c2"))
  content((0.6, 8.05), anchor: "west", text(9pt, style: "italic", fill: c-mut)[inside one layer])

  let inner = 5.55
  content((0.85, inner), text(10pt)[$h$])
  stage(1.45, 3.15, inner, lb[LayerNorm], fill: white)
  rect((3.75, inner - 0.75), (6.35, inner + 0.75), radius: 0.1,
       fill: rgb("#eaf1fb"), stroke: 1.2pt + c-ctx)
  content((5.05, inner + 0.38), text(9.5pt, weight: "bold", fill: c-ink)[KDA])
  content((5.05, inner - 0.06), sm[writes $S$, then])
  content((5.05, inner - 0.42), sm[reads the finished $S$])
  circle((7.0, inner), radius: 0.3, fill: white, stroke: 1pt + c-lin)
  content((7.0, inner), text(10pt)[$+$])
  stage(7.75, 9.45, inner, lb[LayerNorm], fill: white)
  stage(9.75, 12.0, inner, [#lb[MLP] \ #sm[$256 -> 1024 -> 256$]])
  circle((12.6, inner), radius: 0.3, fill: white, stroke: 1pt + c-lin)
  content((12.6, inner), text(10pt)[$+$])
  content((13.35, inner), text(10pt)[$h'$])

  for (a, b) in ((1.05, 1.45), (3.15, 3.75), (6.35, 6.7), (7.3, 7.75),
                 (9.45, 9.75), (12.0, 12.3), (12.9, 13.15)) {
    line((a, inner), (b, inner), stroke: 1pt + c-mut, mark: (end: ">", scale: 0.65))
  }
  for (a, b) in ((1.25, 7.0), (7.55, 12.6)) {
    bezier((a, inner + 0.2), (b, inner + 0.34), (a, inner + 1.5), (b, inner + 1.5),
           stroke: (paint: rgb("#9c9a94"), thickness: 0.9pt, dash: "dashed"),
           mark: (end: ">", scale: 0.6, fill: rgb("#9c9a94")))
  }
  content((4.1, inner + 1.35), sm[residual])
  content((10.0, inner + 1.35), sm[residual])

  content((0.6, 3.55), anchor: "west",
          sm[The same $S$ is written and read once per layer; there are four of them, and none is shared.])
  content((0.6, 3.05), anchor: "west",
          sm[Context tokens write in every layer; query tokens write in none.])
})
