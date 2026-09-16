// Slide template — 16:9, one heading per slide, drawings via cetz.
//
// Deliberately small: a title helper, a palette, a ledger table, and two
// diagram primitives (a neuron and an arrow) that the circuit drawings reuse so
// that predictive coding and PC-ALM are drawn in the same visual language.

#import "@preview/cetz:0.4.2" as cetz

#let accent = rgb("#0072b2")
#let warn = rgb("#c1121f")
#let good = rgb("#1b7f4b")
#let muted = luma(105)
#let soft = luma(240)

#let deck(title: "", subtitle: none, footer: "", body) = {
  set page(
    paper: "presentation-16-9",
    margin: (x: 1.9cm, top: 1.4cm, bottom: 1.1cm),
    footer: context [
      #set text(size: 9pt, fill: muted)
      #footer
      #h(1fr)
      #counter(page).display()
    ],
  )
  set text(size: 19pt)
  set par(justify: false, leading: 0.72em)

  // title slide
  v(3.2cm)
  text(size: 34pt, weight: "bold", fill: accent, title)
  if subtitle != none {
    v(4pt)
    text(size: 19pt, fill: muted, subtitle)
  }
  body
}

#let slide(title, body) = {
  pagebreak(weak: true)
  text(size: 25pt, weight: "bold", fill: accent, title)
  v(2pt)
  line(length: 100%, stroke: 0.7pt + accent.lighten(65%))
  v(10pt)
  set text(size: 17pt)
  body
}

// A claim that carries its own verdict, for the plausibility ledgers.
#let verdict(mark, body) = {
  let (sym, col) = if mark == "yes" { ("✓", good) }
    else if mark == "no" { ("✗", warn) }
    else { ("~", muted) }
  grid(
    columns: (1.1em, 1fr), gutter: 6pt,
    text(fill: col, weight: "bold", sym), body,
  )
}

#let note(body) = block(
  width: 100%, inset: 9pt, radius: 3pt, fill: soft,
  text(size: 15pt, body),
)

// ── shared drawing vocabulary ───────────────────────────────────────────────
// Every circuit in this deck uses the same shapes, so that the one difference
// between predictive coding and PC-ALM is visible as a difference in the
// picture rather than in the caption.

#let neuron(pos, label, fill: white, stroke: black, r: 0.42) = {
  import cetz.draw: circle, content
  circle(pos, radius: r, fill: fill, stroke: 0.9pt + stroke)
  content(pos, text(size: 11pt, label))
}

#let flow(from, to, col: black, dash: none, label: none, bend: 0) = {
  import cetz.draw: line, bezier, content
  if bend == 0 {
    line(from, to, stroke: (paint: col, thickness: 1pt, dash: dash),
         mark: (end: ">", fill: col, scale: 0.5))
  } else {
    let mid = ((from.at(0) + to.at(0)) / 2 + bend, (from.at(1) + to.at(1)) / 2)
    bezier(from, to, mid, stroke: (paint: col, thickness: 1pt, dash: dash),
           mark: (end: ">", fill: col, scale: 0.5))
  }
  if label != none {
    let mid = ((from.at(0) + to.at(0)) / 2 + bend * 1.4, (from.at(1) + to.at(1)) / 2)
    content(mid, text(size: 10pt, fill: col, label))
  }
}
