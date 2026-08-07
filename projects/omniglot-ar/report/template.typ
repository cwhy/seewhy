// Shared styling and helpers for this report. Hand-written — edit freely.

#let accent = rgb("#0072b2")
#let muted = luma(105)

// Print pages are A4; `web: true` keeps the A4 width but lets the page grow to
// one continuous strip, which is what the HTML viewer embeds.
#let report(title: "", subtitle: none, date: none, web: false, body) = {
  set page(
    width: 21cm,
    height: if web { auto } else { 29.7cm },
    margin: (x: 2.2cm, y: 2.2cm),
    numbering: if web { none } else { "1" },
  )
  // No font family: Typst's bundled default (New Computer Modern) renders
  // identically on macOS and on the GPU box. Naming Helvetica here silently
  // fell back to a different face on Linux, so the same report looked
  // different depending on where it was compiled.
  set text(size: 10pt, lang: "en")
  set par(justify: true, leading: 0.68em)
  show heading: set block(above: 1.5em, below: 0.8em)
  show heading.where(level: 1): set text(size: 15pt, weight: 700)
  show heading.where(level: 2): set text(size: 12pt, weight: 700)
  show heading.where(level: 3): set text(size: 10.5pt, weight: 700, fill: muted)
  show link: set text(fill: accent)
  set table(stroke: (x, y) => (top: if y <= 1 { 0.5pt } else { 0pt },
                               bottom: 0.5pt))
  show table.cell.where(y: 0): strong

  block(width: 100%, {
    text(size: 20pt, weight: 700, title)
    if subtitle != none { linebreak(); text(size: 11pt, fill: muted, subtitle) }
    if date != none { linebreak(); text(size: 9pt, fill: muted, date) }
  })
  line(length: 100%, stroke: 0.5pt + muted)
  v(0.6em)
  body
}

// Place a generated figure: #fig(include "/figures/x.typ", caption: [...])
#let fig(body, caption: none) = figure(body, caption: caption, kind: image)

// A boxed aside for a caveat, a decision, or a result worth not missing.
#let callout(title: none, body) = block(
  width: 100%, inset: 10pt, radius: 4pt,
  fill: accent.lighten(92%), stroke: (left: 2pt + accent),
  {
    if title != none { text(weight: 700, size: 9.5pt, title); linebreak() }
    set text(size: 9.5pt)
    body
  },
)

// A compact key/value block for run settings.
#let kv(..pairs) = block(
  width: 100%,
  table(
    columns: (auto, 1fr),
    stroke: none,
    inset: (x: 0pt, y: 3pt),
    column-gutter: 12pt,
    ..pairs.pos().map(p => (text(fill: muted, size: 9pt, p.at(0)),
                            text(size: 9pt, raw(str(p.at(1)))))).flatten(),
  ),
)
