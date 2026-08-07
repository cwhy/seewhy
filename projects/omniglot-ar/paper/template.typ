// Paper-style template: title block, abstract, numbered sections, outline.
// Hand-written — edit freely. The short-form report lives in ../report/.

#let accent = rgb("#0072b2")
#let muted = luma(105)
#let rule = 0.5pt + luma(180)

// `web: true` collapses the paper to one continuous page. The published PDF
// uses paged A4; the flag exists for a scrolling preview.
#let paper(
  title: "",
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  web: false,
  body,
) = {
  set document(title: title)
  set page(
    width: 21cm,
    height: if web { auto } else { 29.7cm },
    margin: (x: 2.4cm, top: 2.4cm, bottom: 2.2cm),
    numbering: if web { none } else { "1" },
    number-align: center,
  )
  // No font family: Typst's bundled default renders identically on macOS and
  // on the GPU box, where the report is actually compiled.
  set text(size: 10pt, lang: "en")
  set par(justify: true, leading: 0.62em, first-line-indent: 0pt, spacing: 0.9em)

  set heading(numbering: "1.1")
  show heading: set block(above: 1.4em, below: 0.7em)
  show heading.where(level: 1): set text(size: 12.5pt, weight: 700)
  show heading.where(level: 2): set text(size: 10.5pt, weight: 700)
  show heading.where(level: 3): set text(size: 10pt, weight: 700, style: "italic")

  show link: set text(fill: accent)
  show raw: set text(size: 8.8pt)

  set figure(gap: 0.9em)
  show figure.caption: set text(size: 8.8pt)
  show figure.caption: set par(justify: false)

  set table(stroke: (x, y) => (
    top: if y <= 1 { rule } else { 0pt },
    bottom: rule,
  ))
  show table.cell.where(y: 0): strong
  set table(inset: (x: 6pt, y: 4pt))

  // ── title block ──
  align(center, {
    text(size: 17pt, weight: 700, title)
    if subtitle != none { linebreak(); v(0.2em); text(size: 11pt, fill: muted, subtitle) }
    if authors != none { linebreak(); v(0.5em); text(size: 9.5pt, authors) }
    if date != none { linebreak(); v(0.2em); text(size: 9pt, fill: muted, date) }
  })
  v(1em)

  if abstract != none {
    block(width: 100%, inset: (x: 1.2cm), {
      align(center, text(size: 9.5pt, weight: 700, "Abstract"))
      v(0.3em)
      set text(size: 9.2pt)
      set par(justify: true)
      abstract
    })
    v(1.2em)
  }

  line(length: 100%, stroke: rule)
  v(0.4em)
  block(width: 100%, {
    set text(size: 9.2pt)
    show outline.entry.where(level: 1): set text(weight: 700)
    outline(title: text(size: 10pt, weight: 700, "Contents"), indent: 1.1em, depth: 2)
  })
  v(0.4em)
  line(length: 100%, stroke: rule)
  v(0.8em)

  body
}

// Place a generated figure: #fig(include "/figures/x.typ", caption: [...])
#let fig(body, caption: none) = figure(body, caption: caption, kind: image)

// A boxed aside for a claim, a caveat, or a result worth not missing.
#let callout(title: none, body) = block(
  width: 100%, inset: 9pt, radius: 3pt,
  fill: accent.lighten(93%), stroke: (left: 2pt + accent),
  {
    if title != none { text(weight: 700, size: 9.2pt, title); linebreak() }
    set text(size: 9.2pt)
    body
  },
)

// A compact key/value block for run settings.
#let kv(..pairs) = block(
  width: 100%,
  table(
    columns: (auto, 1fr),
    stroke: none,
    inset: (x: 0pt, y: 2.5pt),
    column-gutter: 12pt,
    ..pairs.pos().map(p => (
      text(fill: muted, size: 8.8pt, p.at(0)),
      text(size: 8.8pt, raw(str(p.at(1)))),
    )).flatten(),
  ),
)

// Inline term for a metric or symbol defined in the methodology.
#let m(body) = text(style: "italic", body)
