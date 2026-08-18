// Paper template — title block, abstract, numbered sections, outline.
//
// Scaffolded by shared_lib.typst_report.scaffold_paper(). Yours to edit; the
// library never rewrites it.
//
// The audience is technical but NOT an ML specialist: someone who writes code
// and reads statistics, but has never trained a transformer. `#gloss` and
// `#notation` exist to make that obligation cheap to meet.

#let accent = rgb("#0072b2")
#let muted = luma(105)
#let flag = rgb("#c1121f")
#let rule = 0.5pt + luma(180)

// Document state, so `#todo` further down the document can see what `paper()`
// was called with. A plain variable would not work — sections are separate
// files evaluated in their own scope.
#let paper-status = state("paper-status", "draft")

// `web: true` collapses the paper to one continuous page. The published PDF
// uses paged A4; the flag exists for a scrolling preview.
//
// `status` is "draft" while sections are still stubs and "final" when the
// paper is done. It is not decoration: in "final" every #todo is a hard
// compile error, so a paper cannot be published with holes in it.
#let paper(
  title: "",
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  status: "draft",
  web: false,
  body,
) = {
  set document(title: title)
  paper-status.update(status)
  set page(
    width: 21cm,
    height: if web { auto } else { 29.7cm },
    margin: (x: 2.4cm, top: 2.4cm, bottom: 2.2cm),
    numbering: if web { none } else { "1" },
    number-align: center,
  )
  // No font family: Typst's bundled default renders identically on macOS and
  // on the GPU box, where the paper is actually compiled. Naming a face here
  // silently falls back to a different one on Linux.
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

  if status == "draft" {
    block(
      width: 100%, inset: 7pt, radius: 3pt,
      fill: flag.lighten(92%), stroke: (left: 2pt + flag),
      text(size: 9pt, fill: flag.darken(20%))[
        *Draft.* Sections are still being written; numbers and conclusions may
        change. Unwritten passages are marked #box(fill: flag.lighten(80%),
        inset: (x: 3pt, y: 1pt), radius: 2pt, text(size: 8pt, weight: 700,
        fill: flag.darken(20%), "TODO")) in the body.
      ],
    )
    v(0.9em)
  }

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

// ─────────────────────────────── helpers ────────────────────────────────────

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

// First-use definition of a term the reader is not assumed to know.
// Renders the term in italic followed by the definition in parentheses:
//     #gloss[induction head][a circuit that copies from an earlier match]
// Use it the FIRST time a piece of ML vocabulary appears. The reader knows
// programming and statistics; they do not know what attention is.
#let gloss(term, definition) = [#text(style: "italic", term) (#definition)]

// Symbol table for the methodology section. One row per symbol:
//     #notation(
//       ($N$, [number of support images per episode]),
//       ($p_theta$, [the model's predicted distribution over values]),
//     )
#let notation(..rows) = block(
  width: 100%,
  table(
    columns: (auto, 1fr),
    stroke: none,
    inset: (x: 0pt, y: 3pt),
    column-gutter: 14pt,
    ..rows.pos().map(r => (
      text(size: 9.2pt, r.at(0)),
      text(size: 9.2pt, r.at(1)),
    )).flatten(),
  ),
)

// An unwritten passage. Visible in a draft, fatal in the final build — which
// is the whole point: "I'll fill that in before publishing" is exactly the
// promise that does not survive contact with a deadline. Typst reports the
// error at this call site, so the panic message need not locate it.
#let todo(body) = context {
  if paper-status.get() == "final" {
    panic("unwritten #todo in a paper built with status: \"final\"")
  }
  box(
    fill: flag.lighten(80%), inset: (x: 3pt, y: 1pt), radius: 2pt,
    text(size: 8.5pt, weight: 700, fill: flag.darken(20%), [TODO: #body]),
  )
}
