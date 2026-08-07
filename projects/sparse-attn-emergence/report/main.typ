// Report entry point. The section list below IS the report's structure —
// add, remove, or reorder #include lines; each section is an independent file.

#import "/template.typ": *

#show: report.with(
  title: "What kind of pattern is hard to learn?",
  subtitle: "Position-keyed vs content-keyed routing, and why the architecture ranking inverts",
  date: "2026-08-07",
  web: sys.inputs.at("web", default: "0") == "1",
)

#include "/sections/01-question.typ"
#include "/sections/02-tasks.typ"
#include "/sections/03-positional.typ"
#include "/sections/04-content.typ"
#include "/sections/05-memorisation.typ"
#include "/sections/06-reading.typ"
