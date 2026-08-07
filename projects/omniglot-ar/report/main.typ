// Report entry point. The section list below IS the report's structure —
// add, remove, or reorder #include lines; each section is an independent file.

#import "/template.typ": *

#show: report.with(
  title: "Omniglot AR — exp1 & exp2",
  subtitle: "Token-level in-context classification on class-disjoint Omniglot",
  date: "2026-08-06",
  web: sys.inputs.at("web", default: "0") == "1",
)

#include "/sections/01-question.typ"
#include "/sections/02-substrate.typ"
#include "/sections/03-setup.typ"
#include "/sections/04-results.typ"
#include "/sections/05-reading.typ"
