"""
Checks a Typst report or paper tree before it is published.

Two kinds of problem, both of which have actually happened in this repo:

  structural — a section file nobody includes, an include pointing at nothing,
               a figure whose data was never generated, a citation with no
               bibliography entry, an unwritten `#todo`. `sparse-attn-emergence`
               carries an orphaned `06-reading.typ` superseded by `07-reading`.

  numeric    — a number in the prose that no longer matches `results.jsonl`.
               Prose numbers are copied by hand once and then outlive the run
               that produced them; a paper republished as work proceeds is
               exposed to this on every rerun.

The numeric check is deliberately advisory. It reads the scalars in
`results.jsonl` and asks whether each literal in the prose could have come from
one of them, at the precision it was written to: `0.228` matches a stored
`0.22814`. Literals that legitimately are not results — years, counts,
hardware, arithmetic done in the text — go in `.lint-allow`, one per line with
a reason after `#`. A growing allow-list is itself worth noticing, which is why
they are listed in a file rather than marked invisibly in the prose.

Calibration, measured on `omniglot-ar`'s finished paper (95 distinct literals,
10 result rows). Sensitivity is the share of matching literals that stop
matching when perturbed by one unit in their last written place — i.e. the
share of *actually stale* numbers the check would catch:

    pool             scaling      matched   missed when stale
    with curves      K/M/G         71/95          35%
    with curves      none          64/95          28%
    scalars only     K/M/G         57/95          26%
    scalars only     none          51/95          18%   <- default

Two things follow. Per-epoch curves must stay out of the pool: a curve covers
its whole range at fine resolution, so nearly any plausible number matches a
point on one, and including them nearly doubles the miss rate. And the check is
a net, not a proof — roughly one stale number in five slips through even at the
sensitive setting, because a coarse literal like `0.3` is not discriminating
against any realistic pool. Treat a clean numeric report as "nothing obvious",
never as "the numbers are verified".

    from shared_lib.paper_lint import check_paper
    report = check_paper("projects/omniglot-ar/paper")
    print(report)
    if report.errors:
        raise SystemExit(1)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .typst_report import label_uses

ALLOW_FILE = ".lint-allow"

# Lists longer than this are per-epoch curves rather than summary values, and
# are kept out of the pool the prose is checked against. See the calibration
# table above: including them nearly doubles the miss rate.
MAX_LIST = 8

# Structural problems that make the document wrong rather than unfinished.
# Only these block a publish; everything else is reported and moves on.
ERROR_KINDS = frozenset({"missing-include", "missing-figure", "missing-asset"})

# `#todo[...]`, the scaffold's unwritten-passage marker.
_TODO_RE = re.compile(r"#todo\[")
_INCLUDE_RE = re.compile(r'#include\s+"(/[^"]+)"')
_FIG_INCLUDE_RE = re.compile(r'include\s+"(/figures/[^"]+\.typ)"')
_JSON_DATA_RE = re.compile(r'json\("(/[^"]+)"\)')
# Typst spells a citation and a cross-reference the same way, `@key`. Keys may
# contain dots and colons but never end in one, so a sentence-final period is
# not part of the key. Which of the two a given `@key` is can only be settled
# by looking for a matching `<label>` in the document — see _check_citations.
_CITE_RE = re.compile(r"(?<![\w@])@([A-Za-z][\w:.-]*[\w]|[A-Za-z])")
_BIB_KEY_RE = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", re.M)

# Typst line comments and raw blocks hold code, paths and version numbers that
# are not claims about results. Stripped before any number is looked at.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_RAW_BLOCK_RE = re.compile(r"```.*?```", re.S)
_RAW_INLINE_RE = re.compile(r"`[^`\n]*`")

# A number as it appears in prose. Leading sign is excluded: a hyphen in text
# is far more often a dash or a range than a negative sign, and the magnitude
# is what gets checked anyway.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\w.])")

# Numbers that are almost never claims about measured results. Section numbers,
# small counts, and plain years dominate the false positives otherwise.
_YEAR_RANGE = (1900, 2100)
_SMALL_INT_MAX = 20


@dataclass(frozen=True)
class Finding:
    kind: str
    message: str
    where: str = ""

    @property
    def is_error(self) -> bool:
        return self.kind in ERROR_KINDS

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{'error' if self.is_error else 'warn '}  {self.kind}: {self.message}{loc}"


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)
    checked_numbers: int = 0
    results_values: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.is_error]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.is_error]

    def add(self, kind: str, message: str, where: str = "") -> None:
        self.findings.append(Finding(kind, message, where))

    def __str__(self) -> str:
        if not self.findings:
            return (
                f"clean — {self.checked_numbers} prose numbers checked against "
                f"{self.results_values} values from results.jsonl"
            )
        lines = [str(f) for f in self.findings]
        lines.append(
            f"\n{len(self.errors)} error(s), {len(self.warnings)} warning(s); "
            f"{self.checked_numbers} prose numbers checked against "
            f"{self.results_values} values from results.jsonl"
        )
        return "\n".join(lines)


# ──────────────────────────────── helpers ───────────────────────────────────


def _prose(text: str) -> str:
    """Strip comments and code so only what a reader sees is examined."""
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub(" ", text)
    text = _RAW_BLOCK_RE.sub(" ", text)
    return _RAW_INLINE_RE.sub(" ", text)


def _iter_scalars(obj, max_list: int | None = MAX_LIST) -> Iterator[float]:
    """Every finite number in a results row.

    Lists longer than `max_list` are skipped: those are per-epoch curves, and
    including them is what destroys the check's power. A curve spans its whole
    range at fine resolution, so almost any plausible number matches a point on
    one — see the calibration note in the module docstring. Detected by length
    rather than by key name, since projects call the field different things.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_scalars(v, max_list)
    elif isinstance(obj, (list, tuple)):
        if max_list is not None and len(obj) > max_list:
            return
        for v in obj:
            yield from _iter_scalars(v, max_list)


def load_results_values(results_path: str | Path, *, include_curves: bool = False) -> set[float]:
    """Every finite number in a results.jsonl, flattened.

    Per-epoch curves are excluded unless `include_curves` — they make the pool
    dense enough that the numeric check stops discriminating.
    """
    path = Path(results_path)
    if not path.exists():
        return set()
    max_list = None if include_curves else MAX_LIST
    values: set[float] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        values.update(_iter_scalars(row, max_list))
    return values


def _precision(literal: str) -> int:
    """Decimal places the literal was written to, as an `ndigits` for round().

    Via Decimal so exponent form works: `3e-4` is precise to 4 places, not 0.
    Reading it off the string after a `float()` round-trip loses exactly that.
    """
    try:
        exponent = Decimal(literal).as_tuple().exponent
    except InvalidOperation:
        return 0
    return -int(exponent)


def _matches_any(literal: str, values: Iterable[float], *, unit_scales: bool = False) -> bool:
    """Could this literal have been rounded from one of these values?

    Matched at the precision it was written to, so `0.228` accepts a stored
    `0.22814` — which is how a number actually travels from results.jsonl into
    prose. Derived quantities (a ratio, a sum) will not match, and that is what
    `.lint-allow` is for.

    `unit_scales` also accepts K/M/G-scaled forms, so "3.38 M parameters"
    matches a stored `3378186`. It is off by default: it buys ~6 fewer warnings
    on a real paper and costs 8 points of sensitivity, which is the wrong trade
    for a check whose entire job is noticing a number that moved.
    """
    try:
        target = float(literal)
    except ValueError:
        return False
    nd = _precision(literal)
    # Percentages are always accepted: prose says 22.8, results.jsonl stores
    # 0.228, and that is a presentation choice rather than a different number.
    factors = (1.0, 100.0) + ((1e-3, 1e-6, 1e-9) if unit_scales else ())
    for v in values:
        for f in factors:
            if round(v * f, nd) == target:
                return True
    return False


def _is_uninteresting(literal: str) -> bool:
    """Numbers that are structurally not claims about measurements."""
    try:
        value = float(literal)
    except ValueError:
        return True
    if _precision(literal) <= 0:
        if value.is_integer() and 0 <= value <= _SMALL_INT_MAX:
            return True          # counts, section numbers, "seven runs"
        if _YEAR_RANGE[0] <= value <= _YEAR_RANGE[1] and value.is_integer():
            return True          # years
    return False


def load_allow(paper_dir: str | Path) -> set[str]:
    """Literals the author has declared intentional, from `.lint-allow`."""
    path = Path(paper_dir) / ALLOW_FILE
    if not path.exists():
        return set()
    allowed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            allowed.add(entry)
    return allowed


# ───────────────────────────────── checks ───────────────────────────────────


def _check_structure(root: Path, entry: str, report: LintReport) -> list[Path]:
    """Includes, orphans, figures, assets. Returns the included section files."""
    entry_path = root / entry
    if not entry_path.exists():
        report.add("missing-include", f"no entry file {entry}", str(root))
        return []

    entry_text = entry_path.read_text(encoding="utf-8")
    included: list[Path] = []
    for rel in _INCLUDE_RE.findall(entry_text):
        target = root / rel.lstrip("/")
        if not target.exists():
            report.add("missing-include", f"{entry} includes {rel}, which does not exist", entry)
        else:
            included.append(target)

    sections_dir = root / "sections"
    if sections_dir.is_dir():
        included_set = {p.resolve() for p in included}
        for path in sorted(sections_dir.glob("*.typ")):
            if path.resolve() not in included_set:
                report.add(
                    "orphan-section",
                    f"{path.name} is not included by {entry} — dead file or a rename left behind",
                    f"sections/{path.name}",
                )

    # Figures placed in prose must exist, and must have their data beside them.
    for path in included:
        text = path.read_text(encoding="utf-8")
        for rel in _FIG_INCLUDE_RE.findall(_prose(text)):
            fig_path = root / rel.lstrip("/")
            if not fig_path.exists():
                report.add(
                    "missing-figure",
                    f"places {rel}, which has not been generated — run the project's gen_report.py",
                    path.name,
                )
                continue
            for data_rel in _JSON_DATA_RE.findall(fig_path.read_text(encoding="utf-8")):
                if not (root / data_rel.lstrip("/")).exists():
                    report.add(
                        "missing-asset",
                        f"{rel} reads {data_rel}, which does not exist",
                        fig_path.name,
                    )
    return included


def _check_labels_and_citations(root: Path, sections: Sequence[Path], report: LintReport) -> None:
    texts = {path: _prose(path.read_text(encoding="utf-8")) for path in sections}

    defined: set[str] = set()
    referenced: dict[str, str] = {}
    for path, text in texts.items():
        here_defined, here_referenced = label_uses(text)
        defined |= here_defined
        for key in here_referenced:
            referenced.setdefault(key, path.name)

    bib = root / "refs.bib"
    keys = set(_BIB_KEY_RE.findall(bib.read_text(encoding="utf-8"))) if bib.exists() else set()

    # A label pointed at but defined nowhere fails the whole compile, and the
    # message Typst gives names the label rather than the file that used it.
    for key, where in sorted(referenced.items()):
        if key not in defined and key not in keys:
            kind = "unknown-citation" if bib.exists() else "dangling-label"
            report.add(
                kind,
                f"@{key} is neither defined as a label in this tree nor an entry in refs.bib",
                where,
            )

    if not bib.exists():
        return
    cited = {k for k in referenced if k not in defined}
    for unused in sorted(keys - cited):
        report.add("uncited-reference", f"refs.bib entry {unused} is never cited", "refs.bib")


def _check_todos(sections: Sequence[Path], report: LintReport) -> None:
    for path in sections:
        n = len(_TODO_RE.findall(path.read_text(encoding="utf-8")))
        if n:
            report.add("unwritten", f"{n} #todo remaining", path.name)


def _check_numbers(
    root: Path,
    sections: Sequence[Path],
    values: set[float],
    report: LintReport,
    *,
    unit_scales: bool = False,
) -> None:
    if not values:
        report.add("no-results", "results.jsonl is empty or absent — numbers not checked", "")
        return
    allowed = load_allow(root)

    # Deduplicated across the whole tree: a number repeated in three sections
    # is one thing to check, not three, and listing it three times buries the
    # rest of the report.
    where: dict[str, list[str]] = {}
    for path in sections:
        for literal in _NUMBER_RE.findall(_prose(path.read_text(encoding="utf-8"))):
            if literal in allowed or _is_uninteresting(literal):
                continue
            sections_seen = where.setdefault(literal, [])
            if path.name not in sections_seen:
                sections_seen.append(path.name)

    report.checked_numbers = len(where)
    for literal, in_sections in where.items():
        if not _matches_any(literal, values, unit_scales=unit_scales):
            report.add(
                "unmatched-number",
                f"{literal} does not appear in results.jsonl — stale, derived, "
                f"or belongs in {ALLOW_FILE}",
                ", ".join(in_sections),
            )


def check_paper(
    paper_dir: str | Path,
    *,
    results: str | Path | None = None,
    entry: str = "main.typ",
    check_numbers: bool = True,
    include_curves: bool = False,
    unit_scales: bool = False,
) -> LintReport:
    """Lint a report or paper tree. Returns findings; raises nothing.

    `results` defaults to `results.jsonl` in the project directory holding the
    tree — `projects/foo/paper` → `projects/foo/results.jsonl`.

    `include_curves` and `unit_scales` loosen the numeric check; both cost
    sensitivity and are off by default (see the calibration in the module
    docstring). Turn them on to quieten a paper that legitimately quotes curve
    points or writes magnitudes in millions.
    """
    root = Path(paper_dir)
    report = LintReport()

    sections = _check_structure(root, entry, report)
    _check_labels_and_citations(root, sections, report)
    _check_todos(sections, report)

    if check_numbers:
        results_path = Path(results) if results else root.parent / "results.jsonl"
        values = load_results_values(results_path, include_curves=include_curves)
        report.results_values = len(values)
        _check_numbers(root, sections, values, report, unit_scales=unit_scales)

    return report
