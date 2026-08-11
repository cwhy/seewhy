"""
One command for every report and paper tree in the repo.

    # what moved since the last publish
    uv run python -m shared_lib.publish projects/omniglot-ar/paper --status

    # structural + numeric lint, nothing compiled
    uv run python -m shared_lib.publish projects/omniglot-ar/paper --check

    # one section, fast, nothing uploaded
    uv run python -m shared_lib.publish projects/omniglot-ar/paper --section 06-results

    # whole tree to <dir>/out.pdf, still no upload
    uv run python -m shared_lib.publish projects/omniglot-ar/paper --preview

    # compile and push; prints the shareable URL
    uv run python -m shared_lib.publish projects/omniglot-ar/paper --stable

This replaces the per-project `publish_report.py` / `publish_paper.py`, which
were the same eighty lines copied once per tree. Name, title and show-rule are
read off the tree instead of being configured: `projects/omniglot-ar/paper`
publishes as `omniglot-ar_paper`, titled from `main.typ`.

Figures are NOT regenerated here — run the project's `gen_report.py` first.
This command compiles what is on disk, which is what makes `--status` and the
lint mean something.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .paper_lint import check_paper
from .typst_report import (
    ENTRY, changed_since_publish, preview, preview_section, publish_report,
    read_build_state, show_rule_fn,
)

_TITLE_RE = re.compile(r"^\s*title:\s*\"((?:[^\"\\]|\\.)*)\"", re.M)


def derive_title(report_dir: Path, entry: str = ENTRY) -> str | None:
    """The `title:` passed to the show rule in `main.typ`, if there is one."""
    path = report_dir / entry
    if not path.exists():
        return None
    m = _TITLE_RE.search(path.read_text(encoding="utf-8"))
    return m.group(1).encode().decode("unicode_escape") if m else None


def derive_name(report_dir: Path) -> str:
    """`projects/omniglot-ar/paper` → `omniglot-ar_paper`.

    The published key, so it has to be stable across machines: derived from the
    path rather than from anything inside the tree.
    """
    root = report_dir.resolve()
    project = root.parent.name
    return f"{project}_{root.name}" if project else root.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m shared_lib.publish",
        description="Compile, check and publish a Typst report or paper tree.",
    )
    ap.add_argument("report_dir", type=Path, help="the tree, e.g. projects/foo/paper")
    ap.add_argument("--status", action="store_true", help="show what changed since the last publish, then exit")
    ap.add_argument("--check", action="store_true", help="run the lint, then exit")
    ap.add_argument("--preview", action="store_true", help="compile to <dir>/out.pdf, no upload")
    ap.add_argument("--section", metavar="STEM", help="render one section to <dir>/out.pdf")
    ap.add_argument("--stable", action="store_true",
                    help="publish to a fixed URL, overwritten in place — for a living document "
                         "whose link is shared before it is finished")
    ap.add_argument("--with-svg", action="store_true", help="also upload an SVG rendering")
    ap.add_argument("--name", help="override the published key (default: <project>_<tree>)")
    ap.add_argument("--title", help="override the title (default: read from main.typ)")
    ap.add_argument("--no-lint", action="store_true", help="publish without checking first")
    ap.add_argument("--include-curves", action="store_true",
                    help="lint: also match prose numbers against per-epoch curves (less sensitive)")
    ap.add_argument("--unit-scales", action="store_true",
                    help="lint: also match K/M/G-scaled forms such as '3.38 M' (less sensitive)")
    args = ap.parse_args(argv)

    root: Path = args.report_dir
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    if not (root / ENTRY).exists():
        print(f"no {ENTRY} in {root} — is this a report tree?", file=sys.stderr)
        return 2

    def lint():
        return check_paper(root, include_curves=args.include_curves, unit_scales=args.unit_scales)

    if args.check:
        report = lint()
        print(report)
        return 1 if report.errors else 0

    state = read_build_state(root)
    if state:
        print(f"last published {state['published_at']} → {state['urls']['page']}")
    else:
        print("never published from this checkout")
    print(f"changes: {changed_since_publish(root)}")

    if args.status:
        return 0

    if args.section:
        out = preview_section(root, args.section, root / "out.pdf", fmt="pdf",
                              template_fn=show_rule_fn(root))
        print(f"rendered {args.section} → {out} ({out.stat().st_size:,} bytes)")
        return 0

    if args.preview:
        out = preview(root, root / "out.pdf")
        print(f"compiled → {out} ({out.stat().st_size:,} bytes)")
        return 0

    if not args.no_lint:
        report = lint()
        if report.findings:
            print(report)
        if report.errors:
            print("\nrefusing to publish with errors — fix them or pass --no-lint", file=sys.stderr)
            return 1

    urls = publish_report(
        args.name or derive_name(root),
        root,
        title=args.title or derive_title(root),
        with_svg=args.with_svg,
        stable=args.stable,
    )
    print(f"\n  page  {urls.page}\n  pdf   {urls.pdf}")
    if urls.svg:
        print(f"  svg   {urls.svg}")
    if args.stable:
        print("\n  stable URL — overwritten on every publish, so a shared link stays current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
