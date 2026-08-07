"""
Compile the paper-style report and push it to R2.

Same loop as publish_report.py, pointed at `paper/` instead of `report/`. Only
the PDF is stored; the shared link is a ~3 KB page that renders it with pdf.js
from a CDN.

    # what moved since the last publish
    uv run python projects/omniglot-ar/scripts/publish_paper.py --status

    # one section, fast, nothing uploaded
    uv run python projects/omniglot-ar/scripts/publish_paper.py --section 06-results

    # whole paper to paper/out.pdf, still no upload
    uv run python projects/omniglot-ar/scripts/publish_paper.py --preview

    # compile and push; prints the shareable URL
    uv run python projects/omniglot-ar/scripts/publish_paper.py

Run `gen_report.py --into paper` first if results have changed — this script
compiles what is on disk and does not regenerate figures.
"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from shared_lib.typst_report import (
    changed_since_publish, preview, preview_section, publish_report, read_build_state,
)

PAPER = PROJECT / "paper"
NAME = "omniglot-ar_paper"
TITLE = "Token-level in-context classification on Omniglot"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile and publish the omniglot-ar paper")
    ap.add_argument("--status", action="store_true", help="show changes and exit")
    ap.add_argument("--preview", action="store_true", help="compile to paper/out.pdf, no upload")
    ap.add_argument("--section", metavar="STEM", help="render one section to paper/out.pdf")
    ap.add_argument("--with-svg", action="store_true",
                    help="also upload an SVG rendering (costs R2 space; off by default)")
    args = ap.parse_args()

    changes = changed_since_publish(PAPER)
    state = read_build_state(PAPER)
    if state:
        print(f"last published {state['published_at']} → {state['urls']['page']}")
    print(f"changes: {changes}")

    if args.status:
        return

    if args.section:
        out = preview_section(PAPER, args.section, PAPER / "out.pdf",
                              fmt="pdf", template_fn="paper")
        print(f"rendered {args.section} → {out} ({out.stat().st_size:,} bytes)")
        return

    if args.preview:
        out = preview(PAPER, PAPER / "out.pdf")
        print(f"compiled → {out} ({out.stat().st_size:,} bytes)")
        return

    urls = publish_report(NAME, PAPER, title=TITLE, with_svg=args.with_svg)
    print(f"\n  paper  {urls.page}\n  pdf    {urls.pdf}")
    if urls.svg:
        print(f"  svg    {urls.svg}")


if __name__ == "__main__":
    main()
