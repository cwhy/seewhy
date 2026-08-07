"""
Compile the Typst report and push it to R2.

    # what moved since the last publish
    uv run python projects/omniglot-ar/scripts/publish_report.py --status

    # fast local iteration on one section — compiles nothing else, uploads nothing
    uv run python projects/omniglot-ar/scripts/publish_report.py --section 04-results

    # whole report to a local PDF, still no upload
    uv run python projects/omniglot-ar/scripts/publish_report.py --preview

    # compile and push; prints the shareable URL
    uv run python projects/omniglot-ar/scripts/publish_report.py

Run `gen_report.py` first if results have changed — this script compiles what is
on disk and does not regenerate figures.
"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from shared_lib.typst_report import (
    changed_since_publish, preview, preview_section, publish_report, read_build_state,
)

REPORT = PROJECT / "report"
NAME = "omniglot-ar_report"
TITLE = "Omniglot AR — exp1 & exp2"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile and publish the omniglot-ar report")
    ap.add_argument("--status", action="store_true", help="show changes and exit")
    ap.add_argument("--preview", action="store_true", help="compile to report/out.pdf, no upload")
    ap.add_argument("--section", metavar="STEM", help="render one section to report/out.svg")
    args = ap.parse_args()

    changes = changed_since_publish(REPORT)
    state = read_build_state(REPORT)
    if state:
        print(f"last published {state['published_at']} → {state['urls']['page']}")
    print(f"changes: {changes}")

    if args.status:
        return

    if args.section:
        out = preview_section(REPORT, args.section, REPORT / "out.svg")
        print(f"rendered {args.section} → {out}")
        return

    if args.preview:
        out = preview(REPORT, REPORT / "out.pdf")
        print(f"compiled → {out} ({out.stat().st_size:,} bytes)")
        return

    if not changes.any and state:
        print("nothing changed — publishing anyway to refresh the URL")

    urls = publish_report(NAME, REPORT, title=TITLE)
    print(f"\n  report  {urls.page}\n  pdf     {urls.pdf}\n  svg     {urls.svg}")


if __name__ == "__main__":
    main()
