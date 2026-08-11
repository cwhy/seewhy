"""Publish a markdown report from `reports/` to R2 and print the URL.

The intermediate tier: terse lab-notebook write-ups, one per result, plots
already living on R2 as URLs. The paper is published separately with
`python -m shared_lib.publish projects/recall-gen/paper --stable`.

Usage:
    uv run python projects/recall-gen/scripts/publish.py reports/exp1-exp2.md
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared_lib.report import save_report_file

PROJECT = Path(__file__).parent.parent


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = PROJECT / path if (PROJECT / path).exists() else Path.cwd() / path
    name = sys.argv[2] if len(sys.argv) > 2 else f"recall-gen_report_{path.stem}"
    # No title= — it injects a duplicate <h1> over the markdown's own heading.
    print(save_report_file(name, path))


if __name__ == "__main__":
    main()
