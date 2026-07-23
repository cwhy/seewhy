"""
Convert report-optimizations.md to HTML and upload to R2.

Usage:
    uv run python projects/small_lm/scripts/tmp/upload_report_optimizations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

REPORT = Path(__file__).parent.parent.parent / "report-optimizations.md"

if __name__ == "__main__":
    url = save_report_file("small_lm_report_optimizations", REPORT)
    print(url)
