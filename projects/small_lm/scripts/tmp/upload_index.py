"""
Upload index.md as HTML to R2.

Usage:
    uv run python projects/small_lm/scripts/tmp/upload_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

INDEX = Path(__file__).parent.parent.parent / "index.md"

if __name__ == "__main__":
    url = save_report_file("small_lm_index", INDEX)
    print(url)
