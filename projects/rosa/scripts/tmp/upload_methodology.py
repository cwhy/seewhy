"""
Upload projects/rosa/methodology.md as HTML via shared_lib.report.save_report_file.

Usage: uv run python projects/rosa/scripts/tmp/upload_methodology.py
"""
import sys
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))

from shared_lib.report import save_report_file  # noqa: E402

MD_PATH = PROJ_DIR / "methodology.md"
print(f"Uploading {MD_PATH} ({MD_PATH.stat().st_size:,} bytes)...")
url = save_report_file("rosa_methodology", MD_PATH)
print(f"\nMethodology: {url}")
