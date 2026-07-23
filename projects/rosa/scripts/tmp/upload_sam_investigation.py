"""Upload projects/rosa/sam_pollution_investigation.md as HTML via shared_lib.report."""
import sys
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR.parent.parent))

from shared_lib.report import save_report_file  # noqa: E402

MD = PROJ_DIR / "sam_pollution_investigation.md"
print(f"Uploading {MD} ({MD.stat().st_size:,} bytes)...")
url = save_report_file("rosa_sam_pollution_investigation", MD)
print(f"\nReport: {url}")
