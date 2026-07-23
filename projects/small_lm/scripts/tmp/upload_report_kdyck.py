"""Upload report-kdyck.md as HTML to R2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

REPORT = Path(__file__).parent.parent.parent / "report-kdyck.md"

if __name__ == "__main__":
    url = save_report_file("small_lm_report_kdyck", REPORT)
    print(url)
