import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared_lib.report import save_report_file

url = save_report_file("universal-ar_report_progress", Path("projects/universal-ar/report-progress.md"))
print("PUBLISHED_URL=", url)
