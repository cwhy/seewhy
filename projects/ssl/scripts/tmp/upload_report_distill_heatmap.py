"""Upload distillation heatmap report to R2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

url = save_report_file("ssl_report_distill_heatmap",
                       Path("projects/ssl/report-distill-heatmap.md"))
print(url)
