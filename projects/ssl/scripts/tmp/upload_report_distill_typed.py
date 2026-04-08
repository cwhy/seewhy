"""Upload all 3 typed distillation heatmap reports to R2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

SSL = Path("projects/ssl")
for t in ["sigreg", "dae", "random"]:
    url = save_report_file(f"ssl_report_distill_heatmap_{t}",
                           SSL / f"report-distill-heatmap-{t}.md")
    print(f"{t}: {url}")
