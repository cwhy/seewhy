"""Upload report-kmeans-comparison.md to R2 as HTML."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

url = save_report_file(
    "ssl_report_kmeans_comparison",
    Path(__file__).parent.parent.parent / "report-kmeans-comparison.md",
    title="K-Means Comparison: Random vs SIGReg vs EMA — SSL Encoders"
)
print(url)
