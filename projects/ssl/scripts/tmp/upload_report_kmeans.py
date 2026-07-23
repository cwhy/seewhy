"""Upload report-kmeans.md to R2 as HTML."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

url = save_report_file(
    "ssl_report_kmeans",
    Path(__file__).parent.parent.parent / "report-kmeans.md",
    title="K-Means Clustering Evaluation — EMA-JEPA SSL Encoders"
)
print(url)
