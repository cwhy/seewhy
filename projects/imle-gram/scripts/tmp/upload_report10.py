"""Upload report_10.md to R2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.report import save_report_file

url = save_report_file(
    "imle_report_10",
    Path(__file__).parent.parent.parent / "report_10.md",
    title="Report 10 — Fashion-MNIST and Codebook Diversity (exp24–exp33)"
)
print(url)
