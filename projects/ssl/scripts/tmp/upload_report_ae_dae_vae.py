"""Upload report-ae-dae-vae.md to R2 as HTML."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file

url = save_report_file(
    "ssl_report_ae_dae_vae",
    Path(__file__).parent.parent.parent / "report-ae-dae-vae.md",
)
print(url)
