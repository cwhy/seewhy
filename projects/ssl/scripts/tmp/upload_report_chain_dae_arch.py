import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.report import save_report_file
url = save_report_file("ssl_report_distill_chain_dae_arch",
                       Path("projects/ssl/report-distill-chain-dae-arch.md"))
print(url)
