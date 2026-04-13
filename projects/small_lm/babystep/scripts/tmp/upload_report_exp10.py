import sys
sys.path.insert(0, "/home/newuser/Projects/seewhy")
from shared_lib.report import save_report_file

url = save_report_file(
    "small_lm_babystep_kylo_exp10_saturation",
    "/home/newuser/Projects/seewhy/projects/small_lm/babystep/experiment-10-kylo-saturation/report_manual.md",
    title="Experiment-10: Kylo Saturation Metric",
)
print(url)
