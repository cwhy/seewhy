"""Create experiments33.py — aggressive pixel-space merge threshold."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments32.py").read_text()

src = src.replace('EXP_NAME = "exp32"', 'EXP_NAME = "exp33"')
src = src.replace(
    "uv run python projects/imle-gram/experiments32.py",
    "uv run python projects/imle-gram/experiments33.py")
src = src.replace("imle-cluster-pixmerge", "imle-cluster-pixmerge-agg")

src = src.replace(
    'exp32 — pixel-space merge only, no gravity.\n\n'
    'exp31 diagnosis: gravity (G=500) creates a permanent opposing force against\n'
    'gradient learning — gradient pulls clusters to their data, gravity pulls them\n'
    'toward each other. They fight constantly; match_l2 stays high (~3000 vs ~2500\n'
    'baseline) and lbl_bayes oscillates 10-54%.\n\n'
    'Fix: drop gravity entirely. Keep pixel-space merge check after warmup.\n'
    'Gradient learning runs undisturbed. We only intervene when two prototypes\n'
    'genuinely converge to the same output through natural training dynamics.\n\n'
    'Key parameters:\n'
    '  MERGE_WARMUP   = 20    (epochs before merge check activates)\n'
    '  MERGE_DIST_PIX = 100.0 (pixel-space L2 merge threshold; min=56, p1=488)',
    'exp33 — aggressive pixel-space merge threshold.\n\n'
    'exp32 showed zero merges at threshold=100 (converged min=411, p1=701).\n'
    'Visualization confirms some visually similar prototypes exist.\n'
    'Raising threshold to 800 to catch ~1-2% closest pairs per check.\n\n'
    'Key parameters:\n'
    '  MERGE_WARMUP   = 20    (epochs before merge check activates)\n'
    '  MERGE_DIST_PIX = 800.0 (pixel-space L2 merge threshold; exp32 p1=701, p5=956)'
)

src = src.replace(
    'MERGE_WARMUP   = 20     # epochs before merge check activates\n'
    'MERGE_DIST_PIX = 100.0  # pixel-space L2 merge threshold (min=56, p1=488)',
    'MERGE_WARMUP   = 20     # epochs before merge check activates\n'
    'MERGE_DIST_PIX = 800.0  # pixel-space L2 merge threshold (exp32 p1=701, p5=956)'
)

src = src.replace(
    'save_learning_curves(history, EXP_NAME, subtitle=f"pix_merge warmup={MERGE_WARMUP} dist={MERGE_DIST_PIX} N_CODES={N_CODES}")',
    'save_learning_curves(history, EXP_NAME, subtitle=f"pix_merge warmup={MERGE_WARMUP} dist={MERGE_DIST_PIX} N_CODES={N_CODES}")'
)

dest = Path(__file__).parent.parent.parent / "experiments33.py"
dest.write_text(src)
print(f"wrote {dest}  ({len(src)} chars)")
