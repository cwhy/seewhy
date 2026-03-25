"""Create experiments30.py — pixel-space merge with calibrated threshold."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments29.py").read_text()

src = src.replace(
    'exp29 — gravity in embedding space, merge decision in pixel space.\n\n'
    'Same as exp28 but the merge threshold is applied in decoded pixel space\n'
    '(L2 over 784 dims, values in [0,1]) rather than embedding space.\n'
    'Gravity still operates in embedding space to drive cluster movement.\n'
    'Two clusters are merged only when their decoded prototypes are visually\n'
    'similar, not just when their embeddings happen to be close.\n\n'
    'Key parameters:\n'
    '  GRAVITY_G      = 0.1  (gravitational constant)\n'
    '  MERGE_DIST_PIX = 2.0  (pixel-space L2 merge threshold, max ~28)',
    'exp30 — pixel-space merge with calibrated threshold.\n\n'
    'Same as exp29 but MERGE_DIST_PIX is calibrated from observed prototype\n'
    'distances on a converged exp29 run:\n'
    '  min pairwise pixel-L2 = 56,  p1 = 488,  mean = 1509\n'
    'Threshold of 200 sits between min and p1 — merges only near-duplicate\n'
    'prototypes while leaving well-separated clusters untouched.\n\n'
    'Key parameters:\n'
    '  GRAVITY_G      = 500.0  (gravitational constant)\n'
    '  MERGE_DIST_PIX = 200.0  (pixel-space L2 merge threshold)'
)

src = src.replace('EXP_NAME = "exp29"', 'EXP_NAME = "exp30"')
src = src.replace(
    "uv run python projects/imle-gram/experiments29.py",
    "uv run python projects/imle-gram/experiments30.py")
src = src.replace("imle-cluster-gravity-pixmerge", "imle-cluster-gravity-pixmerge-cal")

src = src.replace(
    'MERGE_DIST_PIX = 2.0    # pixel-space L2 merge threshold (max possible ~28)',
    'MERGE_DIST_PIX = 200.0  # pixel-space L2 merge threshold (calibrated: min=56, p1=488)'
)

dest = Path(__file__).parent.parent.parent / "experiments30.py"
dest.write_text(src)
print(f"wrote {dest}  ({len(src)} chars)")
