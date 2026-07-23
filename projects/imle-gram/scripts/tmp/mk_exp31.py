"""Create experiments31.py — pixel-space merge with warmup + tighter threshold."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments30.py").read_text()

src = src.replace(
    'exp30 — pixel-space merge with calibrated threshold.\n\n'
    'Same as exp29 but MERGE_DIST_PIX is calibrated from observed prototype\n'
    'distances on a converged exp29 run:\n'
    '  min pairwise pixel-L2 = 56,  p1 = 488,  mean = 1509\n'
    'Threshold of 200 sits between min and p1 — merges only near-duplicate\n'
    'prototypes while leaving well-separated clusters untouched.\n\n'
    'Key parameters:\n'
    '  GRAVITY_G      = 500.0  (gravitational constant)\n'
    '  MERGE_DIST_PIX = 200.0  (pixel-space L2 merge threshold)',
    'exp31 — pixel-space merge with warmup + tighter threshold.\n\n'
    'exp30 diagnosis: merging 25-63 clusters/epoch during epochs 0-10 because\n'
    'unconverged prototypes are random and many fall within 200px L2 by chance.\n'
    'This collapses lbl_bayes to ~10% early on. Fix:\n'
    '  1. MERGE_WARMUP=20 — skip gravity+merge for first 20 epochs. Let\n'
    '     gradient training establish meaningful prototypes first.\n'
    '  2. MERGE_DIST_PIX=100 — tighter threshold; exp29 min was 56, p1=488.\n'
    '     100 catches only the closest ~0.05% of pairs at convergence.\n\n'
    'Key parameters:\n'
    '  GRAVITY_G      = 500.0  (gravitational constant)\n'
    '  MERGE_WARMUP   = 20     (epochs before gravity+merge activates)\n'
    '  MERGE_DIST_PIX = 100.0  (pixel-space L2 merge threshold)'
)

src = src.replace('EXP_NAME = "exp30"', 'EXP_NAME = "exp31"')
src = src.replace(
    "uv run python projects/imle-gram/experiments30.py",
    "uv run python projects/imle-gram/experiments31.py")
src = src.replace("imle-cluster-gravity-pixmerge-cal", "imle-cluster-gravity-pixmerge-warm")

src = src.replace(
    'MERGE_DIST_PIX = 200.0  # pixel-space L2 merge threshold (calibrated: min=56, p1=488)',
    'MERGE_WARMUP   = 20     # epochs before gravity+merge activates\n'
    'MERGE_DIST_PIX = 100.0  # pixel-space L2 merge threshold (tighter: min=56, p1=488)'
)

# Guard apply_gravity_and_merge call with warmup check
src = src.replace(
    '        params, opt_state, grav_disp, n_merged = apply_gravity_and_merge(\n'
    '            params, opt_state, usage, epoch)',
    '        if epoch >= MERGE_WARMUP:\n'
    '            params, opt_state, grav_disp, n_merged = apply_gravity_and_merge(\n'
    '                params, opt_state, usage, epoch)\n'
    '        else:\n'
    '            grav_disp, n_merged = 0.0, 0'
)

# Update subtitle
src = src.replace(
    'save_learning_curves(history, EXP_NAME, subtitle=f"gravity G={GRAVITY_G} pix_merge={MERGE_DIST_PIX} N_CODES={N_CODES}")',
    'save_learning_curves(history, EXP_NAME, subtitle=f"gravity G={GRAVITY_G} warmup={MERGE_WARMUP} pix_merge={MERGE_DIST_PIX} N_CODES={N_CODES}")'
)

dest = Path(__file__).parent.parent.parent / "experiments31.py"
dest.write_text(src)
print(f"wrote {dest}  ({len(src)} chars)")
