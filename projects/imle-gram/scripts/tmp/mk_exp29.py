"""Create experiments29.py — gravity in embedding space, merge decision in pixel space."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments28.py").read_text()

# 1. Docstring
src = src.replace(
    'exp28 — 128-cluster flat delta_sum + gravity-driven codebook maintenance.\n\n'
    'Same as exp21 but after each epoch applies Newtonian gravity between all\n'
    'codebook pairs. Mass = fraction of epoch assignments (data-dense clusters\n'
    'attract more strongly). Pairs that end up within MERGE_DIST are merged\n'
    'and the lighter cluster is reinitialised from scratch.\n\n'
    'Key parameters:\n'
    '  GRAVITY_G  = 0.1   (gravitational constant)\n'
    '  MERGE_DIST = 0.15  (L2 merge threshold in embedding space)',
    'exp29 — gravity in embedding space, merge decision in pixel space.\n\n'
    'Same as exp28 but the merge threshold is applied in decoded pixel space\n'
    '(L2 over 784 dims, values in [0,1]) rather than embedding space.\n'
    'Gravity still operates in embedding space to drive cluster movement.\n'
    'Two clusters are merged only when their decoded prototypes are visually\n'
    'similar, not just when their embeddings happen to be close.\n\n'
    'Key parameters:\n'
    '  GRAVITY_G      = 0.1  (gravitational constant)\n'
    '  MERGE_DIST_PIX = 2.0  (pixel-space L2 merge threshold, max ~28)'
)

# 2. EXP_NAME and run command
src = src.replace('EXP_NAME = "exp28"', 'EXP_NAME = "exp29"')
src = src.replace(
    "uv run python projects/imle-gram/experiments28.py",
    "uv run python projects/imle-gram/experiments29.py")
src = src.replace("imle-cluster-gravity", "imle-cluster-gravity-pixmerge")

# 3. Replace MERGE_DIST with MERGE_DIST_PIX constant
src = src.replace(
    'GRAVITY_G  = 500.0  # calibrated: embedding norms ~1.3, min dist ~0.6, need ~100 epochs to merge\n'
    'MERGE_DIST = 0.5    # L2 merge threshold (~80% of observed min distance)',
    'GRAVITY_G      = 500.0  # gravitational constant (embedding space)\n'
    'MERGE_DIST_PIX = 2.0    # pixel-space L2 merge threshold (max possible ~28)'
)

# 4. Replace the merge-check block to use pixel-space distances
OLD_MERGE = '''\
    # ── Merge check ────────────────────────────────────────────────────────
    diff2  = Z_new[np.newaxis, :, :] - Z_new[:, np.newaxis, :]  # (K, K, D)
    r_mat  = np.linalg.norm(diff2, axis=-1)                       # (K, K)
    np.fill_diagonal(r_mat, np.inf)

    rows, cols = np.where(np.triu(r_mat < MERGE_DIST, k=1))'''

NEW_MERGE = '''\
    # ── Merge check (pixel space) ──────────────────────────────────────────
    # Decode post-gravity prototypes; compare in 784-dim pixel space so that
    # merges reflect visual redundancy, not just embedding proximity.
    tmp_params = {**params, "codebook": jnp.array(Z_new)}
    protos = decode_prototypes(tmp_params)          # (K, 784) float32 in [0,1]
    pdiff  = protos[np.newaxis, :, :] - protos[:, np.newaxis, :]  # (K, K, 784)
    r_mat  = np.linalg.norm(pdiff, axis=-1)                        # (K, K)
    np.fill_diagonal(r_mat, np.inf)

    rows, cols = np.where(np.triu(r_mat < MERGE_DIST_PIX, k=1))'''

src = src.replace(OLD_MERGE, NEW_MERGE)

# 5. Sort merged pairs by pixel-space distance (r_mat already pixel-space)
# (no change needed — argsort(r_mat[rows, cols]) still correct)

# 6. Update subtitle
src = src.replace(
    'save_learning_curves(history, EXP_NAME, subtitle=f"gravity G={GRAVITY_G} merge_dist={MERGE_DIST} N_CODES={N_CODES}")',
    'save_learning_curves(history, EXP_NAME, subtitle=f"gravity G={GRAVITY_G} pix_merge={MERGE_DIST_PIX} N_CODES={N_CODES}")'
)

dest = Path(__file__).parent.parent.parent / "experiments29.py"
dest.write_text(src)
print(f"wrote {dest}  ({len(src)} chars)")
