"""Create experiments32.py — pixel-space merge only, no gravity."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments31.py").read_text()

# 1. Docstring
src = src.replace(
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
    '  MERGE_DIST_PIX = 100.0  (pixel-space L2 merge threshold)',
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
    '  MERGE_DIST_PIX = 100.0 (pixel-space L2 merge threshold; min=56, p1=488)'
)

# 2. EXP_NAME, run command, wandb name
src = src.replace('EXP_NAME = "exp31"', 'EXP_NAME = "exp32"')
src = src.replace(
    "uv run python projects/imle-gram/experiments31.py",
    "uv run python projects/imle-gram/experiments32.py")
src = src.replace("imle-cluster-gravity-pixmerge-warm", "imle-cluster-pixmerge")

# 3. Remove GRAVITY_G constant and MERGE_WARMUP stays
src = src.replace(
    'MERGE_WARMUP   = 20     # epochs before gravity+merge activates\n'
    'MERGE_DIST_PIX = 100.0  # pixel-space L2 merge threshold (tighter: min=56, p1=488)',
    'MERGE_WARMUP   = 20     # epochs before merge check activates\n'
    'MERGE_DIST_PIX = 100.0  # pixel-space L2 merge threshold (min=56, p1=488)'
)

# 4. Replace apply_gravity_and_merge with merge_reinit_only (no gravity step)
OLD_FN = '''\
def apply_gravity_and_merge(params, opt_state, usage, epoch):
    """
    Apply gravity between all codebook pairs, then merge+reinit close ones.

    Returns (new_params, new_opt_state, grav_disp, n_merged).
    """
    Z    = np.array(params["codebook"])          # (K, D_EMB)
    K    = Z.shape[0]
    mass = usage.astype(np.float32) / (usage.sum() + 1e-8)   # (K,) normalised

    # ── Gravity step ───────────────────────────────────────────────────────
    diff = Z[np.newaxis, :, :] - Z[:, np.newaxis, :]  # (K, K, D): diff[i,j] = z_j - z_i
    r    = np.linalg.norm(diff, axis=-1, keepdims=True)        # (K, K, 1)
    r3   = (r + 1e-8) ** 3

    mass_prod = (mass[:, np.newaxis] * mass[np.newaxis, :])    # (K, K)
    F         = GRAVITY_G * mass_prod[:, :, np.newaxis] * diff / r3   # (K, K, D)
    np.einsum("iid->id", F)[:] = 0.0              # zero self-force (diagonal)

    delta  = F.sum(axis=1)                        # (K, D) net displacement
    grav_disp = float(np.abs(delta).mean())
    Z_new  = Z + delta

    # ── Merge check (pixel space) ──────────────────────────────────────────
    # Decode post-gravity prototypes; compare in 784-dim pixel space so that
    # merges reflect visual redundancy, not just embedding proximity.
    tmp_params = {**params, "codebook": jnp.array(Z_new)}
    protos = decode_prototypes(tmp_params)          # (K, 784) float32 in [0,1]
    pdiff  = protos[np.newaxis, :, :] - protos[:, np.newaxis, :]  # (K, K, 784)
    r_mat  = np.linalg.norm(pdiff, axis=-1)                        # (K, K)
    np.fill_diagonal(r_mat, np.inf)

    rows, cols = np.where(np.triu(r_mat < MERGE_DIST_PIX, k=1))
    n_merged   = 0

    if len(rows):
        order  = np.argsort(r_mat[rows, cols])
        rows   = rows[order]; cols = cols[order]
        merged = set()
        to_reinit = []

        for i, j in zip(rows.tolist(), cols.tolist()):
            if i in merged or j in merged:
                continue
            keep   = i if usage[i] >= usage[j] else j
            reinit = j if keep == i else i
            merged.add(keep); merged.add(reinit)
            to_reinit.append(reinit)

        # Reinit from scratch (same scale as init_params)
        rng = np.random.default_rng(epoch * 1000)
        for idx in to_reinit:
            Z_new[idx] = rng.standard_normal(D_EMB).astype(np.float32) * (D_EMB ** -0.5)

        n_merged = len(to_reinit)

        # Zero Adam mu/nu for reinited rows.
        # optax Adam state[0] is ScaleByAdamState with .mu and .nu pytrees
        # mirroring params. We directly zero the codebook rows in both.
        try:
            reinit_arr = jnp.array(to_reinit)
            adam = opt_state[0]
            new_mu = {**adam.mu, "codebook": adam.mu["codebook"].at[reinit_arr].set(0.0)}
            new_nu = {**adam.nu, "codebook": adam.nu["codebook"].at[reinit_arr].set(0.0)}
            opt_state = (adam._replace(mu=new_mu, nu=new_nu),) + tuple(opt_state[1:])
        except Exception:
            pass  # if state structure differs, leave as-is

    new_params = {**params, "codebook": jnp.array(Z_new)}
    return new_params, opt_state, grav_disp, n_merged'''

NEW_FN = '''\
def merge_reinit(params, opt_state, usage, epoch):
    """
    Check for near-duplicate prototypes in pixel space and reinit the lighter one.

    No gravity — gradient learning is left undisturbed. We only intervene when
    two decoded prototypes are genuinely visually similar (pixel-L2 < MERGE_DIST_PIX).

    Returns (new_params, new_opt_state, n_merged).
    """
    protos = decode_prototypes(params)              # (K, 784)
    pdiff  = protos[np.newaxis, :, :] - protos[:, np.newaxis, :]  # (K, K, 784)
    r_mat  = np.linalg.norm(pdiff, axis=-1)         # (K, K)
    np.fill_diagonal(r_mat, np.inf)

    rows, cols = np.where(np.triu(r_mat < MERGE_DIST_PIX, k=1))
    n_merged   = 0

    if not len(rows):
        return params, opt_state, 0

    order  = np.argsort(r_mat[rows, cols])
    rows   = rows[order]; cols = cols[order]
    merged = set()
    to_reinit = []

    for i, j in zip(rows.tolist(), cols.tolist()):
        if i in merged or j in merged:
            continue
        keep   = i if usage[i] >= usage[j] else j
        reinit = j if keep == i else i
        merged.add(keep); merged.add(reinit)
        to_reinit.append(reinit)

    Z_new = np.array(params["codebook"])
    rng   = np.random.default_rng(epoch * 1000)
    for idx in to_reinit:
        Z_new[idx] = rng.standard_normal(D_EMB).astype(np.float32) * (D_EMB ** -0.5)

    n_merged = len(to_reinit)

    try:
        reinit_arr = jnp.array(to_reinit)
        adam = opt_state[0]
        new_mu = {**adam.mu, "codebook": adam.mu["codebook"].at[reinit_arr].set(0.0)}
        new_nu = {**adam.nu, "codebook": adam.nu["codebook"].at[reinit_arr].set(0.0)}
        opt_state = (adam._replace(mu=new_mu, nu=new_nu),) + tuple(opt_state[1:])
    except Exception:
        pass

    new_params = {**params, "codebook": jnp.array(Z_new)}
    return new_params, opt_state, n_merged'''

src = src.replace(OLD_FN, NEW_FN)

# 5. Update call site
src = src.replace(
    '        if epoch >= MERGE_WARMUP:\n'
    '            params, opt_state, grav_disp, n_merged = apply_gravity_and_merge(\n'
    '                params, opt_state, usage, epoch)\n'
    '        else:\n'
    '            grav_disp, n_merged = 0.0, 0',
    '        if epoch >= MERGE_WARMUP:\n'
    '            params, opt_state, n_merged = merge_reinit(params, opt_state, usage, epoch)\n'
    '        else:\n'
    '            n_merged = 0'
)

# 6. Update log line — remove grav_str
src = src.replace(
    '        grav_str = f"  grav={grav_disp:.5f}  merged={n_merged}"\n'
    '        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"\n'
    '                     f"  active={n_active}/{N_CODES}{eval_str}{grav_str}  {time.perf_counter()-t0:.1f}s")',
    '        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"\n'
    '                     f"  active={n_active}/{N_CODES}{eval_str}  merged={n_merged}  {time.perf_counter()-t0:.1f}s")'
)

# 7. Update subtitle
src = src.replace(
    'save_learning_curves(history, EXP_NAME, subtitle=f"gravity G={GRAVITY_G} warmup={MERGE_WARMUP} pix_merge={MERGE_DIST_PIX} N_CODES={N_CODES}")',
    'save_learning_curves(history, EXP_NAME, subtitle=f"pix_merge warmup={MERGE_WARMUP} dist={MERGE_DIST_PIX} N_CODES={N_CODES}")'
)

# 8. Remove GRAVITY_G from docstring constants that were left in the code
src = src.replace(
    'GRAVITY_G      = 500.0  # gravitational constant (embedding space)\n'
    'MERGE_WARMUP   = 20     # epochs before merge check activates\n'
    'MERGE_DIST_PIX = 100.0  # pixel-space L2 merge threshold (min=56, p1=488)',
    'MERGE_WARMUP   = 20     # epochs before merge check activates\n'
    'MERGE_DIST_PIX = 100.0  # pixel-space L2 merge threshold (min=56, p1=488)'
)

dest = Path(__file__).parent.parent.parent / "experiments32.py"
dest.write_text(src)
print(f"wrote {dest}  ({len(src)} chars)")
