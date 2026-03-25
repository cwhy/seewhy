"""Create experiments28.py — gravity-driven codebook with merge+reinit."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments21.py").read_text()

GRAVITY_SECTION = '''
# ── Gravity + merge-reinit codebook maintenance ───────────────────────────────
# After each epoch's gradient step:
#   1. Apply Newtonian gravity between all codebook pairs.
#      Force on i from j: G * mass_i * mass_j * (z_j - z_i) / r^3
#      Mass = fraction of epoch assignments (data-dense = heavy attractor).
#   2. After gravity step, find pairs within L2 distance MERGE_DIST.
#   3. Greedy merge: keep the heavier cluster, reinit the lighter one from scratch.
#   4. Zero Adam mu/nu for reinited rows.

GRAVITY_G  = 0.1    # gravitational constant (tune: larger → stronger pull)
MERGE_DIST = 0.15   # L2 distance threshold to trigger a merge

# Global slot so _apply_gravity can access the optimizer pytree structure for resets
optimizer_ref = [None]

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

    # ── Merge check ────────────────────────────────────────────────────────
    diff2  = Z_new[np.newaxis, :, :] - Z_new[:, np.newaxis, :]  # (K, K, D)
    r_mat  = np.linalg.norm(diff2, axis=-1)                       # (K, K)
    np.fill_diagonal(r_mat, np.inf)

    rows, cols = np.where(np.triu(r_mat < MERGE_DIST, k=1))
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

        # Zero Adam mu/nu for reinited rows
        reinit_arr = jnp.array(to_reinit)
        def _zero_rows(leaf, ref):
            if hasattr(ref, "shape") and leaf.shape == ref.shape:
                return leaf.at[reinit_arr].set(0.0)
            return leaf
        opt_state = jax.tree_util.tree_map(_zero_rows, opt_state, optimizer_ref[0])

    new_params = {**params, "codebook": jnp.array(Z_new)}
    return new_params, opt_state, grav_disp, n_merged

'''

out = src

# 1. Docstring
out = out.replace(
    "exp21 — 128-cluster with delta_sum assignment.",
    "exp28 — 128-cluster flat delta_sum + gravity-driven codebook maintenance.\n\n"
    "Same as exp21 but after each epoch applies Newtonian gravity between all\n"
    "codebook pairs. Mass = fraction of epoch assignments (data-dense clusters\n"
    "attract more strongly). Pairs that end up within MERGE_DIST are merged\n"
    "and the lighter cluster is reinitialised from scratch.\n\n"
    "Key parameters:\n"
    "  GRAVITY_G  = 0.1   (gravitational constant)\n"
    "  MERGE_DIST = 0.15  (L2 merge threshold in embedding space)")

out = out.replace('EXP_NAME = "exp21"', 'EXP_NAME = "exp28"')
out = out.replace(
    "uv run python projects/imle-gram/experiments21.py",
    "uv run python projects/imle-gram/experiments28.py")
out = out.replace("imle-cluster-deltasum", "imle-cluster-gravity")

# 2. Insert gravity section before delta-sum assignment
out = out.replace(
    "# ── Delta-sum assignment ─────────────────────────────────────────────────────",
    GRAVITY_SECTION +
    "# ── Delta-sum assignment ─────────────────────────────────────────────────────")

# 3. In train(): register optimizer_ref
out = out.replace(
    "    opt_state   = optimizer.init(params)\n"
    "    step_fn     = make_train_step(optimizer)",
    "    opt_state   = optimizer.init(params)\n"
    "    optimizer_ref[0] = params   # structural reference for tree_map resets\n"
    "    step_fn     = make_train_step(optimizer)")

# 4. After the epoch gradient loop, apply gravity+merge then log
out = out.replace(
    "        e_loss /= num_batches; e_match /= num_batches\n"
    "        history[\"match_loss\"].append(e_match)",
    "        e_loss /= num_batches; e_match /= num_batches\n"
    "        history[\"match_loss\"].append(e_match)\n"
    "\n"
    "        params, opt_state, grav_disp, n_merged = apply_gravity_and_merge(\n"
    "            params, opt_state, usage, epoch)")

# 5. Add gravity stats to the epoch log line
out = out.replace(
    '        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"\n'
    '                     f"  active={n_active}/{N_CODES}{eval_str}  {time.perf_counter()-t0:.1f}s")',
    '        grav_str = f"  grav={grav_disp:.5f}  merged={n_merged}"\n'
    '        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"\n'
    '                     f"  active={n_active}/{N_CODES}{eval_str}{grav_str}  {time.perf_counter()-t0:.1f}s")')

# 6. Update subtitle
out = out.replace(
    'save_learning_curves(history, EXP_NAME, subtitle=f"flat clustering N_CODES={N_CODES} D_U={D_U}")',
    'save_learning_curves(history, EXP_NAME, subtitle=f"gravity G={GRAVITY_G} merge_dist={MERGE_DIST} N_CODES={N_CODES}")')

dest = Path(__file__).parent.parent.parent / "experiments28.py"
dest.write_text(out)
print(f"wrote {dest}  ({len(out)} chars)")
