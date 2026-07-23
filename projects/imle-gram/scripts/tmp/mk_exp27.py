"""Create experiments27.py — merge+reinit codebook maintenance."""
from pathlib import Path

src = (Path(__file__).parent.parent.parent / "experiments21.py").read_text()

MERGE_REINIT_SECTION = '''
# ── Merge-and-reinit codebook maintenance ─────────────────────────────────────
# After each epoch: find near-duplicate codebook pairs (cosine_sim > threshold),
# keep the one with more cumulative assignments, reinitialise the other by
# splitting the most-assigned cluster (embedding + small noise).
# Adam mu/nu for reinited rows are zeroed so stale momentum doesn't interfere.

MERGE_THRESHOLD = 0.9    # cosine similarity above which two clusters merge
REINIT_NOISE    = 0.05   # std of Gaussian noise added to the donor cluster

def merge_and_reinit(params, opt_state, cum_usage, key):
    """
    Identify near-duplicate codebook entries, keep the busier one,
    reinitialise the other near the most-assigned cluster.

    Returns (new_params, new_opt_state, n_merged).
    """
    Z   = np.array(params["codebook"])          # (K, D_EMB)
    nrm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Z_n = Z / nrm                               # unit-normed rows
    C   = Z_n @ Z_n.T                           # (K, K) cosine similarities

    # Collect all above-threshold pairs from upper triangle, sorted by similarity
    rows, cols = np.where(np.triu(C > MERGE_THRESHOLD, k=1))
    if len(rows) == 0:
        return params, opt_state, 0

    order   = np.argsort(-C[rows, cols])        # highest similarity first
    rows    = rows[order]; cols = cols[order]

    merged  = set()
    to_reinit = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i in merged or j in merged:
            continue
        # Keep the busier cluster, reinit the quieter one
        keep   = i if cum_usage[i] >= cum_usage[j] else j
        reinit = j if keep == i else i
        merged.add(keep); merged.add(reinit)
        to_reinit.append(reinit)

    if not to_reinit:
        return params, opt_state, 0

    # Reinit: split the globally most-assigned cluster
    donor     = int(np.argmax(cum_usage))
    noise_key = jax.random.PRNGKey(int(key[0]) ^ int(key[1]))
    new_cb    = np.array(params["codebook"])
    for idx in to_reinit:
        noise         = np.array(jax.random.normal(noise_key, (D_EMB,))) * REINIT_NOISE
        new_cb[idx]   = new_cb[donor] + noise
        noise_key     = jax.random.fold_in(noise_key, idx)

    new_params = {**params, "codebook": jnp.array(new_cb)}

    # Zero Adam mu/nu for reinited rows so stale momentum doesn't interfere.
    # optax state is a tuple; the first element is ScaleByAdamState with .mu/.nu
    # pytrees mirroring params. We use tree_map with a mask on the codebook leaf.
    reinit_arr = jnp.array(to_reinit)
    def _zero_rows(leaf, param_leaf):
        if hasattr(param_leaf, "shape") and leaf.shape == param_leaf.shape:
            return leaf.at[reinit_arr].set(0.0)
        return leaf
    new_opt = jax.tree_util.tree_map(_zero_rows, opt_state, optimizer_ref[0])
    return new_params, new_opt, len(to_reinit)

# Global slot so merge_and_reinit can access the optimizer state structure
optimizer_ref = [None]
'''

MERGE_LOG = '''
        # ── Merge near-duplicate clusters ─────────────────────────────────
        params, opt_state, n_merged = merge_and_reinit(
            params, opt_state, cum_usage_total, jax.random.PRNGKey(epoch))
        if n_merged:
            logging.info(f"  merged {n_merged} near-duplicate cluster(s)")
        # ──────────────────────────────────────────────────────────────────
'''

USAGE_TOTAL_INIT = "    cum_usage_total = np.zeros(N_CODES)\n"
USAGE_TOTAL_UPDATE = "        cum_usage_total += usage\n"

out = src

# 1. Update docstring
out = out.replace(
    "exp21 — 128-cluster with delta_sum assignment.",
    "exp27 — 128-cluster flat delta_sum + merge-and-reinit codebook maintenance.\n\n"
    "Same as exp21 but adds a periodic merge step after each epoch:\n"
    "  - Find all codebook pairs with cosine_sim > MERGE_THRESHOLD (0.9)\n"
    "  - Greedy match: keep the busier cluster, reinit the other\n"
    "  - Reinit = most-assigned cluster embedding + small Gaussian noise\n"
    "  - Adam mu/nu zeroed for reinited rows to avoid stale momentum\n"
    "  - delta_sum assigns naturally to fresh clusters (delta_sum=0 wins)\n"
    "\n"
    "Multiple near-duplicates are handled in a single pass per epoch.")

# 2. EXP_NAME and usage script
out = out.replace('EXP_NAME = "exp21"', 'EXP_NAME = "exp27"')
out = out.replace(
    "uv run python projects/imle-gram/experiments21.py",
    "uv run python projects/imle-gram/experiments27.py")

# 3. experiment name in results
out = out.replace("imle-cluster-deltasum", "imle-cluster-merge-reinit")

# 4. Insert merge+reinit section before load_balanced_assign
out = out.replace(
    "# ── Delta-sum assignment ─────────────────────────────────────────────────────",
    MERGE_REINIT_SECTION +
    "# ── Delta-sum assignment ─────────────────────────────────────────────────────")

# 5. In train(): initialise optimizer_ref and cum_usage_total
out = out.replace(
    "    opt_state   = optimizer.init(params)\n"
    "    step_fn     = make_train_step(optimizer)",
    "    opt_state   = optimizer.init(params)\n"
    "    optimizer_ref[0] = params          # structural reference for tree_map\n"
    "    step_fn     = make_train_step(optimizer)\n"
    + USAGE_TOTAL_INIT)

# 6. After computing usage each epoch, accumulate total and run merge
out = out.replace(
    "        history[\"cluster_usage\"].append(usage.tolist())",
    "        history[\"cluster_usage\"].append(usage.tolist())\n"
    + USAGE_TOTAL_UPDATE
    + MERGE_LOG)

# 7. Log n_merged in the epoch line (already handled by the if n_merged block above)
# Update subtitle
out = out.replace(
    'save_learning_curves(history, EXP_NAME, subtitle=f"flat clustering N_CODES={N_CODES} D_U={D_U}")',
    'save_learning_curves(history, EXP_NAME, subtitle=f"merge-reinit N_CODES={N_CODES} D_U={D_U} thr={MERGE_THRESHOLD}")')

dest = Path(__file__).parent.parent.parent / "experiments27.py"
dest.write_text(out)
print(f"wrote {dest}  ({len(out)} chars)")
