"""
t-SNE snapshot capture for animation.

Runs openTSNE in two phases (early exaggeration + refinement) and returns
a list of (N, 2) arrays — one per captured frame.
"""

import numpy as np


def capture_snapshots(
    E,
    perplexity     = 40,
    n_early        = 10,    # snapshots during early-exaggeration phase
    early_iters    = 25,    # optimisation iters per early snapshot
    n_refine       = 15,    # snapshots during refinement phase
    refine_iters   = 50,    # optimisation iters per refine snapshot
    seed           = 42,
    log_fn         = None,  # optional callable(str) for progress messages
):
    """Run openTSNE and return a list of intermediate (N, 2) position arrays.

    The first element is the PCA initialisation (before any optimisation).
    Then n_early frames from the early-exaggeration phase (exaggeration=12),
    then n_refine frames from the refinement phase (exaggeration=1).

    Total frames = 1 + n_early + n_refine.

    E       : (N, D) float32 embeddings
    log_fn  : called with a progress string after each snapshot; pass
              logging.info for experiment scripts.
    """
    import openTSNE

    def log(msg):
        if log_fn is not None:
            log_fn(msg)

    tsne_init = openTSNE.TSNE(
        n_components  = 2,
        perplexity    = perplexity,
        initialization = "pca",
        random_state  = seed,
        verbose       = False,
    )
    emb = tsne_init.prepare_initial(E.astype(np.float32))
    snapshots = [np.array(emb)]

    for i in range(n_early):
        emb = emb.optimize(n_iter=early_iters, exaggeration=12, momentum=0.5)
        snapshots.append(np.array(emb))
        log(f"  early-exag {i+1}/{n_early}")

    for i in range(n_refine):
        emb = emb.optimize(n_iter=refine_iters, exaggeration=1, momentum=0.8)
        snapshots.append(np.array(emb))
        log(f"  refine {i+1}/{n_refine}")

    return snapshots
