---
name: project_variations_exp39
description: exp39 — Global IMLE + exclusive Voronoi + reconstruction loss DONE: recall=0.02113, recon=0.00074
type: project
---

exp39 = exp38 + reconstruction loss (L_imle + ALPHA * L_recon, ALPHA=1.0).

**Final results (2026-04-23, 200 epochs, ~3.2h):**
- `recall_mse = 0.02113` (vs exp38: 0.02310 — marginal improvement; vs exp35: 0.00666 — 3.2× worse)
- `recon_mse = 0.00074` (vs exp38: 0.08471 — massive improvement; approaching exp35: 0.00115)
- Loss at epoch 0: 0.03376 (L_recon=0.02018, L_imle=0.01358) → epoch 199: 0.00232 (L_recon=0.00070, L_imle=0.00162)
- Steady learning throughout, no collapse

**Key finding:** Adding recon loss fixed the encoder (recon 0.085→0.00074) but did NOT fix recall.
Recall is still 3.2× worse than exp35. The gap is due to local vs global IMLE, not missing recon loss.

**Root cause of recall gap:** Global Voronoi assigns each variation to ~781 training images spread
across the dataset. Local IMLE (exp35) assigns each variation to cover only K=16 neighbours of each
input image. Global assignment doesn't concentrate coverage around each image's local neighbourhood.

**Recall curve:** Drops from 0.01980 at epoch 4 → plateaus at ~0.021 by epoch 20. No further improvement.

**Files:**
- `projects/variations/experiments39.py`
- `projects/variations/scripts/gen_report_exp39.py`
- `scripts/tmp/gen_html_report_exp39.py`

**Why:** The recall metric measures how well variations cover the K=16 nearest neighbours of each image.
Global IMLE covers the full training distribution, not local neighbourhoods.

**How to apply:** To match exp35 recall without precomputed kNN, need a way to bias the global matching
toward local neighbourhoods — e.g., per-batch local NN search with GEMM restricted to nearby images,
or a hybrid loss combining global coverage + local reconstruction.
