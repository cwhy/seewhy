# variations — exp8 report

**Algorithm**: MLP encoder + gram decoder with inverted IMLE, variation loss only (no reconstruction loss).

Key change from exp6:
- **exp6**: trains with L_recon + L_var. `recon_emb` gets gradients from both paths.
- **exp8**: trains with L_var only. `recon_emb` still in params and gets gradients through the
  variation path (cond = recon_emb + cat_emb[cats]), but no explicit reconstruction supervision.
  Reconstruction is evaluated using recon_emb + cat_emb[zeros] to show quality without recon loss.

Architecture unchanged: dedicated `recon_emb`, N_CAT=4, VOCAB_SIZE=8, COND_DIM=16, 4096 total slots.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 64 |
| N_CAT | 4 |
| VOCAB_SIZE | 8 |
| E_DIM | 4 |
| COND_DIM | 16 (= N_CAT × E_DIM) |
| Total slots | 4096 (8⁴) |
| MLP_HIDDEN | 256 |
| DEC_RANK | 16 |
| K_NN | 64 (eval only) |
| ALPHA | 1.0 |
| BATCH_SIZE | 256 |
| N_VAR_SAMPLES | 2 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | 1,682,848 |
| time | 267.5s |

## Results

**Final reconstruction MSE (recon_emb on test set)**: `0.016492`

## Learning curves

Training losses, eval MSE, and per-rank recall MSE (x=iter, y=MSE, legend=key ranks).
Rank 1 = nearest neighbor, rank 64 = furthest; lower MSE = better recall at that rank.

![Learning curves](plots/variations_exp8_curves.png)

## Reconstructions (recon_emb)

Top row: original. Bottom row: decoded with the dedicated `recon_emb` conditioning vector.

![Reconstruction grid](plots/variations_exp8_recon.png)

## Variation grids — 4×8 mosaic (2×2 quadrant layout)

Each image shows:
- **Row 0**: training source + 10 kNN neighbors
- **Row 1**: closest decoded match per kNN (256 EVAL_COND_CATS random slots)
- **Row 2**: 16×16 mosaic as 2×2 block of 8×8 sub-grids.
  Quadrants fix (cat3, cat4): TL=(0,0), TR=(4,0), BL=(0,4), BR=(4,4).
  Within each sub-grid: **row=cat1** (0→7), **col=cat2** (0→7).

| digit 0 | digit 1 | digit 2 | digit 3 | digit 4 |
| :---: | :---: | :---: | :---: | :---: |
| [![digit 0](plots/variations_exp8_var_digit0.png)](plots/variations_exp8_var_digit0.png) | [![digit 1](plots/variations_exp8_var_digit1.png)](plots/variations_exp8_var_digit1.png) | [![digit 2](plots/variations_exp8_var_digit2.png)](plots/variations_exp8_var_digit2.png) | [![digit 3](plots/variations_exp8_var_digit3.png)](plots/variations_exp8_var_digit3.png) | [![digit 4](plots/variations_exp8_var_digit4.png)](plots/variations_exp8_var_digit4.png) |
| digit 5 | digit 6 | digit 7 | digit 8 | digit 9 |
| [![digit 5](plots/variations_exp8_var_digit5.png)](plots/variations_exp8_var_digit5.png) | [![digit 6](plots/variations_exp8_var_digit6.png)](plots/variations_exp8_var_digit6.png) | [![digit 7](plots/variations_exp8_var_digit7.png)](plots/variations_exp8_var_digit7.png) | [![digit 8](plots/variations_exp8_var_digit8.png)](plots/variations_exp8_var_digit8.png) | [![digit 9](plots/variations_exp8_var_digit9.png)](plots/variations_exp8_var_digit9.png) |
