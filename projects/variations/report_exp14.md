# variations — exp14 report

**Algorithm**: deconvnet decoder replacing Gram matrix, B=128, N_VAR=16.

Key change from exp13: Gram-matrix decoder replaced by a transposed-convolution (deconvnet) decoder.
MLP stem projects to (B, DC_C1, 7, 7), then two stride-2 ConvTranspose layers upsample 7×7→14×14→28×28.
Benchmarked speedup vs exp13 (Gram, B=128, N=16): ~21 min vs ~31 min.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 64 |
| N_CAT | 4 |
| VOCAB_SIZE | 8 |
| E_DIM | 4 |
| COND_DIM | 16 (= N_CAT × E_DIM) |
| Total slots | 4096 (8⁴) |
| DC_STEM | 256 |
| DC_C1 | 64 |
| DC_C2 | 32 |
| Decoder | deconvnet (ConvTranspose ×2, 7→14→28) |
| K_NN | 64 |
| ALPHA | 1.0 |
| BATCH_SIZE | 128 |
| N_VAR_SAMPLES | 16 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | 1,294,864 |
| time | 1285.9s |

## Results

**Final reconstruction MSE (recon_emb on test set)**: `0.002309`

## Learning curves

Training losses, eval MSE, and per-rank recall MSE (blue=early epochs → red=final epoch).
The per-rank plot shows recall error by kNN rank; rank 1 is the nearest neighbor.

![Learning curves](plots/variations_exp14_curves.png)

## Reconstructions (recon_emb)

Top row: original. Bottom row: decoded with the dedicated `recon_emb` conditioning vector.

![Reconstruction grid](plots/variations_exp14_recon.png)

## Variation grids — 4×8 mosaic (2×2 quadrant layout)

Each image shows:
- **Row 0**: training source + 10 kNN neighbors
- **Row 1**: closest decoded match per kNN (using 256 EVAL_COND_CATS random slots)
- **Row 2**: 16×16 mosaic arranged as 2×2 block of 8×8 sub-grids.
  Each quadrant fixes (cat3, cat4): TL=(0,0), TR=(4,0), BL=(0,4), BR=(4,4).
  Within each sub-grid: **row=cat1** (0→7), **col=cat2** (0→7).
  Red lines separate quadrants.

| digit 0 | digit 1 | digit 2 | digit 3 | digit 4 |
| :---: | :---: | :---: | :---: | :---: |
| [![digit 0](plots/variations_exp14_var_digit0.png)](plots/variations_exp14_var_digit0.png) | [![digit 1](plots/variations_exp14_var_digit1.png)](plots/variations_exp14_var_digit1.png) | [![digit 2](plots/variations_exp14_var_digit2.png)](plots/variations_exp14_var_digit2.png) | [![digit 3](plots/variations_exp14_var_digit3.png)](plots/variations_exp14_var_digit3.png) | [![digit 4](plots/variations_exp14_var_digit4.png)](plots/variations_exp14_var_digit4.png) |
| digit 5 | digit 6 | digit 7 | digit 8 | digit 9 |
| [![digit 5](plots/variations_exp14_var_digit5.png)](plots/variations_exp14_var_digit5.png) | [![digit 6](plots/variations_exp14_var_digit6.png)](plots/variations_exp14_var_digit6.png) | [![digit 7](plots/variations_exp14_var_digit7.png)](plots/variations_exp14_var_digit7.png) | [![digit 8](plots/variations_exp14_var_digit8.png)](plots/variations_exp14_var_digit8.png) | [![digit 9](plots/variations_exp14_var_digit9.png)](plots/variations_exp14_var_digit9.png) |
