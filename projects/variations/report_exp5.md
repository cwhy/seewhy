# variations — exp5 report

**Algorithm**: MLP encoder + gram decoder with 2×16 categorical embedding conditioning
(inverted IMLE: kNN→closest sample, B=256, N_VAR=2, no checkpointing needed).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 64 |
| N_CAT | 2 |
| VOCAB_SIZE | 16 |
| E_DIM | 4 |
| COND_DIM | 8 (= N_CAT × E_DIM) |
| MLP_HIDDEN | 256 |
| DEC_RANK | 16 |
| K_NN | 64 |
| ALPHA | 1.0 |
| BATCH_SIZE | 256 |
| N_VAR_SAMPLES | 2 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | 1,680,784 |
| time | 363.1s |

## Results

**Final reconstruction MSE (cat=(0,0) on test set)**: `0.005432`

## Learning curves

![Learning curves](plots/variations_exp5_curves.png)

## Reconstructions (cat=(0,0))

Top row: original. Bottom row: decoded with both categories set to index 0.

![Reconstruction grid](plots/variations_exp5_recon.png)

## Variation grids — all 256 (cat1 × cat2) combinations

Each image shows: top strip = training source image + its 10 nearest kNN neighbors;
bottom = 16×16 mosaic where **row = cat1 (0→15)** and **col = cat2 (0→15)**.
The structured layout reveals how each embedding dimension controls variation independently.
Click any image to enlarge.

| digit 0 | digit 1 | digit 2 | digit 3 | digit 4 |
| :---: | :---: | :---: | :---: | :---: |
| [![digit 0](plots/variations_exp5_var_digit0.png)](plots/variations_exp5_var_digit0.png) | [![digit 1](plots/variations_exp5_var_digit1.png)](plots/variations_exp5_var_digit1.png) | [![digit 2](plots/variations_exp5_var_digit2.png)](plots/variations_exp5_var_digit2.png) | [![digit 3](plots/variations_exp5_var_digit3.png)](plots/variations_exp5_var_digit3.png) | [![digit 4](plots/variations_exp5_var_digit4.png)](plots/variations_exp5_var_digit4.png) |
| digit 5 | digit 6 | digit 7 | digit 8 | digit 9 |
| [![digit 5](plots/variations_exp5_var_digit5.png)](plots/variations_exp5_var_digit5.png) | [![digit 6](plots/variations_exp5_var_digit6.png)](plots/variations_exp5_var_digit6.png) | [![digit 7](plots/variations_exp5_var_digit7.png)](plots/variations_exp5_var_digit7.png) | [![digit 8](plots/variations_exp5_var_digit8.png)](plots/variations_exp5_var_digit8.png) | [![digit 9](plots/variations_exp5_var_digit9.png)](plots/variations_exp5_var_digit9.png) |
