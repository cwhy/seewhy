# variations — exp2 report

**Algorithm**: MLP encoder + gram decoder with 8-bit binary conditioning (inverted IMLE: kNN→closest sample, B=64, N_VAR=16).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 64 |
| N_BITS | 8 |
| MLP_HIDDEN | 256 |
| DEC_RANK | 16 |
| K_NN | 64 |
| ALPHA | 1.0 |
| BATCH_SIZE | 256 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | 1,680,656 |
| time | 1923.7s |

## Results

**Final reconstruction MSE (b=0⁸ on test set)**: `0.003105`

## Learning curves

![Learning curves](plots/variations_exp2_curves.png)

## Reconstructions (b=0⁸)

Top row: original. Bottom row: decoded with all bits zero.

![Reconstruction grid](plots/variations_exp2_recon.png)

## Variation grids — all 256 bit patterns

Each image shows: top strip = training source image + its 10 nearest kNN neighbors;
bottom = 16×16 mosaic of all 256 possible 8-bit patterns decoded for that source.
Click any image to enlarge.

| digit 0 | digit 1 | digit 2 | digit 3 | digit 4 |
| :---: | :---: | :---: | :---: | :---: |
| [![digit 0](plots/variations_exp2_var_digit0.png)](plots/variations_exp2_var_digit0.png) | [![digit 1](plots/variations_exp2_var_digit1.png)](plots/variations_exp2_var_digit1.png) | [![digit 2](plots/variations_exp2_var_digit2.png)](plots/variations_exp2_var_digit2.png) | [![digit 3](plots/variations_exp2_var_digit3.png)](plots/variations_exp2_var_digit3.png) | [![digit 4](plots/variations_exp2_var_digit4.png)](plots/variations_exp2_var_digit4.png) |
| digit 5 | digit 6 | digit 7 | digit 8 | digit 9 |
| [![digit 5](plots/variations_exp2_var_digit5.png)](plots/variations_exp2_var_digit5.png) | [![digit 6](plots/variations_exp2_var_digit6.png)](plots/variations_exp2_var_digit6.png) | [![digit 7](plots/variations_exp2_var_digit7.png)](plots/variations_exp2_var_digit7.png) | [![digit 8](plots/variations_exp2_var_digit8.png)](plots/variations_exp2_var_digit8.png) | [![digit 9](plots/variations_exp2_var_digit9.png)](plots/variations_exp2_var_digit9.png) |
