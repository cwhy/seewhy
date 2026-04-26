# variations — exp34 report (CIFAR-10, LATENT=2048, N_VAR=256)

**Algorithm**: CNN encoder + wide deconvnet decoder. Primary change: N_VAR=256 (was 128 in exp33), B=32 (was 64).

Goal: match training coverage to eval coverage. Eval uses 256 fixed conditions with K_NN=16 → 16× coverage ratio.
exp33 trained at 8× coverage; exp34 trains at 16× to close this gap.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 2048 |
| Encoder | CNN: Conv(64/128/256, stride-2) + FC(2048) → 2048 |
| N_VAR_SAMPLES | 256 (was 128 in exp33) |
| BATCH_SIZE | 32 (was 64 in exp33, keep B×N_VAR=8,192) |
| K_NN | 16 |
| ALPHA | 1.0 |
| Coverage ratio | 16.0× (was 8× in exp33) |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | 1024 / 256 / 128 / 64 |
| kNN type | pixel-space L2 (sliced from k64) |
| n_params | 19,929,360 |
| time | 59559.1s |

## Results

**Final reconstruction MSE**: `0.001151`

## Learning curves

![Learning curves](plots/variations_exp34_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp34_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp34_var_class0.png)](plots/variations_exp34_var_class0.png) | [![automobile](plots/variations_exp34_var_class1.png)](plots/variations_exp34_var_class1.png) | [![bird](plots/variations_exp34_var_class2.png)](plots/variations_exp34_var_class2.png) | [![cat](plots/variations_exp34_var_class3.png)](plots/variations_exp34_var_class3.png) | [![deer](plots/variations_exp34_var_class4.png)](plots/variations_exp34_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp34_var_class5.png)](plots/variations_exp34_var_class5.png) | [![frog](plots/variations_exp34_var_class6.png)](plots/variations_exp34_var_class6.png) | [![horse](plots/variations_exp34_var_class7.png)](plots/variations_exp34_var_class7.png) | [![ship](plots/variations_exp34_var_class8.png)](plots/variations_exp34_var_class8.png) | [![truck](plots/variations_exp34_var_class9.png)](plots/variations_exp34_var_class9.png) |
