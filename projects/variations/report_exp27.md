# variations — exp27 report (CIFAR-10, LATENT=1024)

**Algorithm**: CNN encoder + wide deconvnet decoder. Primary change: LATENT_DIM=1024 (was 512 in exp23), ENC_FC1=1024.

Tests the encoder bottleneck hypothesis: each LATENT doubling (64→128→512) halved MSE.
LATENT=1024 (3× compression) may push below 0.0020.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 1024 (was 512 in exp23) |
| Encoder | CNN: Conv(64/128/256, stride-2) + FC(1024) → 1024 |
| N_VAR_SAMPLES | 64 |
| K_NN | 16 |
| ALPHA | 1.0 |
| Coverage ratio | 4.0× |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | 1024 / 256 / 128 / 64 |
| kNN type | pixel-space L2 (sliced from k64) |
| n_params | 11,538,704 |
| time | 7395.6s |

## Results

**Final reconstruction MSE**: `0.001725`

## Learning curves

![Learning curves](plots/variations_exp27_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp27_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp27_var_class0.png)](plots/variations_exp27_var_class0.png) | [![automobile](plots/variations_exp27_var_class1.png)](plots/variations_exp27_var_class1.png) | [![bird](plots/variations_exp27_var_class2.png)](plots/variations_exp27_var_class2.png) | [![cat](plots/variations_exp27_var_class3.png)](plots/variations_exp27_var_class3.png) | [![deer](plots/variations_exp27_var_class4.png)](plots/variations_exp27_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp27_var_class5.png)](plots/variations_exp27_var_class5.png) | [![frog](plots/variations_exp27_var_class6.png)](plots/variations_exp27_var_class6.png) | [![horse](plots/variations_exp27_var_class7.png)](plots/variations_exp27_var_class7.png) | [![ship](plots/variations_exp27_var_class8.png)](plots/variations_exp27_var_class8.png) | [![truck](plots/variations_exp27_var_class9.png)](plots/variations_exp27_var_class9.png) |
