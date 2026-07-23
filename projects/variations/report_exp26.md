# variations — exp26 report (CIFAR-10, DeltaNet Autoregressive Decoder)

**Algorithm**: CNN encoder + DeltaNet autoregressive decoder over 64 patch tokens.
Replaces the parallel deconvnet with a delta-rule linear RNN that models p(x|z,style)
as a product of conditionals. LATENT_DIM=512 (same as exp23).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 512 |
| Decoder | DeltaNet: DM=256, DK=64, N_LAYERS=2, T_SEQ=64 (4×4 patches) |
| Encoder | CNN: Conv(64/128/256, stride-2) + FC(512) → 512 |
| N_VAR_SAMPLES | 64 |
| K_NN | 16 |
| ALPHA | 1.0 |
| Coverage ratio | 4.0× |
| n_params | 4,430,946 |
| time | 33990.0s |

## Results

**Final reconstruction MSE**: `0.004676`

## Learning curves

![Learning curves](plots/variations_exp26_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp26_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp26_var_class0.png)](plots/variations_exp26_var_class0.png) | [![automobile](plots/variations_exp26_var_class1.png)](plots/variations_exp26_var_class1.png) | [![bird](plots/variations_exp26_var_class2.png)](plots/variations_exp26_var_class2.png) | [![cat](plots/variations_exp26_var_class3.png)](plots/variations_exp26_var_class3.png) | [![deer](plots/variations_exp26_var_class4.png)](plots/variations_exp26_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp26_var_class5.png)](plots/variations_exp26_var_class5.png) | [![frog](plots/variations_exp26_var_class6.png)](plots/variations_exp26_var_class6.png) | [![horse](plots/variations_exp26_var_class7.png)](plots/variations_exp26_var_class7.png) | [![ship](plots/variations_exp26_var_class8.png)](plots/variations_exp26_var_class8.png) | [![truck](plots/variations_exp26_var_class9.png)](plots/variations_exp26_var_class9.png) |
