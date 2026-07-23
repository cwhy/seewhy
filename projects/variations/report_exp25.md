# variations — exp25 report (CIFAR-10, LATENT=512 + ALPHA=3.0)

**Algorithm**: CNN encoder + wide deconvnet. LATENT=512, stronger variation loss ALPHA=3.0.

Hypothesis: with 512-d latent, increasing variation loss weight from 1.0 to 3.0 forces
broader coverage of output modes without losing reconstruction quality.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 512 |
| ALPHA | **3.0** (was 1.0 in exp23) |
| N_VAR_SAMPLES | 64 |
| K_NN | 16 |
| Coverage ratio | 4.0× |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | 1024 / 256 / 128 / 64 |
| kNN type | pixel-space L2 (sliced from k64) |
| n_params | 8,129,808 |
| time | 7377.8s |

## Results

**Final reconstruction MSE**: `0.002793`

## Learning curves

![Learning curves](plots/variations_exp25_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp25_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp25_var_class0.png)](plots/variations_exp25_var_class0.png) | [![automobile](plots/variations_exp25_var_class1.png)](plots/variations_exp25_var_class1.png) | [![bird](plots/variations_exp25_var_class2.png)](plots/variations_exp25_var_class2.png) | [![cat](plots/variations_exp25_var_class3.png)](plots/variations_exp25_var_class3.png) | [![deer](plots/variations_exp25_var_class4.png)](plots/variations_exp25_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp25_var_class5.png)](plots/variations_exp25_var_class5.png) | [![frog](plots/variations_exp25_var_class6.png)](plots/variations_exp25_var_class6.png) | [![horse](plots/variations_exp25_var_class7.png)](plots/variations_exp25_var_class7.png) | [![ship](plots/variations_exp25_var_class8.png)](plots/variations_exp25_var_class8.png) | [![truck](plots/variations_exp25_var_class9.png)](plots/variations_exp25_var_class9.png) |
