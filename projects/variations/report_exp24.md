# variations — exp24 report (CIFAR-10, LATENT=512 + intra-class kNN)

**Algorithm**: CNN encoder + wide deconvnet. LATENT=512, intra-class pixel-space kNN.

Hypothesis: kNN restricted to same CIFAR-10 class gives semantically consistent variation targets.
Precomputed knn_cifar_intraclass_k64.npy (within-class L2 kNN, global indices).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 512 |
| kNN type | **intra-class** (within CIFAR-10 class, pixel-space L2) |
| N_VAR_SAMPLES | 64 |
| K_NN | 16 |
| ALPHA | 1.0 |
| Coverage ratio | 4.0× |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | 1024 / 256 / 128 / 64 |
| n_params | 8,129,808 |
| time | 7440.5s |

## Results

**Final reconstruction MSE**: `0.002228`

## Learning curves

![Learning curves](plots/variations_exp24_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp24_recon.png)

## Variation grids (intra-class neighbors shown)

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp24_var_class0.png)](plots/variations_exp24_var_class0.png) | [![automobile](plots/variations_exp24_var_class1.png)](plots/variations_exp24_var_class1.png) | [![bird](plots/variations_exp24_var_class2.png)](plots/variations_exp24_var_class2.png) | [![cat](plots/variations_exp24_var_class3.png)](plots/variations_exp24_var_class3.png) | [![deer](plots/variations_exp24_var_class4.png)](plots/variations_exp24_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp24_var_class5.png)](plots/variations_exp24_var_class5.png) | [![frog](plots/variations_exp24_var_class6.png)](plots/variations_exp24_var_class6.png) | [![horse](plots/variations_exp24_var_class7.png)](plots/variations_exp24_var_class7.png) | [![ship](plots/variations_exp24_var_class8.png)](plots/variations_exp24_var_class8.png) | [![truck](plots/variations_exp24_var_class9.png)](plots/variations_exp24_var_class9.png) |
