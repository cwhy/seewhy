# variations — exp16 report (CIFAR-10, CNN encoder)

**Algorithm**: CNN encoder + deconvnet decoder on CIFAR-10, B=128, N_VAR=16.

Key change from exp15: MLP encoder (3072→1024→512→128) replaced by CNN encoder
(3 stride-2 conv layers → FC head). Same 3-stage deconvnet decoder.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 128 |
| Encoder | CNN: Conv(64,3×3,s2) → Conv(128,3×3,s2) → Conv(256,3×3,s2) → FC(512) → 128 |
| N_CAT | 4 |
| VOCAB_SIZE | 8 |
| COND_DIM | 16 |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | 512 / 128 / 64 / 32 |
| K_NN | 64 |
| BATCH_SIZE | 128 |
| N_VAR_SAMPLES | 16 |
| MAX_EPOCHS | 100 |
| n_params | 3,824,528 |
| time | 1595.7s |

## Results

**Final reconstruction MSE**: `0.004826` (vs exp15 MLP encoder: 0.00489)

## Learning curves

![Learning curves](plots/variations_exp16_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp16_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp16_var_class0.png)](plots/variations_exp16_var_class0.png) | [![automobile](plots/variations_exp16_var_class1.png)](plots/variations_exp16_var_class1.png) | [![bird](plots/variations_exp16_var_class2.png)](plots/variations_exp16_var_class2.png) | [![cat](plots/variations_exp16_var_class3.png)](plots/variations_exp16_var_class3.png) | [![deer](plots/variations_exp16_var_class4.png)](plots/variations_exp16_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp16_var_class5.png)](plots/variations_exp16_var_class5.png) | [![frog](plots/variations_exp16_var_class6.png)](plots/variations_exp16_var_class6.png) | [![horse](plots/variations_exp16_var_class7.png)](plots/variations_exp16_var_class7.png) | [![ship](plots/variations_exp16_var_class8.png)](plots/variations_exp16_var_class8.png) | [![truck](plots/variations_exp16_var_class9.png)](plots/variations_exp16_var_class9.png) |
