# variations — exp18 report (CIFAR-10, ResNet encoder + 4-stage decoder)

**Algorithm**: ResNet encoder + 4-stage deconvnet (starts at 2×2) on CIFAR-10.

Changes from exp17:
- Encoder: plain CNN → ResNet (2 residual blocks per downsampling stage, no BN, GELU)
- Decoder: 3-stage (4×4 start) → 4-stage (2×2 start), adds DC_C0=512 stage

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 128 |
| Encoder | ResNet: DownBlock(64) + ResBlock → DownBlock(128) + ResBlock → DownBlock(256) + ResBlock → FC(512→128) |
| N_CAT | 4 |
| VOCAB_SIZE | 8 |
| COND_DIM | 16 |
| DC_STEM / DC_C0 / DC_C1 / DC_C2 / DC_C3 | 1024 / 512 / 256 / 128 / 64 |
| K_NN | 64 |
| BATCH_SIZE | 128 |
| N_VAR_SAMPLES | 16 |
| MAX_EPOCHS | 100 |
| n_params | 9,902,928 |
| time | 2066.8s |

## Results

**Final reconstruction MSE**: `0.004648` (vs exp17 wide decoder: TBD, exp16 CNN: 0.002545)

## Learning curves

![Learning curves](plots/variations_exp18_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp18_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp18_var_class0.png)](plots/variations_exp18_var_class0.png) | [![automobile](plots/variations_exp18_var_class1.png)](plots/variations_exp18_var_class1.png) | [![bird](plots/variations_exp18_var_class2.png)](plots/variations_exp18_var_class2.png) | [![cat](plots/variations_exp18_var_class3.png)](plots/variations_exp18_var_class3.png) | [![deer](plots/variations_exp18_var_class4.png)](plots/variations_exp18_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp18_var_class5.png)](plots/variations_exp18_var_class5.png) | [![frog](plots/variations_exp18_var_class6.png)](plots/variations_exp18_var_class6.png) | [![horse](plots/variations_exp18_var_class7.png)](plots/variations_exp18_var_class7.png) | [![ship](plots/variations_exp18_var_class8.png)](plots/variations_exp18_var_class8.png) | [![truck](plots/variations_exp18_var_class9.png)](plots/variations_exp18_var_class9.png) |
