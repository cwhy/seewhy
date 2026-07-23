# variations — exp15 report (CIFAR-10)

**Algorithm**: deconvnet decoder on CIFAR-10, B=128, N_VAR=16.

Adapts exp14 (MNIST) to CIFAR-10: 32×32 RGB images, deeper encoder (3072→1024→512→128),
wider decoder (stem=512, C1=128, C2=64, C3=32, 3 ConvTranspose stages: 4×4→8×8→16×16→32×32).

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 128 |
| N_CAT | 4 |
| VOCAB_SIZE | 8 |
| E_DIM | 4 |
| COND_DIM | 16 (= N_CAT × E_DIM) |
| Total slots | 4096 (8⁴) |
| DC_STEM | 512 |
| DC_C1 | 128 |
| DC_C2 | 64 |
| DC_C3 | 32 |
| Decoder | deconvnet (ConvTranspose ×3, 4→8→16→32) |
| K_NN | 64 |
| ALPHA | 1.0 |
| BATCH_SIZE | 128 |
| N_VAR_SAMPLES | 16 |
| MAX_EPOCHS | 100 |
| ADAM_LR | 3e-4 |
| n_params | 5,027,600 |
| time | 1576.3s |

## Results

**Final reconstruction MSE (recon_emb on test set)**: `0.004887`

## Learning curves

Training losses, eval MSE, and per-rank recall MSE (blue=early epochs → red=final epoch).

![Learning curves](plots/variations_exp15_curves.png)

## Reconstructions (recon_emb)

Top row: original. Bottom row: decoded with the dedicated `recon_emb` conditioning vector.

![Reconstruction grid](plots/variations_exp15_recon.png)

## Variation grids — 4×8 mosaic (2×2 quadrant layout)

Each image shows:
- **Row 0**: training source + 10 kNN neighbors
- **Row 1**: closest decoded match per kNN (using 256 EVAL_COND_CATS random slots)
- **Row 2**: 16×16 mosaic arranged as 2×2 block of 8×8 sub-grids.
  Each quadrant fixes (cat3, cat4): TL=(0,0), TR=(4,0), BL=(0,4), BR=(4,4).
  Within each sub-grid: **row=cat1** (0→7), **col=cat2** (0→7).
  Red lines separate quadrants.

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp15_var_class0.png)](plots/variations_exp15_var_class0.png) | [![automobile](plots/variations_exp15_var_class1.png)](plots/variations_exp15_var_class1.png) | [![bird](plots/variations_exp15_var_class2.png)](plots/variations_exp15_var_class2.png) | [![cat](plots/variations_exp15_var_class3.png)](plots/variations_exp15_var_class3.png) | [![deer](plots/variations_exp15_var_class4.png)](plots/variations_exp15_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp15_var_class5.png)](plots/variations_exp15_var_class5.png) | [![frog](plots/variations_exp15_var_class6.png)](plots/variations_exp15_var_class6.png) | [![horse](plots/variations_exp15_var_class7.png)](plots/variations_exp15_var_class7.png) | [![ship](plots/variations_exp15_var_class8.png)](plots/variations_exp15_var_class8.png) | [![truck](plots/variations_exp15_var_class9.png)](plots/variations_exp15_var_class9.png) |
