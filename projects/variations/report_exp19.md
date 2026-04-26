# variations — exp19 report (CIFAR-10, feature-space kNN)

**Algorithm**: Wide CNN encoder + wide deconvnet decoder + dynamic encoder-space kNN.

Key change from exp17: pixel-space kNN replaced with encoder-embedding kNN.
kNN recomputed every 10 epochs using 128-d encoder embeddings (cosine similarity via FAISS).
Bootstrap: initial kNN from pixel space.

## Hyperparameters

| param | value |
|-------|-------|
| LATENT_DIM | 128 |
| Encoder | CNN: Conv(64,3×3,s2) → Conv(128,3×3,s2) → Conv(256,3×3,s2) → FC(512→128) |
| N_CAT | 4 |
| VOCAB_SIZE | 8 |
| COND_DIM | 16 |
| DC_STEM / DC_C1 / DC_C2 / DC_C3 | 1024 / 256 / 128 / 64 |
| K_NN | 64 |
| kNN type | dynamic encoder-space (cosine, FAISS), updated every 10 epochs |
| BATCH_SIZE | 128 |
| N_VAR_SAMPLES | 16 |
| MAX_EPOCHS | 100 |
| n_params | 7,539,600 |
| time | 2035.1s |

## Results

**Final reconstruction MSE**: `0.0046` (vs exp17 wide decoder: TBD, exp16 CNN: 0.002545)

## Learning curves

![Learning curves](plots/variations_exp19_curves.png)

## Reconstructions

![Reconstruction grid](plots/variations_exp19_recon.png)

## Variation grids

| class 0 (airplane) | class 1 (automobile) | class 2 (bird) | class 3 (cat) | class 4 (deer) |
| :---: | :---: | :---: | :---: | :---: |
| [![airplane](plots/variations_exp19_var_class0.png)](plots/variations_exp19_var_class0.png) | [![automobile](plots/variations_exp19_var_class1.png)](plots/variations_exp19_var_class1.png) | [![bird](plots/variations_exp19_var_class2.png)](plots/variations_exp19_var_class2.png) | [![cat](plots/variations_exp19_var_class3.png)](plots/variations_exp19_var_class3.png) | [![deer](plots/variations_exp19_var_class4.png)](plots/variations_exp19_var_class4.png) |
| class 5 (dog) | class 6 (frog) | class 7 (horse) | class 8 (ship) | class 9 (truck) |
| [![dog](plots/variations_exp19_var_class5.png)](plots/variations_exp19_var_class5.png) | [![frog](plots/variations_exp19_var_class6.png)](plots/variations_exp19_var_class6.png) | [![horse](plots/variations_exp19_var_class7.png)](plots/variations_exp19_var_class7.png) | [![ship](plots/variations_exp19_var_class8.png)](plots/variations_exp19_var_class8.png) | [![truck](plots/variations_exp19_var_class9.png)](plots/variations_exp19_var_class9.png) |
