# variations — experiment summary

## Reports

| exp | description | recall_mse | recon_mse | time | report |
|-----|-------------|------------|-----------|------|--------|
| exp35 | Local IMLE, precomputed kNN, 128 samples \| 16 images, 4×8×4+2048 | 0.00666 | 0.00081 | 8.2h | — |
| exp37 | Global IMLE (non-exclusive), FAISS, 64 samples \| — images, 4×8×4+2048 | 0.04646 | — | 186 min | — |
| exp38 | Global IMLE + exclusive Voronoi, 64 samples \| — images, 4×8×4+2048 | 0.02310 | — | 187 min | — |
| exp39 | Global IMLE + Voronoi + recon loss, 64 samples \| — images, 4×8×4+2048 | 0.02113 | 0.00074 | 193 min | — |
| exp40 | Global IMLE + 2nd-nearest target, 64 samples \| — images, 4×8×4+2048 | 0.02304 | — | 45 min | — |
| exp41 | Local IMLE on-the-fly kNN, 64 samples \| 16 images, 4×8×4+2048 | 0.00749 | 0.00113 | 65 min | [report](https://media.tanh.xyz/seewhy/26-04-25/variations_exp41_report.html) |
| exp42 | exp41 + N_VAR=96 + 100ep + stop_grad kNN, 96 samples \| 16 images, 4×8×4+2048 | 0.00784 | 0.00116 | 72 min | [report](https://media.tanh.xyz/seewhy/26-04-25/variations_exp42_report.html) |
| exp43 | exp42 + 150ep schedule, 96 samples \| 16 images, 4×8×4+2048 | 0.00739 | 0.00110 | 118 min | [report](https://media.tanh.xyz/seewhy/26-04-25/variations_exp43_report.html) |
| exp44 | exp43 + COND_DIM=64 (E_DIM 4→16), 96 samples \| 16 images, 4×8×16+2048 | 0.00715 | 0.00115 | 99 min | [report](https://media.tanh.xyz/seewhy/26-04-25/variations_exp44_report.html) |
| exp45 | exp44 + N_VAR=128, two-phase chunked IMLE, 128 samples \| 16 images, 4×8×16+2048 | **0.00703** | 0.00113 | 107 min | [report](https://media.tanh.xyz/seewhy/26-04-25/variations_exp45_report.html) |

## Key findings

**Local vs global IMLE (exp35–40)**
- Global Voronoi cells cover ~781 images each across the full dataset; recall metric measures K=16 *local* neighbours → 3× gap vs local IMLE
- On-the-fly local kNN (B×50k GEMM per batch) closes the gap without a precomputed file

**On-the-fly kNN matches precomputed quality (exp41 vs exp35)**
- Single forward+backward pass since kNN targets come from fixed X_train
- 30s/epoch vs 148s/epoch — 5× faster due to Python for-loop vs lax.scan
- Recall within 12% of precomputed at the same epoch budget

**Schedule length matters more than N_VAR (exp42 vs exp43)**
- 100ep schedule insufficient: model still improving at epoch 99 (exp42 recall=0.00784)
- 150ep schedule converges fully via early stopping (exp43 recall=0.00739)
- N_VAR 64→96 provides no measurable benefit on its own

**COND_DIM is a real bottleneck (exp43 vs exp44)**
- E_DIM 4→16 (COND_DIM 16→64) improves recall by 3.2%: 0.00739 → 0.00715
- Wider conditioning gives the decoder more leverage to steer across the local neighbourhood
- Remaining gap to exp35 (0.00666): ~7%

**Two-phase chunked IMLE unlocks N_VAR=128 (exp45)**
- Phase 1: decode N_VAR in chunks with stop_gradient → no activation storage → fits in VRAM
- Phase 2: re-decode only K=16 winners with full gradient → 1024 items vs 8192
- Same ~44s/epoch as exp44 despite 33% more candidates

## Architecture (exp41+)

- **Encoder**: CNN Conv(64/128/256, stride-2) + FC(2048→2048) → LATENT=2048
- **Decoder**: MLP(IN_DIM→1024→256×4×4) + ConvTranspose(256/128/64) → 32×32×3
- **Conditioning**: N_CAT=4 categoricals, VOCAB_SIZE=8, E_DIM=4→16, COND_DIM=16→64
- **IMLE**: inverse, K=16 local neighbours via B×50k GEMM, N_VAR candidates per step
- **Loss**: L_imle + 1.0 × L_recon (zero-cat reconstruction)
- **kNN**: on-the-fly exact pixel-L2 via GPU GEMM, no precomputed file

## Parameter scaling levers

| param | current | what it controls |
|-------|---------|-----------------|
| `LATENT_DIM` | 2048 | encoder content code size |
| `N_VAR_SAMPLES` | 96→128 | IMLE search breadth per step |
| `K_NN` | 16 | neighbours per image (training targets) |
| `N_CAT` | 4 | combinatorial space axes |
| `VOCAB_SIZE` | 8 | values per axis (8⁴=4096 combinations) |
| `E_DIM` | 4→16 | conditioning dims per category |
| `COND_DIM` | 16→64 | total decoder conditioning input |
| `DC_STEM/C1/C2/C3` | 1024/256/128/64 | decoder capacity |
| `ENC_C1/C2/C3` | 64/128/256 | encoder capacity |
