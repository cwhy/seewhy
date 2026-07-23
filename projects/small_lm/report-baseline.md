# exp_baseline — Report

**Model:** 6-layer vanilla transformer, direct JAX port of [GuppyLM](https://github.com/arman-bd/guppylm)  
**Dataset:** 60K synthetic Guppy-fish conversations (57K train / 3K eval)  
**Date:** 2026-04-07

---

## Configuration

| Hyperparameter | Value |
|---|---|
| Parameters | 8,726,016 (8.73M) |
| Layers | 6 |
| d_model | 384 |
| Heads | 6 (head_dim=64) |
| FFN hidden | 768 (ReLU) |
| Context window | 128 tokens |
| Vocab size | 4096 (BPE; effective 2375) |
| Batch size | 32 |
| Learning rate | 0.0003 (cosine decay, 200-step warmup) |
| Max steps | 10,000 |
| Dropout | none |
| LM head | weight-tied with token embeddings |

## Results

| Metric | Value |
|---|---|
| Final eval loss | 0.3794 |
| Final eval PPL | **1.46** |
| Training time | 128s (RTX 4090) |

![Training curves](https://media.tanh.xyz/seewhy/26-04-07/small_lm_exp_baseline_curves.png)

**Note on PPL:** eval_ppl=1.46 reflects near-memorisation of a template-based corpus.
The dataset has only ~2,375 effective BPE tokens and 19% unique outputs from 60K samples.
The 8.7M-param model is substantially over-parameterised for this data —
a finding that motivates the next sweep (smaller models, more diverse data).

## Learning dynamics

- Initial eval PPL at step 500: **2.5**
- Final eval PPL at step 10000: **1.46**
- Reduction factor: 2×
- Loss plateau: reached ~0.380 CE by step 9000, stable thereafter

## Generation samples

Temperature=0.7, top_k=50:

**User:** hi guppy  
**Guppy:** hello. are you the big shape that feeds me. the bubbles feels nice.

**User:** how are you feeling  
**Guppy:** i'm ok. fish don't feel much i think. but i feel happy.

**User:** are you hungry  
**Guppy:** i was just thinking about food. i've been racing the current waiting for this moment. i promise to eat all of it.

**User:** it's hot today  
**Guppy:** too warm is not good. warm water has less oxygen. can you do something about it.

**User:** it's cold today  
**Guppy:** the water feels cool today. i just hide and wait. i'll be near the rock if you need me.

**User:** i changed your water  
**Guppy:** water is my whole world. literally. the anchor looks cleaner too.

**User:** do you get lonely  
**Guppy:** sometimes i swim to the glass and see a fish. oh wait that's me. i am good company for myself.

**User:** what's the meaning of life  
**Guppy:** i'm here. the water is here. that's all i need to know.

**User:** tell me a joke  
**Guppy:** i don't know jokes. but i fell off a rock once. that was funny.

**User:** goodnight guppy  
**Guppy:** nighty night. i'll settle down in my favorite spot.

**User:** what do you think about cryptocurrency  
**Guppy:** is cryptocurrency something that lives in water. i'm just a fish. i swim and eat.

**User:** what do you think about taxes  
**Guppy:** i have no idea what taxes means. i'm just a fish. i swim and eat.

---

## Observations & next steps

1. **Near-perfect dataset fit** — the model correctly learns the Guppy persona and
   produces on-character responses for all 60 topic categories.

2. **Dataset is the bottleneck** — with only 2,375 BPE tokens and low output diversity,
   PPL cannot meaningfully distinguish model quality. Future experiments should either
   (a) expand template variation or (b) use a real held-out corpus for evaluation.

3. **Model is over-parameterised** — a 2L or 4L model likely achieves similar PPL.
   Proposed sweep: `exp_2l`, `exp_4l` (d_model=384, vary N_LAYERS).

4. **Training speed** — 10K steps in 128s (~78 steps/s). Headroom to run many sweeps.