# Seewhy — Research Index

All uploaded reports across projects.

---

## SSL — Self-Supervised Learning

Evaluating SSL training regimes (AE, DAE, VAE, EMA-JEPA, SigReg, Masked Distillation)
on 9 encoder architectures across MNIST and Fashion-MNIST.

| Report | Description |
|--------|-------------|
| [Feature Extractors: Full 6-Way Comparison](https://media.tanh.xyz/seewhy/26-04-06/ssl_report_ae_dae_vae.html) | AE / DAE / VAE / EMA / SigReg / Masked across 9 archs — K-Means, KNN, probe |
| [Masked Distillation Heatmap — EMA teacher](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_heatmap.html) | Arch×Arch heatmap: which student learns best from which teacher |
| [Masked Distillation Heatmap — DAE teacher](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_heatmap_dae.html) | Same heatmap with DAE-trained teacher |
| [Masked Distillation Heatmap — Random teacher](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_heatmap_random.html) | Baseline: untrained random teacher |
| [Masked Distillation Heatmap — SigReg teacher](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_heatmap_sigreg.html) | SigReg teacher variant |
| [Chain Distillation: DAE→MLP→sub-student](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_chain.html) | Can knowledge survive 2-hop cross-arch distillation? |
| [Chain Distillation: Same-arch, 10 hops](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_chain_same_arch.html) | Knowledge degradation over 10 same-arch distillation hops |
| [Chain Distillation: DAE-seeded, 10 hops](https://media.tanh.xyz/seewhy/26-04-07/ssl_report_distill_chain_dae_arch.html) | Same chain but starting from DAE teacher instead of random |

---

## small_lm — Tiny LM Training Dynamics

An 8.7M-param JAX transformer trained on synthetic persona conversations.
Studying architecture, pre-training, and identity fine-tuning dynamics.

| Report | Description |
|--------|-------------|
| [Index](https://media.tanh.xyz/seewhy/26-04-07/small_lm_index.html) | Project overview and file structure |
| [Baseline](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_baseline.html) | Direct GuppyLM port — 6L, d=384, 10K steps. Eval ppl=1.46 |
| [Optimizations](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_optimizations.html) | Muon, RoPE, separate LM head vs baseline. Pareto-front analysis |
| [AI Eval](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_ai_eval.html) | LLM-judged fish-persona quality: 5 rounds, 60 questions |
| [k-Dyck Pretraining](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_kdyck.html) | Procedural warm-up on balanced brackets before fine-tuning |
| [Babystep Identity — Kylo](https://media.tanh.xyz/seewhy/26-04-08/small_lm_babystep_kylo.html) | Teaching a model to roleplay a shiba inu via LLM-supervised self-play (4 runs, 600 recovered Q&A pairs) |
