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

---

## sparse-attn-emergence — Emergent Capabilities & Sparse Attention

Small-scale replication of [arXiv:2606.25010](https://arxiv.org/abs/2606.25010): do
capabilities emerge abruptly, at seed-random times, because a sparse task-relevant
attention pattern is hard for SGD to find? Synthetic tasks where the correct attention
pattern is known by construction. 16 seeds per config, all trained simultaneously under
one `jax.vmap`.

Published as a linked **minisite** — every page reachable from the hub. R2 keys are
date-foldered, so republishing mints new URLs; this row points at the latest run.

| Report | Description |
|--------|-------------|
| [Overview — hub](https://media.tanh.xyz/seewhy/26-08-07/sparse_attn_emergence_index.html) | Claims table H1–H5 with verdicts and reading order — start here |
| [The paper in plain terms](https://media.tanh.xyz/seewhy/26-08-07/sparse_attn_emergence_paper.html) | Explainer for readers who haven't read the paper: what emergence is, why sparse attention would explain it |
| [Findings](https://media.tanh.xyz/seewhy/26-08-07/sparse_attn_emergence_findings.html) | All seven experiments in one place, with diagrams. H1–H4 hold; H5 in direction only; a degenerate column found in the task design |
| [Mistakes](https://media.tanh.xyz/seewhy/26-08-07/sparse_attn_emergence_mistakes.html) | Every error made, what it cost, and how each surfaced — including one wrong claim caught only by reader pushback |

---

## recall-gen — Does In-Context Recall Generalise?

Each MNIST image is one token. A KDA linear RNN holds M context images in a fixed
16 384-number state; the model completes a query image whose bottom half is masked.
Training uses only episodes where the answer is already in the context, so the task
is pure look-up — then we withhold the answer and see what is left. Retrieval and
generalisation are mutually exclusive by construction, which they are not in language.

| Report | Description |
|--------|-------------|
| [**Paper** — Recall-Gen](https://media.tanh.xyz/seewhy/paper/recall-gen_paper.html) | **Start here.** The full write-up for a reader new to the project. Stable URL |
| [1 — Recall generalises, completion does not](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_report_01-recall-only.html) | Retrieval transfers to unseen images at id. acc. 1.000; completion ends worse than ridge regression and *decays* through training |
| [2 — Generalisation appears when retrieval fails](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_report_02-context-size.html) | Context-size sweep. At M=256 the recall-trained model gains nothing from the answer being present and lands on the completion-trained ceiling |
| [**4 — Does training on recall produce generalisation?**](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_report_04-two-senses.html) | **The question, answered directly.** Separates the two senses of "generalisation" the other reports conflate; four purpose-built figures. Read this before 1–3 |
| [3 — Perfect retrieval, chance-level completion](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_report_03-digit-split.html) | Digit split (train 0–4, test 5–9): id. acc. 1.000 and nMSE 1.006 on the same unseen digits. Plus the state-size control and the fine-tuning probe |
