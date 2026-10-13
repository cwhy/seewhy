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

## random-transformers — Algorithmic Capabilities of Random Transformers

Independent replication of [arXiv:2410.04368](https://arxiv.org/abs/2410.04368)
(Zhong & Andreas, NeurIPS 2024): how much can a transformer compute when every
attention and feed-forward weight is frozen at its random initialisation, and
only the embedding and unembedding matrices are trained? Reimplemented in JAX
from the paper text; all seven experiments reproduced.

| Report | Description |
|--------|-------------|
| **[Frozen Random Transformers Already Compute](https://media.tanh.xyz/seewhy/paper/random-transformers_paper.html)** | **The paper — start here.** All four algorithmic tasks at 1.000 from five seeds out of five; 17 of 20 main-table cells within 0.15 of the published numbers; subspace selection and its falsification test both reproduce |
| [Warmup decides the baseline, not the frozen model](https://media.tanh.xyz/seewhy/26-08-19/random_transformers_report_budget.html) | The paper's stated optimiser cannot reproduce its own fully trained baselines. A 500-step warmup and cosine decay live in the authors' code and not in the text |

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
| [Plan — what to do next, and what we learned](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_plan.html) | The binding constraint (the context carries no usable signal at M=16), a staged plan to fix it, and the methodological lessons the first phase bought |
| [5 — Train short, test long; and how far recall travels](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_report_05-transfer.html) | Evaluation-only. Long contexts buy nothing at inference — the M=256 result is a training-time effect. And the recall mechanism is MNIST-shaped: 0.651 on Fashion-MNIST, 0.116 on the same pixels permuted |
| [3 — Perfect retrieval, chance-level completion](https://media.tanh.xyz/seewhy/26-08-12/recall-gen_report_03-digit-split.html) | Digit split (train 0–4, test 5–9): id. acc. 1.000 and nMSE 1.006 on the same unseen digits. Plus the state-size control and the fine-tuning probe |

---

## bayes-cbp — 跳跃重置即贝叶斯变点滤波

Formulation only, no experiments. Checks a note that derives Continual Backprop
style resets from Bayesian first principles via a jump-SDE. Source note in
[source-note.md](bayes-cbp/source-note.md).

| Report | Description |
|--------|-------------|
| [Paper (中文) — 跳跃重置即贝叶斯变点滤波](https://media.tanh.xyz/seewhy/26-10-09/bayes-cbp_paper.html) | CBP's jump term is exactly the prediction step of a change-point prior; reset probability = Bayes factor "fresh vs stay", which equals the base rate ρ for dead units. Four formula fixes to the note (forward equation, variance injection, mixing bound, the σ²-ratio reset strength) |

---

## curve-clusters — Clusters as Curves, Not Balls

Can a dataset with no natural order be compressed by storing it as smooth
curves rather than points? Order N is a dial whose N=1 setting is provably
online k-means, so the baseline lives inside the method. One report per
research question; each is readable standalone and carries the full background.

| Report | Description |
|--------|-------------|
| [Overview](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_probe.html) | **Entry point.** All five questions, the short answer, and two retracted claims |
| [Q1 · Is a curve better than a ball?](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q1_shape.html) | Yes at equal stored floats — 35–55%. But it is the straight stick, not the bend |
| [Q2 · Does it survive more budget and data?](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q2_scaling.html) | Yes. The trend suggesting otherwise was budget entangled with points-per-cluster |
| [Q3 · Is it just buying more addresses?](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q3_addresses.html) | Largely yes. At matched rate k-means wins on all five low-dim sets |
| [Q4 · Which dimension matters?](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q4_dimension.html) | Intrinsic, entirely. 3.3× at d=1, nothing by d=32. Real images: 3–7% |
| [Q5 · K or N?](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q5_k_or_n.html) | K. Orders 2–8 are indistinguishable; only N=1 vs N>1 matters |
| [Q6 · Trim the centre update?](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q6_trimmed.html) | Only helps on contaminated data; the half is too aggressive — keep ~80% |

---

## local-learning — Backprop vs Local Learning Rules

Replication of a widely shared ["quick MNIST test"](https://x.com/mayfer/status/2099541904241385615)
(2026-09-14) that put backprop, Sakana's PC-ALM ([arXiv:2605.31022](https://arxiv.org/abs/2605.31022))
and Hinton's Forward-Forward in one table at ~56K parameters. All three numbers
reproduce; the ranking turns out to be two architectures and one shared learning
rate rather than a comparison of learning rules. Two further arms: plain
predictive coding, the ablation that isolates what PC-ALM adds; EGGROLL
([arXiv:2511.16652](https://arxiv.org/abs/2511.16652)), a gradient-free
evolution strategy; and feedback alignment, which replaces backprop's backward
weights with fixed random matrices.

| Report | Description |
|--------|-------------|
| [**Paper** — Local Learning vs Backprop](https://media.tanh.xyz/seewhy/paper/local-learning_paper.html) | **Start here.** The full write-up for a reader new to the project. Stable URL |
| [Slides — Can cortex run predictive coding?](https://media.tanh.xyz/seewhy/paper/local-learning_slides.html) | A 12-slide deck on PC and PC-ALM from the biological-plausibility angle alone: what each fixes, what each still owes (weight transport, in both cases), and the one experiment that would settle it. Drawn circuits, stable URL |
| [Auditing PC-ALM and EGGROLL](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_audits.html) | Both check out: PC-ALM reproduces its own theorem to six decimals and matches the authors' code to within half a point when run side by side; EGGROLL's estimator is unbiased and its low-rank claim holds. One finding: antithetic sampling costs EGGROLL a factor of sqrt(2) per forward pass here |
| [Auditing the Forward-Forward implementation](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_ff_audit.html) | Is our Forward-Forward really Forward-Forward? Mechanics verified; uniform negative labels found to cost 1.7 points shallow and 28.2 deep, which withdraws two of the paper's conclusions |
| [Five epochs, not one](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_five_epochs.html) | Does the budget hold the ranking up? Re-tuned per arm for the longer budget — six of eight want a smaller rate — the one-epoch ordering survives and the spread widens. Forward-Forward's apparent decay turns out to be a rate effect |
| [Replicating a "quick MNIST test"](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_replication.html) | Lab note: the three numbers reproduced over five seeds, then the controls the table is missing — backprop on Forward-Forward's own architecture (0.9116 vs 0.8869), Forward-Forward on the deep-narrow one (0.5751), and a gradient-free arm that reaches 0.9025 with an update almost orthogonal to the true gradient |

---

## dust — DUST vs EGGROLL: Forward-Only Transformer Pretraining

Replication of Q Labs' [Dust](https://qlabs.sh/research/dust) (Oct 2026): train a transformer language model
with no backward pass by perturbing *activations*, independently at every token, so one forward pass is one
population member per token. The question: does its claimed advantage over weight-space evolution strategies
(EGGROLL, the arm from [local-learning](#local-learning--backprop-vs-local-learning-rules)) hold up with the baseline
implemented here? JAX port of the authors' released code, the paper's FineWeb protocol at 100K, 1M and 10M tokens.
Reports and paper explain both algorithms with diagrams. The ordering reproduces, and so does the order of magnitude of
the efficiency claim, with a wide interval: a three-point extrapolation said hundreds and a fourth point moved it to thousands.
Under Adam and Muon, backprop gains more than DUST: DUST's equality with backprop at 1,024 draws is a property of SGD.

| Report | Description |
|--------|-------------|
| [**Paper** — DUST vs EGGROLL](https://media.tanh.xyz/seewhy/paper/dust_paper.html) | **Start here.** Both algorithms explained for a reader new to them, the 100K / 1M / 10M-token ladders, the alignment study, the extrapolation and its warning, the Adam and Muon runs, and the limitations. Stable URL |
| [Better optimisers cut DUST's loss, but backprop's more](https://media.tanh.xyz/seewhy/26-10-08/dust_report_6_optimisers.html) | The 1M-token ladder again under Adam and Muon. Muon takes DUST at 1,024 draws from 5.939 to 5.466; backprop falls to 5.358 (Adam) and 5.372 (Muon), so DUST goes from level to 0.095 behind, and at 4,096 draws (one seed) is 0.036 behind. EGGROLL is not helped by Muon |
| [At 10M tokens DUST trails backprop by 0.25 nats; at 100K nothing clears the floor](https://media.tanh.xyz/seewhy/26-10-08/dust_report_5_other_budgets.html) | The budgets either side of the ladder. DUST at 256 draws 5.232 (paper 5.231), backprop 4.984; EGGROLL 7.128 vs 6.906 published. Six updates at 100K are too few for any method to beat the token-frequency floor |
| [64 DUST draws beat 4,096 EGGROLL passes: 0.7 nats, at 59× less compute](https://media.tanh.xyz/seewhy/26-10-08/dust_report_4_ladder.html) | The 1M-token ladder. Under SGD, DUST at 1,024 draws (5.939) is level with backprop (5.937), and at 4,096 (one seed) 0.047 below it. Extrapolating EGGROLL's ladder: 3,554× to 16,810× DUST's 64 draws by two line fits, never by a power law; the paper's 6,000–8,200× |
| [DUST's gradient aligns with backprop's; EGGROLL's stays near zero](https://media.tanh.xyz/seewhy/26-10-08/dust_report_3_alignment.html) | Cosine to the backprop gradient against population on a 100M-token checkpoint. DUST 0.99 on the head and 0.73 on attention at 16,384 draws; EGGROLL 0.024 and 0.010, matching the random-direction prediction; at 131,072 passes EGGROLL is still under 0.05 except the head |
| [Our EGGROLL baseline trails the paper's](https://media.tanh.xyz/seewhy/26-10-08/dust_report_2_eggroll_baseline.html) | EGGROLL explained, then the tuning grid. Weaker than the paper's by 0.34 nats at 64 passes and 0.08 at 4,096; rate, noise scale, Adam constants, rank shaping and float32 do not close it |
| [The JAX port reproduces DUST's published numbers](https://media.tanh.xyz/seewhy/26-10-07/dust_report_1_port.html) | How DUST works in one page, then the checks: backprop (5.937 vs 5.959 published), DUST at 256 and 1,024 draws (6.074 vs 6.067; 5.939 vs 5.934), and the gradient estimate against the authors' own PyTorch on identical weights |

---

## recall-gan — An In-Context GAN Where the Discriminator Wins

A GAN whose real data is a 16-image context: the discriminator reads the
context into one KDA state and must accept context images and reject the
generator's. The discriminator is the product. Method and current results in
the paper; run history in [experiment_log.md](recall-gan/experiment_log.md).

| Report | Description |
|--------|-------------|
| [**Paper** — Recall-GAN](https://media.tanh.xyz/seewhy/paper/recall-gan_paper.html) | **Start here.** The method, self-contained. With a context-aware generator the discriminator relies on its state (swap AUROC 0.970 → 0.570), rejects unseen real images it never trained on (0.961), and classifies digits from 16 examples at 0.839 vs 0.788 for the best pixel RBF density. 0/12 failures with Adam β1 0 / β2 0.99. Stable URL |
| [6 — Trained on synthetic worlds, the discriminator learns the world, not the item](https://media.tanh.xyz/seewhy/26-10-11/recall-gan_report_06_synthetic_to_mnist.html) | Standalone. Trained on recall-gen's synthetic prior, tested on MNIST: no transfer (AUROC 0.51–0.52, classifier 0.12–0.15). Even on held-out worlds it scores stored and unstored items of a world alike (0.502) while separating other worlds (0.81–0.87) |
| [5 — Pretraining the discriminator does not survive the GAN](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_report_05_pretrained_discriminator.html) | Standalone. A discriminator pretrained on real non-members reaches AUROC 0.995 alone, loses it within 500 GAN steps, and 4/6 context-aware runs then collapse (0/6 from scratch) |
| [4 — A blind generator collapses under prefix training too](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_report_04_prefix_blind.html) | Prefix training with a generator that cannot see the context: 12/12 collapse within 2k steps, the discriminator ignores its state and classifies at chance. The blind generator cannot copy, but gives no pressure at all |
| [3 — Random real queries do not stop the copying](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_report_03_random_real_queries.html) | Drawing the real queries at random makes the prefix-trained generator copy the newest image more (88% vs 51%, chance ~15%) — almost pixel for pixel. A near-copy of a context image labelled "not in context" is unsatisfiable: twin AUROC 0.436 |
| [2 — Under prefix training the generator copies the last image](https://media.tanh.xyz/seewhy/26-10-10/recall-gan_report_02_prefix_training.html) | Training on every prefix ("teacher forcing"), ± nearest-image context: the generator draws the last context image's digit 51% / 91% of the time (chance ~15%), which teaches the discriminator to reject that digit — 1-example digit classifier below chance (0.084); 16-example 0.529 vs 0.843 plain. With the nearest-image context the generator fills the context with one digit |
| [1 — The discriminator wins; the generator learns one trick](https://media.tanh.xyz/seewhy/26-10-08/recall-gan_report_01_progress.html) | *Superseded.* Built on experiments outside the project's design (extra negatives, frozen generator, warm start); kept as a record |
