# Enhancing Evolutionary Algorithms for Large-Scale Language Models: Insights from Recent Research

## Background

Evolutionary algorithms (EA) refer to a class of black-box optimisation methods, such as Evolution Strategies (ES), that sample parameter perturbations from a distribution, evaluate them using a fitness function and then update the population distribution to favour better solutions. EA has several appealing properties for large language model (LLM) training: it does not require differentiability and is highly parallelisable because evaluations of different perturbations are independent. These methods can handle discrete and noisy objectives and are attractive for fine-tuning or pretraining LLMs when back-propagation is expensive or impractical. However, naive EA can be inefficient for billion-parameter models. Random perturbations require massive matrix multiplications and produce low arithmetic intensity, making GPU throughput poor.

The question "can we have better ways to do EA?" is timely because recent papers and blogs propose novel methods for post-training or pretraining LLMs using either random search or improved ES, and some offer principles about why certain optimisers (e.g., Muon) behave favourably. This report synthesises three recent research papers and a Chinese blog to assess the state of EA for LLMs and to suggest better practices.

---

## Neural Thickets: Random Guessing and Ensembling Around Pretrained Weights

**Paper:** *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* (Yulu Gan & Phillip Isola, 2026)

### Key Ideas

1. **Dense neighborhoods around pretrained weights.** After pretraining, the parameter space around an LLM's weights contains many task-specific solutions. The authors show that in small models this "solution density" is tiny; good perturbations are needles in a haystack. In large, well-pretrained models, the density of task-improving weights increases dramatically. Figure 3 in the paper demonstrates scaling laws: the fraction of random perturbations that improve the base model increases with model size.

2. **Diversity of perturbations.** Perturbations that improve one task often hurt performance on others, showing the neighborhood is diverse. Diversity also scales with model size.

3. **RandOpt algorithm.** Motivated by high solution density, the authors propose RandOpt, a fully parallel post-training method. It samples N random Gaussian perturbations of the pretrained weights, evaluates them on a downstream task, selects the top K, and ensembles their predictions via majority vote. RandOpt requires no gradient computation and runs in O(1) training steps; the main cost is inference-time ensembling.

4. **Performance.** RandOpt achieves accuracy competitive with reinforcement-learning-based methods such as PPO, gradient-reinforced policy optimisation (GRPO) and classical ES, when given similar FLOPs budgets. Although not always superior, its success indicates that once a model is large enough, post-training becomes easy; random guessing with ensembling is viable.

### Strengths

- **Simplicity and parallelism:** RandOpt is conceptually simple and embarrassingly parallel. It requires no gradient calculation or back-propagation, making it appealing for tasks where gradient information is unavailable or expensive.
- **Scaling insight:** The paper provides empirical evidence that solution density and diversity scale with model size, explaining why random perturbations can work.
- **Competitive performance:** With enough perturbations, RandOpt matches gradient-based post-training methods on tasks such as reasoning, programming and chemistry.

### Limitations

- **Inference overhead:** Ensembling top K models increases inference cost by roughly a factor of K. The authors mention distillation as a possible solution but do not fully address inference latency.
- **Large evaluation budget:** Random sampling needs thousands of perturbations (e.g., 5,000) to find good experts. For very large models, evaluating so many variants may still be costly.
- **No structure in perturbations:** All perturbations are full-rank Gaussian noise; they ignore potential structure in LLM weight matrices. This leads to poor arithmetic intensity and may not scale to extremely large models.

---

## Evolution Strategies at the Hyperscale: Low-Rank Perturbations (EGGROLL)

**Paper:** *Evolution Strategies at the Hyperscale* (Bidipta Sarkar et al., revised Feb 2026)

### Key Ideas

1. **Low-rank perturbations (EGGROLL).** The authors introduce Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL). Instead of sampling full-rank matrices for perturbations, EGGROLL constructs a perturbation for each weight matrix M ∈ ℝ^{m×n} as E = (1/r) AB⊤, where A ∈ ℝ^{m×r} and B ∈ ℝ^{n×r} are random matrices with r ≪ min(m,n). This reduces storage from mn to (m+n)r and lowers the amount of data that needs to be moved across the GPU. Low-rank adapters here play a role analogous to LoRA in gradient-based training.

2. **High arithmetic intensity.** EGGROLL batches a population of low-rank adapters and shares the base activations, enabling a single forward pass to apply all AB⊤ updates via specialised batched matrix multiplications. This yields over a hundredfold increase in training throughput for billion-parameter models.

3. **Full-rank update by averaging.** Although each individual perturbation is low-rank, the final update is the weighted average of many rank-r perturbations; thus the aggregate update has rank min(Nr, m, n) and is effectively full-rank.

4. **Theoretical analysis.** The authors analyse Gaussian ES in high dimensions, showing that with an appropriate noise scaling σ = O(d^{-1/2}), the ES update linearises and converges to the gradient for a broad class of objectives. They prove that fixed low-rank updates converge to full-rank Gaussian ES at a rate O(r^{-1}).

5. **Empirical results.** EGGROLL matches or outperforms naïve ES and GRPO on a wide range of tasks while running up to 100× faster. It enables stable pretraining of an int8 recurrent language model purely with ES.

### Strengths

- **Scalability:** By using low-rank perturbations and batching, EGGROLL attains near-batch-inference throughput on GPUs. This makes ES practical for models with billions of parameters.
- **Theoretical grounding:** The paper offers rigorous analysis, showing that low-rank ES converges to full-rank ES as dimension grows.
- **Flexible objective:** EGGROLL retains ES's ability to optimise non-differentiable or integer models (e.g., int8 RNNs).

### Limitations

- **Still requires many evaluations:** Although faster, EGGROLL remains a population-based method; it needs large populations (up to 1,048,576) for stable convergence.
- **Low-rank distribution design:** Choosing the rank r and sampling distribution for A and B is non-trivial and may affect performance. Implementation complexity is higher than simple Gaussian perturbations.
- **Potential mode collapse:** Low-rank search spaces might miss directions not spanned by the sampled low-rank subspace unless the population is large enough.

---

## Muon Is Scalable for LLM Training: Orthogonalised Momentum and Scaling Laws

**Paper:** *Muon is Scalable for LLM Training* (Jingyuan Liu et al., Feb 2025)

### Key Ideas

1. **Muon optimiser.** Muon, introduced by K. Jordan et al. (2024), updates matrix parameters using orthogonalised momentum via the Newton–Schulz iteration. Given gradient momentum M_t, it computes an orthogonal matrix O_t = NewtonSchulz(M_t) and updates weights via W_t = W_{t−1} − η_t O_t. Orthogonalisation ensures updates are isometric, preventing the optimiser from following dominant directions only.

2. **Scalability improvements.** The authors find Muon's performance on small models does not immediately translate to billion-parameter LLMs. They identify two key issues and propose fixes:
   - **Add weight decay:** Without weight decay, Muon's weights and layer outputs grow unbounded. Incorporating the standard AdamW weight decay rule (W_t = W_{t−1} − η_t(O_t + λW_{t−1})) stabilises training and improves performance. Experiments show that Muon with weight decay outperforms vanilla Muon and AdamW in the long term.
   - **Consistent update scale:** Muon's update RMS depends on the shape of the matrix. For a matrix of shape [A, B], the theoretical update RMS is 1/√max(A,B). This variability causes updates to be too small for wide matrices and too large for narrow ones. The authors propose scaling the update by √max(A,B) and matching the RMS to that of AdamW (around 0.2–0.4). With this adjustment, Muon can reuse AdamW's learning rate and weight decay hyper-parameters.

3. **Distributed implementation:** They implement Muon using ZeRO-1 style state partitioning to reduce memory and communication costs.

4. **Scaling law experiments.** Experiments show that Muon achieves roughly 2× computational efficiency compared to AdamW under compute-optimal training and improves the performance–FLOP Pareto frontier when training large mixture-of-experts (MoE) models. The released 16B-parameter Moonlight MoE model trained with Muon outperforms comparable models using fewer FLOPs.

### Strengths

- **Better optimisation geometry:** Orthogonalising the gradient momentum aligns updates with the spectral norm, providing a principled steepest-descent direction under operator norm constraints. This may help maintain stability in very deep networks.
- **Scalability fixes:** The paper analyses why Muon fails to scale and provides two practical adjustments (weight decay and RMS matching) that enable out-of-the-box scaling.
- Open-source implementation and checkpoints facilitate reproduction and further research.

### Limitations

- **Requires matrix-structured parameters:** Muon only applies to matrix weights; other parameters (e.g., bias vectors, layer norm scales) must still use optimisers like AdamW.
- **Orthogonalisation cost:** Newton–Schulz iterations add extra matrix multiplications, though the authors use only five iterations and find little benefit in increasing this number.

---

## Insights from "Why Do We Prefer Isotropy? A Steepest-Descent Perspective"

**Blog:** *为什么我们偏爱各向同性？基于最速下降的理解* (Why do we prefer isotropy? A steepest-descent perspective)

Although the original blog is not fully accessible, reposts and summaries reveal its key argument: optimisation dynamics depend on the metric under which the steepest descent is defined. In deep learning, we often impose normalisation (BatchNorm, LayerNorm, RMSNorm) and whiten inputs to promote **isotropy** — the property that feature dimensions have equal variance.

The blog argues that when the feature space is isotropic, the steepest descent direction in parameter space coincides with the steepest descent in feature space, making optimisation more efficient. If features are anisotropic (unequal variance), gradient descent may update parameters in directions that have little effect on the output, leading to slower convergence. Therefore, normalisation and orthogonalisation (as used in Muon) help align the geometry of the optimiser with the true curvature of the loss landscape.

This insight provides a theoretical underpinning for why techniques such as Muon's orthogonalised updates, LoRA's low-rank adapters and isotropic perturbations can be beneficial.

---

## Towards Better Evolutionary Algorithms for LLMs

The reviewed works reveal several trends and open avenues to improve EA for large models.

### 1. Use Structure in Perturbations

- **Low-rank or factorised perturbations:** EGGROLL shows that sampling low-rank perturbations dramatically improves GPU efficiency while maintaining ES behaviour. Future EA methods should leverage matrix structure (e.g., block-diagonal, Kronecker-factored or FFT-based) to reduce compute and memory.
- **Layer-wise adaptive noise scales:** Instead of constant variance, adapt the perturbation scale per layer (e.g., inversely proportional to layer norm or weight RMS) to balance exploration across layers. This echoes the isotropy principle — ensuring each parameter contributes equally to the fitness gradient.

### 2. Combine EA with Gradient Information

- **Hybrid algorithms:** One can initialise the population distribution around gradient-based solutions (e.g., from Muon or AdamW) and update using ES only on a low-dimensional subspace. Prior works such as GRPO use gradient information to guide population updates. Combining EGGROLL's low-rank ES with Muon's orthogonalised gradients may yield faster convergence than pure EA.
- **Gradient-informed sampling:** Instead of purely random perturbations, sample directions from the span of recent gradients or use principal components of the gradient covariance. This focuses exploration on directions with high curvature.

### 3. Exploit Solution Density and Diversity

- **Adaptive population size and selection:** Neural Thickets shows that solution density increases with model size. In high-density regimes, small populations may suffice; in low-density regimes, larger populations or more structured search (e.g., CMA-ES) might be needed. One could adaptively increase the number of perturbations until enough improvements are found.
- **Ensemble distillation:** RandOpt's ensembling cost can be mitigated by distilling the ensemble into a single model or by using weighted averaging of weights ("model soups").
- **Task-specific selection:** Because perturbations are specialists, one could maintain multiple expert populations for different tasks and switch between them, akin to mixture-of-experts gating.

### 4. Apply Isotropy and Normalisation Principles

- **Normalise features and gradients:** Following the blog's reasoning, EA can benefit from maintaining isotropy in the parameter update space. For example, weight normalisation or spectral normalisation can ensure that each parameter direction has similar impact on the output, making random perturbations more effective.
- **Orthogonalise perturbations:** Similar to Muon's orthogonalised gradients, one could orthogonalise random perturbation vectors to sample directions that span diverse subspaces and avoid concentrating exploration in dominant eigenvectors of the Hessian.

### 5. Expand EA to New Modalities and Objectives

- **Discrete and black-box objectives:** EGGROLL demonstrates ES can train int8 RNNs and tasks with non-differentiable objectives. Future work can explore EA for RL tasks with sparse rewards or for aligning models to human preferences when feedback is limited.
- **Population-based training for meta-learning:** Use EA to evolve optimiser hyper-parameters (learning rate schedules, weight decay, clip norms) or architecture choices, while using gradient descent to update weights. This leverages EA's strength in high-level search.

---

## Conclusion

Recent research suggests that evolutionary algorithms remain a viable and increasingly efficient tool for large-scale LLM post-training and pretraining. The *Neural Thickets* paper reveals that large pretrained models inhabit weight-space regions where many random perturbations yield task-specific improvements; simple random search plus ensembling can rival gradient-based methods. *Evolution Strategies at the Hyperscale* introduces EGGROLL, a low-rank ES that achieves hundredfold speed-ups on GPUs by structuring perturbations and provides theoretical guarantees on convergence. *Muon is Scalable for LLM Training* analyses why orthogonalised gradient optimisers need weight decay and update-scale adjustments to scale and shows that Muon plus these fixes can outperform AdamW with half the FLOPs. A broader principle underlying these works is isotropy — ensuring that optimisation directions align with meaningful dimensions in feature space; both orthogonalisation (Muon) and normalisation aim at achieving this.

To answer the question "can we have better ways to do EA?" the evidence points to yes. Better EA for LLMs should:

1. Exploit structure (low-rank or orthogonal perturbations) to improve efficiency
2. Integrate gradient information and adapt perturbation scales
3. Take advantage of high solution density through adaptive population and ensembling strategies
4. Apply isotropy and normalisation principles to align search directions
5. Expand EA to new tasks and modalities

These innovations promise to make EA competitive with, and sometimes superior to, gradient-based methods in the era of large language models.

---

## References

- **Evolution Strategies at the Hyperscale** — https://arxiv.org/pdf/2511.16652.pdf
- **Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights** — https://arxiv.org/pdf/2603.12228.pdf
- **Muon is Scalable for LLM Training** — https://arxiv.org/pdf/2502.16982.pdf
