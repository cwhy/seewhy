# exp53 — Additive z+eps, eps ~ N(0, I_2048), 300ep

**HTML report**: https://media.tanh.xyz/seewhy/26-04-26/variations_exp53_report.html

## Result

| Metric | Value |
|--------|-------|
| Final recall MSE | **0.00843** |
| Best recall MSE | 0.00843 |
| Final recon MSE | 0.00107 |
| Training time | ~215 min |
| Epochs | 300 |

Worse than exp51 (Gaussian concat, 0.00787). Ruled out.

## Key Change from exp51

Instead of concatenating eps to z (exp51: IN_DIM=2112), add eps directly to z:
`z_var = z + eps` where `eps ~ N(0, I_2048)`. Decoder input stays at 2048.
Fewer parameters than exp51 (no extra decoder width).

## Verdict

**Ruled out.** Adding noise in latent space forces the decoder to resolve ambiguity
between natural latent variation and conditioning noise — harder than a dedicated
conditioning channel. Concatenation (exp51) is better. Discrete VOCAB scaling is better still.

## Comparison

| Exp | Conditioning | eps dim | Decoder input | Recall MSE |
|-----|-------------|---------|---------------|------------|
| exp49 | discrete VOCAB=32 | — | z ++ cond (2112) | **0.00658** |
| exp51 | Gaussian concat | 64 | z ++ eps (2112) | 0.00787 |
| **exp53** | **Gaussian additive** | **2048** | **z + eps (2048)** | **0.00843** |
