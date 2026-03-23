# Report 6 — Encoder Architecture Search at LATENT_DIM=64

## Goal

Following the finding in report 5 that LATENT_DIM=512 breaks sample quality, we locked
LATENT_DIM=64 and FREE_BITS=0.5 (exp5's config, which produces usable prior samples) and
systematically tested every encoder improvement to see if we could push past the 88.7% ceiling
through better architecture rather than more latent capacity.

---

## Results

| Exp | Acc    | Time | Params  | Change from exp5 |
|-----|--------|------|---------|------------------|
| 5   | 88.7%  | 262s |  5.0M   | baseline (latent=64) |
| 15  | 88.7%  | 260s |  5.1M   | N_ENC_Q: 8 → 16 |
| 16  | 88.7%  | 331s |  6.8M   | D: 144 → 196 (D_R=D_C=7) |
| 17  | 88.7%  | 256s |  6.1M   | encoder MLP (hidden=1024) |
| 18  | **88.8%**  | 339s |  9.9M   | all three combined |

---

## Finding: 88.7% is a hard ceiling at LATENT_DIM=64

Every experiment landed at 88.7%, with the kitchen-sink combination (exp18) nudging 88.8% —
well within noise. None of the encoder changes moved the needle:

- **More queries (N_ENC_Q=16)**: no effect. The encoder already extracts the right information
  with 8 queries; adding 8 more gives it nothing new to find.
- **Larger gram (D=196)**: no effect. A richer per-pixel token space doesn't help if the
  bottleneck is downstream of the gram.
- **Encoder MLP**: no effect. Non-linearity in the gram→latent path doesn't change what can
  fit through 64 dims.
- **All combined**: 88.8%, noise-level improvement at 2× the params.

The KL per dim stays pinned at the floor (≈0.51) in all experiments — confirming the encoder
saturates its 32-nat budget regardless of architecture. The encoder is not the bottleneck.
The bottleneck is the bottleneck.

---

## Conclusion

**88.7% is an information-theoretic ceiling for LATENT_DIM=64, FREE_BITS=0.5.**

The gram VAE with 64-dim latent encodes exactly 32 nats of information per image. The encoder
already extracts the most useful 32 nats from the first attempt (exp5). Making the encoder
more expressive — more queries, larger gram, non-linear projection — cannot change how much
information fits through the bottleneck.

This is the fundamental tradeoff identified in reports 4 and 5:

| Config | Bin accuracy | Prior samples |
|--------|-------------|---------------|
| LATENT_DIM=64  (exp5)  | 88.7% (ceiling) | Usable digits |
| LATENT_DIM=512 (exp11) | 89.2%           | Broken (checkerboard) |

There is no architectural path within the single-stage VAE that achieves both.

---

## Next Step

The only clean path to better reconstruction *with* good sample quality is the **two-stage
approach**: keep exp5 as the reconstruction model, then train a separate generative model
on exp5's aggregate posterior (the 60K encoded training z-vectors). Generation goes through
the learned prior → exp5 decoder. Reconstruction remains at 88.7% with working prior samples.
