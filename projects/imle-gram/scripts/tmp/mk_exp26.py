"""Create experiments26.py from experiments21.py with codebook cosine repulsion."""
from pathlib import Path

BASE = Path(__file__).parent.parent.parent

txt = open(BASE / "experiments21.py").read()

e26 = (txt
    .replace(
        "exp21 — 128-cluster with delta_sum assignment.",
        "exp26 — 128-cluster flat delta_sum + codebook cosine repulsion.\n\n"
        "Same as exp21 but adds a pairwise cosine repulsion loss on the codebook\n"
        "to push embeddings apart and prevent prototype collapse:\n\n"
        "    Z_n   = codebook / ||codebook||          # unit-normed rows\n"
        "    G     = Z_n @ Z_n.T                     # (K,K) cosine similarities\n"
        "    L_rep = (||G||_F² - K) / (K*(K-1))     # normalised to [0,1]\n"
        "    loss  = nll + REP_LAMBDA * L_rep\n\n"
        "REP_LAMBDA=0.1 caps the repulsion at ~7% of the NLL.")
    .replace(
        "uv run python projects/imle-gram/experiments21.py",
        "uv run python projects/imle-gram/experiments26.py")
    .replace('EXP_NAME = "exp21"', 'EXP_NAME = "exp26"')
    .replace("imle-cluster-deltasum", "imle-cluster-deltasum-rep")
    .replace(
        "ADAM_LR    = 3e-4",
        "ADAM_LR    = 3e-4\n\nREP_LAMBDA = 0.1   # weight for codebook cosine repulsion loss")
    .replace(
        "def imle_loss(params, x, matched_indices):\n"
        "    z_soft = soft_lookup(params, matched_indices)\n"
        "    logits = decode(params, z_soft)\n"
        "    bins   = quantize(x)\n"
        "    log_p  = jax.nn.log_softmax(logits, axis=-1)\n"
        "    return -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()",
        "def imle_loss(params, x, matched_indices):\n"
        "    z_soft = soft_lookup(params, matched_indices)\n"
        "    logits = decode(params, z_soft)\n"
        "    bins   = quantize(x)\n"
        "    log_p  = jax.nn.log_softmax(logits, axis=-1)\n"
        "    nll    = -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()\n"
        "\n"
        "    # Codebook cosine repulsion: push all pairs toward orthogonality\n"
        '    Z_n   = params["codebook"] / (jnp.linalg.norm(params["codebook"], axis=1, keepdims=True) + 1e-8)\n'
        "    G     = Z_n @ Z_n.T                                         # (K, K) cosine similarities\n"
        "    L_rep = ((G ** 2).sum() - N_CODES) / (N_CODES * (N_CODES - 1))   # normalised to [0,1]\n"
        "\n"
        "    return nll + REP_LAMBDA * L_rep")
)

out = BASE / "experiments26.py"
out.write_text(e26)
print(f"wrote {out}  ({len(e26)} chars)")
