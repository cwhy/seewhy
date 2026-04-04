"""
Generate EMA experiments at LATENT_DIM = 512, 1024, 2048.
(exp_ema already covers LATENT=4096)

Adds SIGReg as a diagnostic metric (computed at eval checkpoints, not in loss)
so we can compare embedding quality across latent sizes even without SIGReg training.
"""

from pathlib import Path

SSL  = Path(__file__).parent.parent.parent
src  = (SSL / "exp_ema.py").read_text()

# ── Patch: add SIGReg diagnostic to base source ───────────────────────────────

# 1. Add N_PROJ / SIGREG_KNOTS to hyperparams (after EMA_TAU line)
src = src.replace(
    "EMA_TAU      = 0.996   # EMA momentum (target = τ*target + (1-τ)*online)\n\nADAM_LR",
    "EMA_TAU      = 0.996   # EMA momentum (target = τ*target + (1-τ)*online)\n\n"
    "N_PROJ       = 1024   # SIGReg diagnostic: random projection directions\n"
    "SIGREG_KNOTS = 17     # SIGReg diagnostic: quadrature knots over [0, 3]\n\nADAM_LR"
)

# 2. Add SIGReg precomputed constants (after ROWS/COLS)
src = src.replace(
    "# ── Utilities ─────────────────────────────────────────────────────────────────",
    "# SIGReg quadrature (diagnostic only — not used in training loss)\n"
    "_t_vec = jnp.linspace(0.0, 3.0, SIGREG_KNOTS)\n"
    "_dt    = 3.0 / (SIGREG_KNOTS - 1)\n"
    "_trap  = jnp.full(SIGREG_KNOTS, 2.0 * _dt).at[0].set(_dt).at[-1].set(_dt)\n"
    "SIGREG_PHI     = jnp.exp(-0.5 * _t_vec ** 2)\n"
    "SIGREG_WEIGHTS = _trap * SIGREG_PHI\n\n"
    "# ── Utilities ─────────────────────────────────────────────────────────────────"
)

# 3. Add sigreg() function after n_params/append_result
src = src.replace(
    "# ── Parameters ────────────────────────────────────────────────────────────────",
    "# ── SIGReg diagnostic ─────────────────────────────────────────────────────────\n\n"
    "def sigreg_diag(emb, key):\n"
    "    \"\"\"Epps-Pulley statistic on emb — diagnostic only, not used in loss.\"\"\"\n"
    "    B   = emb.shape[0]\n"
    "    A   = jax.random.normal(key, (LATENT_DIM, N_PROJ))\n"
    "    A   = A / jnp.linalg.norm(A, axis=0, keepdims=True)\n"
    "    proj = emb @ A\n"
    "    x_t  = proj[:, :, None] * _t_vec[None, None, :]\n"
    "    cos_mean = jnp.cos(x_t).mean(0)\n"
    "    sin_mean = jnp.sin(x_t).mean(0)\n"
    "    err = (cos_mean - SIGREG_PHI[None, :]) ** 2 + sin_mean ** 2\n"
    "    return (err @ SIGREG_WEIGHTS * B).mean()\n\n"
    "# ── Parameters ────────────────────────────────────────────────────────────────"
)

# 4. Add sigreg_diag to history init and eval block in train()
src = src.replace(
    '    history      = {"loss": [], "probe_acc": []}',
    '    history      = {"loss": [], "probe_acc": [], "sigreg_diag": []}'
)
src = src.replace(
    "        probe_str = \"\"\n"
    "        if (epoch + 1) % EVAL_EVERY == 0:\n"
    "            # Evaluate online encoder (what gets trained)\n"
    "            acc = linear_probe_accuracy(online_params, X_tr, Y_tr, X_te, Y_te)\n"
    "            history[\"probe_acc\"].append((epoch, acc))\n"
    "            probe_str = f\"  probe_acc={acc:.1%}\"",

    "        probe_str = \"\"\n"
    "        if (epoch + 1) % EVAL_EVERY == 0:\n"
    "            # Evaluate online encoder (what gets trained)\n"
    "            acc = linear_probe_accuracy(online_params, X_tr, Y_tr, X_te, Y_te)\n"
    "            history[\"probe_acc\"].append((epoch, acc))\n"
    "            # SIGReg diagnostic on a fixed batch of online embeddings\n"
    "            diag_x  = jnp.array(np.array(X_tr[:BATCH_SIZE]))\n"
    "            diag_e  = encode(online_params, diag_x)\n"
    "            sreg    = float(sigreg_diag(diag_e, jax.random.PRNGKey(epoch)))\n"
    "            history[\"sigreg_diag\"].append((epoch, sreg))\n"
    "            probe_str = f\"  probe_acc={acc:.1%}  sigreg_diag={sreg:.3f}\""
)

# 5. Add sigreg_diag_curve to result logging
src = src.replace(
    '        "loss_curve"      : [round(v, 6) for v in history["loss"]],',
    '        "loss_curve"      : [round(v, 6) for v in history["loss"]],\n'
    '        "sigreg_diag_curve": history["sigreg_diag"],'
)

# ── Generate latent-size variants ─────────────────────────────────────────────

configs = [
    ("exp_ema_512",  512,  "DEC_RANK² = 1024 → projected to LATENT_DIM=512"),
    ("exp_ema_1024", 1024, "DEC_RANK² = 1024 = LATENT_DIM"),
    ("exp_ema_2048", 2048, "DEC_RANK² = 1024 → projected to LATENT_DIM=2048"),
]

for exp_name, latent, rank_comment in configs:
    text = src
    text = text.replace(
        "exp_ema — LeWM JEPA on MNIST: EMA target encoder baseline, LATENT_DIM=4096.",
        f"{exp_name} — LeWM JEPA on MNIST: EMA target encoder baseline, LATENT_DIM={latent}."
    )
    text = text.replace(
        "    uv run python projects/ssl/exp_ema.py",
        f"    uv run python projects/ssl/{exp_name}.py"
    )
    text = text.replace('EXP_NAME = "exp_ema"', f'EXP_NAME = "{exp_name}"')
    text = text.replace("LATENT_DIM   = 4096", f"LATENT_DIM   = {latent}")
    text = text.replace(
        "DEC_RANK     = 32    # DEC_RANK² = 1024 → projected to LATENT_DIM=4096",
        f"DEC_RANK     = 32    # {rank_comment}"
    )
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")

# Also patch exp_ema.py itself to add sigreg_diag (for the 4096 result)
(SSL / "exp_ema_diag.py").write_text(
    src.replace('EXP_NAME = "exp_ema"', 'EXP_NAME = "exp_ema"')
)
print("wrote exp_ema_diag.py  (patched exp_ema with sigreg diagnostic — run to get diag for 4096)")
