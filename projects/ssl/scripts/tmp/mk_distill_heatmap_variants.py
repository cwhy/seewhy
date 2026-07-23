"""Generate exp_distill_heatmap_sigreg/dae/random.py from the EMA version."""
from pathlib import Path

SSL    = Path(__file__).parent.parent.parent
SRC    = SSL / "exp_distill_heatmap.py"
source = SRC.read_text()

# ── Sigreg variant ────────────────────────────────────────────────────────────

SIGREG_PKLS = '''\
TEACHER_PKLS = {
    "mlp"        : "params_exp_sigreg_mlp.pkl",
    "mlp2"       : "params_exp_sigreg_mlp2.pkl",
    "conv"       : "params_exp_sigreg_conv.pkl",
    "conv2"      : "params_exp_sigreg_conv2.pkl",
    "vit"        : "params_exp_sigreg_vit.pkl",
    "vit_patch"  : "params_exp_sigreg_vit_patch.pkl",
    "gram_single": "params_exp_sigreg_gram_single.pkl",
    "gram_glu"   : "params_exp_sigreg_gram_glu.pkl",
}

TEACHER_ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch", "gram_single", "gram_glu"]'''

EMA_PKLS = '''\
TEACHER_PKLS = {
    "mlp"        : "params_exp_mlp_ema_1024.pkl",
    "mlp2"       : "params_exp_mlp2_ema_1024.pkl",
    "conv"       : "params_exp_conv_ema_1024.pkl",
    "conv2"      : "params_exp_conv2_ema_1024.pkl",
    "vit"        : "params_exp_vit_ema_1024.pkl",
    "vit_patch"  : "params_exp_vit_patch_ema.pkl",
    "gram_dual"  : "params_exp_ema_1024.pkl",
    "gram_single": "params_exp_gram_single_ema_1024.pkl",
    "gram_glu"   : "params_exp_gram_glu_ema.pkl",
}'''

# Sigreg: replace TEACHER_PKLS, add TEACHER_ARCHS, patch jsonl and loop
sigreg = source
sigreg = sigreg.replace(
    'exp_distill_heatmap — arch × arch masked distillation sweep.\n\nFor each (teacher_arch, student_arch) pair:\n  - Teacher: frozen EMA-JEPA encoder (LATENT_DIM=1024)',
    'exp_distill_heatmap_sigreg — arch × arch masked distillation sweep.\n\nFor each (teacher_arch, student_arch) pair:\n  - Teacher: frozen SigReg encoder (LATENT_DIM=1024)',
)
sigreg = sigreg.replace(
    'Results go to results_distill_heatmap.jsonl.',
    'Results go to results_distill_heatmap_sigreg.jsonl.',
)
sigreg = sigreg.replace(
    'uv run python projects/ssl/exp_distill_heatmap.py',
    'uv run python projects/ssl/exp_distill_heatmap_sigreg.py',
)
sigreg = sigreg.replace(
    'JSONL       = SSL / "results_distill_heatmap.jsonl"',
    'JSONL       = SSL / "results_distill_heatmap_sigreg.jsonl"',
)
sigreg = sigreg.replace(EMA_PKLS, SIGREG_PKLS)
sigreg = sigreg.replace(
    'for teacher_arch in ARCHS:',
    'for teacher_arch in TEACHER_ARCHS:',
)
sigreg = sigreg.replace(
    'logging.info(f"Already done: {len(done)}/81 pairs")',
    'logging.info(f"Already done: {len(done)}/72 pairs")',
)

# Add TEACHER_ARCHS = ARCHS fallback for EMA (so report scripts still work)
# For sigreg, TEACHER_ARCHS is already defined in SIGREG_PKLS block

out = SSL / "exp_distill_heatmap_sigreg.py"
out.write_text(sigreg)
print(f"Written: {out.name}")

# ── DAE variant ───────────────────────────────────────────────────────────────

DAE_PKLS = '''\
TEACHER_PKLS = {
    "mlp"        : "params_exp_dae_mlp.pkl",
    "mlp2"       : "params_exp_dae_mlp2.pkl",
    "conv"       : "params_exp_dae_conv.pkl",
    "conv2"      : "params_exp_dae_conv2.pkl",
    "vit"        : "params_exp_dae_vit.pkl",
    "vit_patch"  : "params_exp_dae_vit_patch.pkl",
    "gram_dual"  : "params_exp_dae_gram_dual.pkl",
    "gram_single": "params_exp_dae_gram_single.pkl",
    "gram_glu"   : "params_exp_dae_gram_glu.pkl",
}

TEACHER_ARCHS = ARCHS'''

dae = source
dae = dae.replace(
    'exp_distill_heatmap — arch × arch masked distillation sweep.\n\nFor each (teacher_arch, student_arch) pair:\n  - Teacher: frozen EMA-JEPA encoder (LATENT_DIM=1024)',
    'exp_distill_heatmap_dae — arch × arch masked distillation sweep.\n\nFor each (teacher_arch, student_arch) pair:\n  - Teacher: frozen DAE encoder (LATENT_DIM=1024)',
)
dae = dae.replace(
    'Results go to results_distill_heatmap.jsonl.',
    'Results go to results_distill_heatmap_dae.jsonl.',
)
dae = dae.replace(
    'uv run python projects/ssl/exp_distill_heatmap.py',
    'uv run python projects/ssl/exp_distill_heatmap_dae.py',
)
dae = dae.replace(
    'JSONL       = SSL / "results_distill_heatmap.jsonl"',
    'JSONL       = SSL / "results_distill_heatmap_dae.jsonl"',
)
dae = dae.replace(EMA_PKLS, DAE_PKLS)
dae = dae.replace(
    'for teacher_arch in ARCHS:',
    'for teacher_arch in TEACHER_ARCHS:',
)

out = SSL / "exp_distill_heatmap_dae.py"
out.write_text(dae)
print(f"Written: {out.name}")

# ── Random variant ────────────────────────────────────────────────────────────

RANDOM_PKLS = '''\
TEACHER_PKLS = {}   # unused for random teachers

TEACHER_ARCHS = ARCHS'''

# Replace the pkl-based teacher loading in __main__ with random init
OLD_LOADER = '''\
    # Main loop — one teacher at a time (reload teacher params once per teacher)
    for teacher_arch in ARCHS:
        pkl = PARAMS_DIR / TEACHER_PKLS[teacher_arch]
        if not pkl.exists():
            logging.warning(f"Teacher params not found: {pkl.name} — skipping")
            continue
        teacher_params = {k: jnp.array(v)
                          for k, v in pickle.load(open(pkl, "rb")).items()}
        logging.info(f"Teacher {teacher_arch}: {n_params(teacher_params):,} params from {pkl.name}")'''

NEW_LOADER = '''\
    # Main loop — one teacher at a time (randomly initialised teacher)
    for i_t, teacher_arch in enumerate(TEACHER_ARCHS):
        key_gen_t  = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 200 + i_t))
        teacher_params = STUDENT_INIT_FNS[teacher_arch](key_gen_t)
        logging.info(f"Teacher {teacher_arch}: random init  {n_params(teacher_params):,} params")'''

rand = source
rand = rand.replace(
    'exp_distill_heatmap — arch × arch masked distillation sweep.\n\nFor each (teacher_arch, student_arch) pair:\n  - Teacher: frozen EMA-JEPA encoder (LATENT_DIM=1024)',
    'exp_distill_heatmap_random — arch × arch masked distillation sweep.\n\nFor each (teacher_arch, student_arch) pair:\n  - Teacher: randomly initialised encoder (no training)',
)
rand = rand.replace(
    'Results go to results_distill_heatmap.jsonl.',
    'Results go to results_distill_heatmap_random.jsonl.',
)
rand = rand.replace(
    'uv run python projects/ssl/exp_distill_heatmap.py',
    'uv run python projects/ssl/exp_distill_heatmap_random.py',
)
rand = rand.replace(
    'JSONL       = SSL / "results_distill_heatmap.jsonl"',
    'JSONL       = SSL / "results_distill_heatmap_random.jsonl"',
)
rand = rand.replace(EMA_PKLS, RANDOM_PKLS)
rand = rand.replace(OLD_LOADER, NEW_LOADER)
rand = rand.replace(
    'for teacher_arch in ARCHS:',
    'for teacher_arch in TEACHER_ARCHS:',
)

out = SSL / "exp_distill_heatmap_random.py"
out.write_text(rand)
print(f"Written: {out.name}")

# ── Syntax check all 3 ────────────────────────────────────────────────────────

import ast
for name in ["exp_distill_heatmap_sigreg.py", "exp_distill_heatmap_dae.py", "exp_distill_heatmap_random.py"]:
    p = SSL / name
    try:
        ast.parse(p.read_text())
        print(f"OK  {name}")
    except SyntaxError as e:
        print(f"ERR {name}: {e}")
