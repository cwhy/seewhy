"""
Evaluate linear probe + 1-NN for all training methods × architectures.
Includes random baseline. Saves to results_knn_comparison.jsonl.

Schema per row:
  {"encoder": "MLP-shallow", "training": "ae", "dataset": "mnist",
   "knn_l2_k1": 0.87, "probe_acc": 0.92}
"""
import importlib.util, json, pickle, sys, time
from pathlib import Path

import numpy as np, jax, jax.numpy as jnp, faiss, optax, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.random_utils import infinite_safe_keys_from_key

SSL  = Path(__file__).parent.parent.parent / "projects" / "ssl"
OUT  = SSL / "results_knn_comparison.jsonl"

BATCH_SIZE   = 256
PROBE_EPOCHS = 10

ARCHS = [
    dict(name="MLP-shallow",
         ema_file="exp_mlp_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_mlp.py",           "exp_ae_mlp"),
             dae   =("exp_dae_mlp.py",          "exp_dae_mlp"),
             vae   =("exp_vae_mlp.py",          "exp_vae_mlp"),
             sigreg=("exp_sigreg_mlp.py",        "exp_sigreg_mlp"),
             ema   =("exp_mlp_ema_1024.py",      "exp_mlp_ema_1024"),
         )),
    dict(name="MLP-deep",
         ema_file="exp_mlp2_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_mlp2.py",           "exp_ae_mlp2"),
             dae   =("exp_dae_mlp2.py",          "exp_dae_mlp2"),
             vae   =("exp_vae_mlp2.py",          "exp_vae_mlp2"),
             sigreg=("exp_sigreg_mlp2.py",        "exp_sigreg_mlp2"),
             ema   =("exp_mlp2_ema_1024.py",      "exp_mlp2_ema_1024"),
         )),
    dict(name="ConvNet-v1",
         ema_file="exp_conv_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_conv.py",            "exp_ae_conv"),
             dae   =("exp_dae_conv.py",           "exp_dae_conv"),
             vae   =("exp_vae_conv.py",           "exp_vae_conv"),
             sigreg=("exp_sigreg_conv.py",         "exp_sigreg_conv"),
             ema   =("exp_conv_ema_1024.py",       "exp_conv_ema_1024"),
         )),
    dict(name="ConvNet-v2",
         ema_file="exp_conv2_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_conv2.py",           "exp_ae_conv2"),
             dae   =("exp_dae_conv2.py",          "exp_dae_conv2"),
             vae   =("exp_vae_conv2.py",          "exp_vae_conv2"),
             sigreg=("exp_sigreg_conv2.py",        "exp_sigreg_conv2"),
             ema   =("exp_conv2_ema_1024.py",      "exp_conv2_ema_1024"),
         )),
    dict(name="ViT",
         ema_file="exp_vit_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_vit.py",             "exp_ae_vit"),
             dae   =("exp_dae_vit.py",            "exp_dae_vit"),
             vae   =("exp_vae_vit.py",            "exp_vae_vit"),
             sigreg=("exp_sigreg_vit.py",          "exp_sigreg_vit"),
             ema   =("exp_vit_ema_1024.py",        "exp_vit_ema_1024"),
         )),
    dict(name="ViT-patch",
         ema_file="exp_vit_patch_ema.py",
         methods=dict(
             ae    =("exp_ae_vit_patch.py",       "exp_ae_vit_patch"),
             dae   =("exp_dae_vit_patch.py",      "exp_dae_vit_patch"),
             vae   =("exp_vae_vit_patch.py",      "exp_vae_vit_patch"),
             sigreg=("exp_sigreg_vit_patch.py",    "exp_sigreg_vit_patch"),
             ema   =("exp_vit_patch_ema.py",       "exp_vit_patch_ema"),
         )),
    dict(name="Gram-dual",
         ema_file="exp_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_gram_dual.py",       "exp_ae_gram_dual"),
             dae   =("exp_dae_gram_dual.py",      "exp_dae_gram_dual"),
             vae   =("exp_vae_gram_dual.py",      "exp_vae_gram_dual"),
             sigreg=("exp12a.py",                  "exp12a"),
             ema   =("exp_ema_1024.py",            "exp_ema_1024"),
         )),
    dict(name="Gram-single",
         ema_file="exp_gram_single_ema_1024.py",
         methods=dict(
             ae    =("exp_ae_gram_single.py",     "exp_ae_gram_single"),
             dae   =("exp_dae_gram_single.py",    "exp_dae_gram_single"),
             vae   =("exp_vae_gram_single.py",    "exp_vae_gram_single"),
             sigreg=("exp_sigreg_gram_single.py",  "exp_sigreg_gram_single"),
             ema   =("exp_gram_single_ema_1024.py","exp_gram_single_ema_1024"),
         )),
    dict(name="Gram-GLU",
         ema_file="exp_gram_glu_ema.py",
         methods=dict(
             ae    =("exp_ae_gram_glu.py",        "exp_ae_gram_glu"),
             dae   =("exp_dae_gram_glu.py",       "exp_dae_gram_glu"),
             vae   =("exp_vae_gram_glu.py",       "exp_vae_gram_glu"),
             sigreg=("exp_sigreg_gram_glu.py",     "exp_sigreg_gram_glu"),
             ema   =("exp_gram_glu_ema.py",        "exp_gram_glu_ema"),
         )),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_module(fname):
    spec = importlib.util.spec_from_file_location("_m", SSL / fname)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def load_params(exp_name):
    p = pickle.load(open(SSL / f"params_{exp_name}.pkl", "rb"))
    return {k: jnp.array(v) for k, v in p.items()}

def extract_emb(encode_fn, params, X):
    parts = []
    for i in range(0, len(X), 512):
        parts.append(np.array(encode_fn(params, jnp.array(np.array(X[i:i+512])))))
    return np.concatenate(parts, 0).astype(np.float32)

def knn1(E_tr, Y_tr, E_te, Y_te):
    idx = faiss.IndexFlatL2(E_tr.shape[1])
    idx.add(E_tr)
    _, I = idx.search(E_te, 1)
    return float((Y_tr[I[:, 0]] == Y_te).mean())

def linear_probe(E_tr, Y_tr, E_te, Y_te):
    LATENT = E_tr.shape[1]
    n_tr   = (len(E_tr) // BATCH_SIZE) * BATCH_SIZE
    E_b    = jnp.array(E_tr[:n_tr].reshape(-1, BATCH_SIZE, LATENT))
    Y_b    = jnp.array(Y_tr[:n_tr].reshape(-1, BATCH_SIZE))
    W, b   = jnp.zeros((LATENT, 10)), jnp.zeros(10)
    opt    = optax.adam(1e-3)
    state  = opt.init((W, b))

    def step(carry, inputs):
        (W, b), s = carry
        e, y = inputs
        loss, g = jax.value_and_grad(
            lambda wb: optax.softmax_cross_entropy_with_integer_labels(
                e @ wb[0] + wb[1], y).mean()
        )((W, b))
        upd, s = opt.update(g, s, (W, b))
        return (optax.apply_updates((W, b), upd), s), loss

    @jax.jit
    def epoch(W, b, s, E_b, Y_b):
        (W, b), s = jax.lax.scan(step, ((W, b), s), (E_b, Y_b))[0]
        return W, b, s

    for _ in range(PROBE_EPOCHS):
        W, b, state = epoch(W, b, state, E_b, Y_b)

    logits = jnp.array(E_te) @ W + b
    return float((jnp.argmax(logits, 1) == jnp.array(Y_te)).mean())

def append(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")

# ── Load done set ──────────────────────────────────────────────────────────────

done = set()
if OUT.exists():
    with open(OUT) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done.add((r["encoder"], r["training"], r["dataset"]))
                except Exception:
                    pass
logging.info(f"Already done: {len(done)}")

# ── Load datasets ──────────────────────────────────────────────────────────────

logging.info("Loading datasets...")
datasets = {}
for ds in ("mnist", "fashion_mnist"):
    d = load_supervised_image(ds)
    datasets[ds] = dict(
        X_tr=d.X.reshape(d.n_samples, 784).astype(np.float32),
        Y_tr=np.array(d.y, dtype=np.int32),
        X_te=d.X_test.reshape(d.n_test_samples, 784).astype(np.float32),
        Y_te=np.array(d.y_test, dtype=np.int32),
    )

# ── Evaluate ──────────────────────────────────────────────────────────────────

for arch in ARCHS:
    name     = arch["name"]
    ema_mod  = load_module(arch["ema_file"])

    # Random baseline
    key_gen  = infinite_safe_keys_from_key(jax.random.PRNGKey(0))
    rand_p   = ema_mod.init_params(key_gen)
    encode_r = ema_mod.encode

    for ds_key, ds in datasets.items():
        key = (name, "random", ds_key)
        if key not in done:
            logging.info(f"  {name} / random / {ds_key}")
            E_tr = extract_emb(encode_r, rand_p, ds["X_tr"])
            E_te = extract_emb(encode_r, rand_p, ds["X_te"])
            append({"encoder": name, "training": "random", "dataset": ds_key,
                    "knn_l2_k1": round(knn1(E_tr, ds["Y_tr"], E_te, ds["Y_te"]), 6),
                    "probe_acc": round(linear_probe(E_tr, ds["Y_tr"], E_te, ds["Y_te"]), 6)})
            done.add(key)

    # Trained methods
    for training, (fname, exp_name) in arch["methods"].items():
        try:
            mod    = load_module(fname)
            params = load_params(exp_name)
        except Exception as e:
            logging.warning(f"  {name}/{training}: skip — {e}")
            continue

        for ds_key, ds in datasets.items():
            key = (name, training, ds_key)
            if key in done:
                logging.info(f"  {name} / {training} / {ds_key} — skip")
                continue
            logging.info(f"  {name} / {training} / {ds_key}")
            t0   = time.perf_counter()
            E_tr = extract_emb(mod.encode, params, ds["X_tr"])
            E_te = extract_emb(mod.encode, params, ds["X_te"])
            append({"encoder": name, "training": training, "dataset": ds_key,
                    "knn_l2_k1": round(knn1(E_tr, ds["Y_tr"], E_te, ds["Y_te"]), 6),
                    "probe_acc": round(linear_probe(E_tr, ds["Y_tr"], E_te, ds["Y_te"]), 6),
                    "time_s": round(time.perf_counter()-t0, 1)})
            done.add(key)

logging.info("Done.")
