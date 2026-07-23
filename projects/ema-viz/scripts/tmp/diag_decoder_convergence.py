"""Diagnose linear decoder convergence — train for 600 epochs, log test MSE at intervals."""
import pickle, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, optax

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image

SSL_DIR = Path(__file__).parent.parent.parent.parent / "ssl"
latent_dim = 32

data = load_supervised_image("mnist")
X_tr = data.X.reshape(data.n_samples, 784).astype(np.float32)
X_te = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

with open(SSL_DIR / f"params_exp_mlp_ema_{latent_dim}.pkl", "rb") as f:
    raw = pickle.load(f)
enc = {k: jnp.array(v) for k, v in raw.items()}

@jax.jit
def encode(p, x):
    h = jax.nn.gelu(x/255.0 @ p["W_enc1"] + p["b_enc1"])
    return h @ p["W_enc2"] + p["b_enc2"]

Z_tr = np.concatenate([np.array(encode(enc, jnp.array(X_tr[i:i+512].astype(np.float32)))) for i in range(0, len(X_tr), 512)])
Z_te = np.concatenate([np.array(encode(enc, jnp.array(X_te[i:i+512].astype(np.float32)))) for i in range(0, len(X_te), 512)])

batch, epochs, lr = 512, 600, 3e-3
k1, _ = jax.random.split(jax.random.PRNGKey(0))
params = {"P": jax.random.normal(k1, (784, latent_dim)) * latent_dim**-0.5, "b": jnp.zeros(784)}
opt = optax.adam(lr)
state = opt.init(params)

n  = (len(Z_tr) // batch) * batch
Zb = jnp.array(Z_tr[:n].reshape(-1, batch, latent_dim))
Xb = jnp.array(X_tr[:n].reshape(-1, batch, 784) / 255.0)

@jax.jit
def step(params, state, z, x):
    def loss(p):
        P_pos = jnp.exp(jnp.clip(p["P"], -10, 10))
        xh = jax.nn.sigmoid(z @ P_pos.T + p["b"])
        return -(x * jnp.log(xh + 1e-7) + (1-x) * jnp.log(1-xh + 1e-7)).mean()
    l, g = jax.value_and_grad(loss)(params)
    u, s = opt.update(g, state, params)
    return optax.apply_updates(params, u), s, l

log_at = {0, 9, 19, 49, 99, 149, 199, 299, 399, 499, 599}

for ep in range(epochs):
    for bi in range(len(Zb)):
        params, state, _ = step(params, state, Zb[bi], Xb[bi])
    if ep in log_at:
        P_pos = jnp.exp(jnp.clip(params["P"], -10, 10))
        xh = np.array(jax.nn.sigmoid(jnp.array(Z_te[:1000]) @ P_pos.T + params["b"]))
        mse = float(np.mean((xh - X_te[:1000] / 255.0) ** 2))
        print(f"epoch {ep:3d}  test_mse={mse:.4f}", flush=True)
