"""
mk_exp_vae_kl_all.py — Generate VAE β-sweep experiments for all 9 architectures.

Patches each base exp_vae_*.py file:
  - Changes BETA, EXP_NAME, JSONL
  - Replaces __main__ to eval probe on MNIST + F-MNIST (train probe on each separately)
  - Saves to results_vae_kl.jsonl with schema:
    {experiment, arch, beta, probe_acc_mnist, probe_acc_fmnist, time_s}

MLP-shallow is skipped (exp_vae_kl_b*.py already exist with results).

Usage:
    uv run python scripts/tmp/mk_exp_vae_kl_all.py
"""
import re
from pathlib import Path

SSL = Path(__file__).parent.parent.parent / "projects" / "ssl"

ARCHS = [
    # (base_file, arch_short, arch_name, skip_if_done)
    ("exp_vae_mlp.py",        "mlp",        "MLP-shallow",  True),   # already done
    ("exp_vae_mlp2.py",       "mlp2",       "MLP-deep",     False),
    ("exp_vae_conv.py",       "conv",       "ConvNet-v1",   False),
    ("exp_vae_conv2.py",      "conv2",      "ConvNet-v2",   False),
    ("exp_vae_vit.py",        "vit",        "ViT",          False),
    ("exp_vae_vit_patch.py",  "vit_patch",  "ViT-patch",    False),
    ("exp_vae_gram_dual.py",  "gram_dual",  "Gram-dual",    False),
    ("exp_vae_gram_single.py","gram_single","Gram-single",  False),
    ("exp_vae_gram_glu.py",   "gram_glu",  "Gram-GLU",     False),
]

BETAS = [
    (0.0,    "b0"),
    (0.0001, "b1e4"),
    (0.001,  "b1e3"),
    (0.01,   "b1e2"),
    (0.1,    "b1e1"),
    (1.0,    "b1"),
]

MAIN_TEMPLATE = '''\
if __name__ == "__main__":
    logging.info(f"JAX devices: {{jax.devices()}}")
    logging.info(f"BETA = {{BETA}}  arch = {arch_name!r}")

    # Skip if already done
    done_exps = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_exps.add(json.loads(line).get("experiment"))
                    except Exception:
                        pass
    if EXP_NAME in done_exps:
        logging.info(f"{{EXP_NAME}} already in results_vae_kl.jsonl — skipping")
        exit(0)

    mnist   = load_supervised_image("mnist")
    fmnist  = load_supervised_image("fashion_mnist")
    X_tr    = mnist.X.reshape(mnist.n_samples,           784).astype(np.float32)
    X_te    = mnist.X_test.reshape(mnist.n_test_samples, 784).astype(np.float32)
    Y_tr    = np.array(mnist.y)
    Y_te    = np.array(mnist.y_test)
    X_tr_f  = fmnist.X.reshape(fmnist.n_samples,           784).astype(np.float32)
    X_te_f  = fmnist.X_test.reshape(fmnist.n_test_samples, 784).astype(np.float32)
    Y_tr_f  = np.array(fmnist.y)
    Y_te_f  = np.array(fmnist.y_test)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {{n_params(params):,}}")

    params, history, elapsed = train(params, X_tr, Y_tr, X_te, Y_te)

    with open(Path(__file__).parent / f"params_{{EXP_NAME}}.pkl", "wb") as f:
        pickle.dump({{k: np.array(v) for k, v in params.items()}}, f)

    probe_mnist  = linear_probe_accuracy(params, X_tr,   Y_tr,   X_te,   Y_te)
    probe_fmnist = linear_probe_accuracy(params, X_tr_f, Y_tr_f, X_te_f, Y_te_f)

    append_result({{
        "experiment"       : EXP_NAME,
        "arch"             : {arch_name!r},
        "beta"             : BETA,
        "probe_acc_mnist"  : round(probe_mnist,  6),
        "probe_acc_fmnist" : round(probe_fmnist, 6),
        "time_s"           : round(elapsed, 1),
    }})
    logging.info(f"Done. probe_mnist={{probe_mnist:.1%}}  probe_fmnist={{probe_fmnist:.1%}}  {{elapsed:.1f}}s")
'''

generated = []

for base_file, arch_short, arch_name, skip in ARCHS:
    if skip:
        # MLP-shallow: just update existing results to add "arch" field if missing
        print(f"Skipping {arch_name} (already done)")
        continue

    base_text = (SSL / base_file).read_text()

    # Find the line number of __main__ to cut there
    main_idx = base_text.index("\nif __name__ == \"__main__\":")
    body = base_text[:main_idx]

    for beta, beta_suffix in BETAS:
        exp_name = f"exp_vae_kl_{arch_short}_{beta_suffix}"

        # Patch EXP_NAME, BETA, JSONL
        patched = body
        patched = re.sub(
            r'^EXP_NAME = "[^"]*"',
            f'EXP_NAME = "{exp_name}"',
            patched, flags=re.MULTILINE
        )
        patched = re.sub(
            r'^BETA = \S+',
            f'BETA = {repr(beta)}',
            patched, flags=re.MULTILINE
        )
        patched = re.sub(
            r'JSONL = Path\(__file__\)\.parent / "results\.jsonl"',
            'JSONL = Path(__file__).parent / "results_vae_kl.jsonl"',
            patched
        )
        # Fix docstring
        patched = re.sub(
            r'BETA=0\.001',
            f'BETA={beta}',
            patched, count=1
        )

        # Append new __main__
        new_main = MAIN_TEMPLATE.format(arch_name=arch_name)
        full = patched + "\n" + new_main

        out_path = SSL / f"{exp_name}.py"
        out_path.write_text(full)
        generated.append((exp_name, arch_name, beta))

print(f"\nGenerated {len(generated)} experiment files:")
for exp, arch, beta in generated:
    print(f"  {exp}  ({arch}, beta={beta})")
