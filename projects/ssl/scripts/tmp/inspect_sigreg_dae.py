"""Inspect sigreg and dae param keys to understand encoder architectures."""
import pickle, json
from pathlib import Path

SSL = Path("projects/ssl")

print("=== SIGREG param keys ===")
for arch in ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch", "gram_glu", "gram_single"]:
    p = SSL / f"params_exp_sigreg_{arch}.pkl"
    if p.exists():
        keys = list(pickle.load(open(p, "rb")).keys())
        print(f"  {arch}: {keys}")

print()
print("=== DAE param keys ===")
for arch in ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch", "gram_dual", "gram_glu", "gram_single"]:
    p = SSL / f"params_exp_dae_{arch}.pkl"
    if p.exists():
        keys = list(pickle.load(open(p, "rb")).keys())
        print(f"  {arch}: {keys}")

print()
print("=== Sigreg probe accs from results.jsonl ===")
rows = [json.loads(l) for l in open(SSL / "results.jsonl")]
for r in rows:
    if "sigreg" in r["experiment"]:
        print(f"  {r['experiment']:40} probe={r['final_probe_acc']:.1%}  latent={r.get('latent_dim','?')}")

print()
print("=== DAE probe accs ===")
for r in rows:
    if "exp_dae" in r["experiment"]:
        print(f"  {r['experiment']:40} probe={r['final_probe_acc']:.1%}  latent={r.get('latent_dim','?')}")
