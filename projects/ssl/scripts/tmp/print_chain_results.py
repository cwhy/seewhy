import json
rows = [json.loads(l) for l in open("projects/ssl/results_distill_chain_same_arch.jsonl")]
ARCHS = ["mlp2", "vit_patch", "gram_dual", "gram_single", "gram_glu"]
LABELS = {"mlp2":"MLP-2","vit_patch":"ViT-patch","gram_dual":"Gram-dual",
          "gram_single":"Gram-single","gram_glu":"Gram-GLU"}
data = {a: {} for a in ARCHS}
for r in rows:
    if r["arch"] in ARCHS:
        data[r["arch"]][r["hop"]] = (r["probe_acc"]*100, r["knn_1nn_acc"]*100)

print(f"{'Arch':<14}", end="")
for h in range(11): print(f"  hop{h:2d}", end="")
print()
print("-"*80)
for a in ARCHS:
    print(f"{LABELS[a]:<14}", end="")
    for h in range(11):
        if h in data[a]:
            print(f"  {data[a][h][0]:5.1f}", end="")
        else:
            print(f"  {'':5}", end="")
    print()
