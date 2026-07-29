"""
The retrieval circuit as block-matrix geometry: every matrix drawn to scale with its
contraction dimensions, and the learned matrices rendered as real heatmaps.

Usage (server): uv run python projects/universal-ar/scripts/gen_diagram_blocks.py
"""
import io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_media
from shared_lib.report import save_report

ROOT = Path(__file__).parent.parent
S = json.load(open(ROOT / "circuit.json"))
MM = json.load(open(ROOT / "circuit_mats.json"))
best = max((h for l in S["layers"] for h in l["heads"]), key=lambda z: z["copy_top1"])
bl = MM["layer"]

C = dict(emb="#bcd7f0", emb_s="#5a8fc0", wt="#e3d7f2", wt_s="#8f79c8",
         out="#c6e6c9", out_s="#5aa564", txt="#16202b", mut="#6b7785",
         ok="#1e8449", bad="#c0392b", box_s="#c3ccd6", hi="#fff4cc")
P = []
def T(x, y, s, sz=11.5, col=None, w="normal", anc="start", mono=False, rot=None):
    ff = ' font-family="ui-monospace,Menlo,monospace"' if mono else ""
    tr = f' transform="rotate({rot},{x},{y})"' if rot else ""
    P.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col or C["txt"]}" font-weight="{w}" text-anchor="{anc}"{ff}{tr}>{s}</text>')
def rect(x, y, w, h, fill, stroke, sw=1.2, rx=2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

SC = 0.32                       # px per unit dimension
DIM = {256: "d_model", 32: "d_head", 43: "n_val", 42: "n_out",
       785: "n_pos", 64: "n_ref", 1: "1"}
def dname(n):  return DIM.get(n, str(n))
def dlab(n):   return f"{dname(n)}={n}" if n in DIM and n != 1 else str(n)
def dim(n):    return max(16, n * SC)

def mat(x, cy, rows, cols, label, fill, stroke):
    """Matrix block, vertically CENTRED on cy. Returns (x_right, w, h)."""
    w, h = dim(cols), dim(rows)
    y = cy - h/2
    rect(x, y, w, h, fill, stroke, 1.4)
    T(x + w/2, cy + 4, label, 11.5, w="bold", anc="middle", mono=True)
    T(x + w/2, y - 8, dlab(cols), 9.5, C["mut"], anc="middle", mono=True)   # cols above
    T(x - 7, cy + 3.5, dlab(rows), 9.5, C["mut"], anc="end", mono=True)     # rows at left
    return x + w, w, h

def chain(x, cy, blocks, result=None, gap=30):
    """Lay out A · B · C [= R] on one centred axis; annotate contractions."""
    xs = x; spans = []
    for i, (r, c, nm, f_, s_) in enumerate(blocks):
        xe, w, h = mat(xs, cy, r, c, nm, f_, s_)
        spans.append((xs, xe, h))
        if i < len(blocks) - 1:
            T(xe + gap/2, cy + 5, "·", 17, C["mut"], "bold", "middle")
            # contraction annotation under the join
            T(xe + gap/2, cy + max(h, dim(blocks[i+1][0]))/2 + 20, dname(c),
              9, C["ok"], anc="middle", mono=True)
            xs = xe + gap
        else:
            xs = xe
    if result:
        T(xs + gap/2, cy + 5, "=", 17, C["mut"], "bold", "middle")
        r, c, nm, f_, s_ = result
        xs, w, h = mat(xs + gap, cy, r, c, nm, f_, s_)
    return xs

def heat(x, y, A, cell, title, note=""):
    """Render a real matrix as a heatmap; returns (w, h)."""
    n, m = len(A), len(A[0])
    T(x, y - 8, title, 10.5, C["mut"], mono=True)
    for i in range(n):
        for j in range(m):
            v = max(-1.0, min(1.0, A[i][j]))
            if v >= 0:
                r, g, b = int(255 - v*225), int(255 - v*123), int(255 - v*182)
            else:
                r, g, b = int(255 + v*60), int(255 + v*110), int(255 + v*20)
            P.append(f'<rect x="{x+j*cell:.1f}" y="{y+i*cell:.1f}" width="{cell}" height="{cell}" fill="rgb({r},{g},{b})"/>')
    rect(x, y, m*cell, n*cell, "none", C["box_s"], 1)
    if note: T(x, y + n*cell + 15, note, 10, C["mut"])
    return m*cell, n*cell

W, H = 1400, 1180
P.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">')
P.append(f'<rect width="{W}" height="{H}" fill="white"/>')
P.append('<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6.5" refY="2.8" orient="auto">'
         f'<path d="M0,0 L6.5,2.8 L0,5.6 Z" fill="{C["mut"]}"/></marker></defs>')
T(28, 34, "The retrieval circuit as block matrices", 20, w="bold")
T(28, 56, f"blocks drawn to scale · heatmaps are the trained weights (layer {bl}, head {best['head']}) · "
          f"retrieval label {S['retr_lab']:.3f}, pixel {S['retr_pix']:.3f}", 11.5, C["mut"])

# ══ 1 · SELECT ══
y = 80
rect(20, y, W-40, 470, "#f8fbf9", C["ok"], 1.6, 8)
T(38, y+26, "1 · SELECT   —   score  =  x_q\u1d40 · M · x_j", 14, C["ok"], "bold")
T(38, y+46, "one scalar per candidate token", 11, C["mut"])

cy1 = y + 118
chain(96, cy1, [(1, 256, "x_q\u1d40", C["emb"], C["emb_s"]),
                (256, 256, "M", C["wt"], C["wt_s"]),
                (256, 1, "x_j", C["emb"], C["emb_s"])],
      result=(1, 1, "s", C["out"], C["out_s"]))
T(96, cy1 + 76, "the two d_model contractions cancel \u2192 one number per candidate", 10.5, C["mut"])

cy2 = y + 250
T(38, cy2 - 62, "M is low-rank \u2014 the head has only d_head dimensions to work with:", 12, w="bold")
chain(96, cy2, [(256, 32, "Wq", C["wt"], C["wt_s"]),
                (32, 256, "Wk\u1d40", C["wt"], C["wt_s"])],
      result=(256, 256, "M", C["wt"], C["wt_s"]))
T(96, cy2 + 76, "the d_head=32 waist is the bottleneck the whole address match must pass through", 10.5, C["mut"])

# the three Gram matrices — real heatmaps
gx = 640
T(gx, y + 62, "x is a SUM of three embeddings, so the score expands into Gram matrices  EᵀME.", 12, w="bold")
T(gx, y + 80, "These are the trained ones — a bright diagonal means “matches itself, not others”:", 11.5, C["mut"])
hy = y + 100
gw, _ = heat(gx, hy, MM["gram_ref"], 4.6, "E_ref M E_refᵀ   (64 refs, first 48)",
             f"ref:  {best['ref_dd']:+.2f} σ   ← strongest")
gw2, _ = heat(gx + gw + 46, hy, MM["gram_pos"], 4.6, "E_pos M E_posᵀ   (785 positions, 48 sampled)",
              f"pos:  {best['pos_dd']:+.2f} σ")
gw3, _ = heat(gx + gw + gw2 + 92, hy, MM["gram_val"], 4.6, "E_val M E_valᵀ   (42 values)",
              f"value: {best['val_dd']:+.2f} σ   ← weak, but not 0")
T(gx, hy + 250, "ADDRESS fields (ref, pos) are diagonal-dominant → the probe selects on address.", 11.5, C["ok"], "bold")
T(gx, hy + 268, f"CONTENT is {best['ref_dd']/best['val_dd']:.1f}× weaker but present — it is not fully suppressed.", 11.5, C["mut"])
T(gx, hy + 292, "⟹ softmax over 6810 keys collapses α onto the one token matching ref AND pos.", 11.5, C["ok"], "bold")

# ══ 2 · TRANSPORT + DECODE ══
y = 570
rect(20, y, W-40, 470, "#faf8fd", C["wt_s"], 1.6, 8)
T(38, y+24, "2 · TRANSPORT + DECODE   —   the copy matrix  C  =  E_val · Wv · Wo · head_W", 14, C["wt_s"], "bold")
T(38, y+46, "chain the four learned matrices; the inner dimensions contract away and what remains maps value → value:", 11.5, C["mut"])

cy = y + 130
chain(96, cy, [(43, 256, "E_val", C["emb"], C["emb_s"]),
               (256, 32, "Wv", C["wt"], C["wt_s"]),
               (32, 256, "Wo", C["wt"], C["wt_s"]),
               (256, 42, "head_W", C["wt"], C["wt_s"])],
      result=(43, 42, "C", C["out"], C["out_s"]))
T(96, cy + 92, "n_val \u2192 d_model \u2192 d_head \u2192 d_model \u2192 n_out : everything contracts except the map from", 11, C["mut"])
T(96, cy + 110, "the value WRITTEN to the value READ. C is what the network can actually read back out.", 11, C["mut"])

# real copy heatmap
kx = 760
kw, kh = heat(kx, cy - 14, MM["copy"], 7.0, "the trained C  (42 × 42)  — rows: value written, cols: value read")
T(kx, cy + kh + 8, "a bright diagonal = write value a, read value a", 11, C["ok"], "bold")
rect(kx + kw + 40, cy - 14, 300, 150, "white", C["ok"], 1.8, 7)
T(kx + kw + 56, cy + 10, "MEASURED", 12, C["ok"], "bold")
T(kx + kw + 56, cy + 34, f"top-1 on diagonal: {best['copy_top1']*100:.0f}%", 13.5, w="bold", mono=True)
T(kx + kw + 56, cy + 54, f"= {best['copy_top1']*42:.0f}× the 1/42 chance rate", 10.5, C["mut"])
T(kx + kw + 56, cy + 78, "through ONE head's OV path alone,", 10.5, C["mut"])
T(kx + kw + 56, cy + 94, "ignoring residual, LN and 3 MLPs.", 10.5, C["mut"])
T(kx + kw + 56, cy + 118, f"end-to-end retrieval: {S['retr_lab']:.3f}", 11.5, C["ok"], "bold")

# ══ 3 · the failure, same geometry ══
y = 1060
rect(20, y, W-40, 96, "#fdf7f7", C["bad"], 1.6, 8)
T(38, y+26, "3 · WHY THE SAME BLOCKS CANNOT GENERALISE", 13.5, C["bad"], "bold")
T(38, y+50, f"The ref Gram is the strongest term ({best['ref_dd']:+.2f} σ). For a generalisation query no context token carries that ref, so that", 11.5)
T(38, y+70, "matrix contributes nothing for every candidate; all support labels tie on pos alone → α uniform → C averages every label → ln 2 = 0.6931.", 11.5)
P.append("</svg>")

svg = "".join(P)
url = save_media("universal-ar_circuit_blocks.svg", io.BytesIO(svg.encode()), "image/svg+xml")
md = f"""# The retrieval circuit as block matrices

![blocks]({url})

The same circuit as before, but with the algebra drawn instead of written: every
matrix is a block scaled to its true dimensions, contraction dims are annotated where
they cancel, and the learned matrices are rendered as **real heatmaps of the trained
weights** (layer {bl}, head {best['head']}).

## 1 · SELECT

```
score  =  x_qᵀ · M · x_j          (1×256)(256×256)(256×1) → scalar
M      =  Wq · Wkᵀ                (256×32)(32×256)
```

The 32-d waist is the bottleneck the entire address match must pass through.

Because `x` is a **sum** of three embeddings, the score expands into Gram matrices
`EᵀME`, drawn from the trained weights:

| Gram matrix | diagonal dominance | reading |
|---|---|---|
| `E_ref M E_refᵀ` | **{best['ref_dd']:+.2f} σ** | strongest — ref identifies the sample |
| `E_pos M E_posᵀ` | **{best['pos_dd']:+.2f} σ** | position identifies the slot |
| `E_val M E_valᵀ` | {best['val_dd']:+.2f} σ | content — {best['ref_dd']/best['val_dd']:.1f}× weaker, **not** suppressed |

A bright diagonal means "matches itself, not others". Both address Grams have one;
the value Gram is comparatively flat. Softmax over 6810 keys then collapses α onto
the single token matching **ref and pos**.

## 2 · TRANSPORT + DECODE

```
C  =  E_val · Wv · Wo · head_W
      (43×256)(256×32)(32×256)(256×42)  →  (43×42)
```

43 → 256 → 32 → 256 → 42: everything contracts except the map from *value written*
to *value read*. **C is the circuit's output half**, and the trained one is shown as
a heatmap: **{best['copy_top1']*100:.0f}% of values have themselves as top read-out,
{best['copy_top1']*42:.0f}× the 1/42 chance rate** — through one head's OV path alone,
ignoring the residual stream, LayerNorm and three MLP blocks. End-to-end retrieval is
{S['retr_lab']:.3f}, so this head builds the copy and later blocks sharpen it.

## 3 · The failure, in the same geometry

The ref Gram is the strongest term. For a generalisation query **no context token
carries that ref**, so that matrix contributes nothing for any candidate; all support
label tokens tie on the pos term alone, α goes uniform, and C averages every label —
cross-entropy pinned at ln 2 = 0.6931.
"""
rep = save_report("universal-ar_circuit_blocks", md)
print("DIAGRAM:", url); print("REPORT:", rep)
