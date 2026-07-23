"""
Generate the Cross-Completion (variant B) report for universal-ar: an
architecture diagram (token composition, full training context, and how the
label/pixels are held out for train vs test) plus results. Publishes to R2.

Usage (server): uv run python projects/universal-ar/scripts/gen_report_cross_completion.py
"""
import io, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure, save_media
from shared_lib.report import save_report

ROWS = [json.loads(l) for l in open(Path(__file__).parent.parent / "results.jsonl") if l.strip()]
def row(name): return next((r for r in ROWS if r["experiment"] == name), None)

C = dict(obs="#bcd7f0", obs_s="#5a8fc0", ho="#efefef", ho_s="#cfcfcf",
         lab="#cdbff0", lab_s="#8f79c8", ref="#bfe6de", ref_s="#4fa895",
         tgt="#d0453f", box="#f4f7fb", box_s="#c3ccd6", txt="#1c2733", mut="#67727e")


def arch_svg() -> str:
    W, H = 1040, 720
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["mut"]}"/></marker></defs>']
    def t(x, y, s, sz=13, col=C["txt"], w="normal", anc="start"):
        P.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col}" font-weight="{w}" text-anchor="{anc}">{s}</text>')
    def box(x, y, w, h, fill=C["box"], stroke=C["box_s"], rx=7):
        P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
    def chip(x, y, s, fill, stroke):
        P.append(f'<rect x="{x}" y="{y}" width="150" height="26" rx="5" fill="{fill}" stroke="{stroke}"/>')
        t(x+75, y+17, s, 12, C["txt"], "normal", "middle")

    t(28, 32, "Cross-completion (variant B) — tokens, context, and what is held out", 19, C["txt"], "bold")

    # ── 1 · token / sample composition ──
    t(28, 66, "1 · How a token is composed", 15, C["txt"], "bold")
    box(40, 78, 150, 30); t(115, 98, "pixel = (pos p, value v)", 12, C["txt"], "normal", "middle")
    P.append(f'<line x1="190" y1="93" x2="238" y2="93" stroke="{C["mut"]}" marker-end="url(#a)"/>')
    t(214, 84, "bind", 10.5, C["mut"], "normal", "middle")
    box(240, 78, 210, 30); t(345, 98, "token = r_pos[p] ⊛ ψ(value)", 12.5, C["txt"], "normal", "middle")
    t(470, 90, "⊛ = HRR bind (role ⊛ filler).", 12, C["mut"])
    t(470, 106, "roles r_pos / r_label / r_ref are keys; value / label / ref are fillers.", 12, C["mut"])
    # sample pattern
    t(40, 138, "A sample's pattern superposes its observed pixel-tokens and adds a label and a ref token:", 12.5, C["txt"])
    t(40, 168, "P", 15, C["txt"], "bold"); t(52, 172, "s", 10, C["mut"])
    t(66, 168, "=  Σ",  15, C["txt"]); t(96, 174, "obs", 9, C["mut"])
    chip(120, 155, "r_pos ⊛ value", C["obs"], C["obs_s"]); t(276, 172, "+", 15, C["txt"])
    chip(292, 155, "r_label ⊛ label", C["lab"], C["lab_s"]); t(448, 172, "+", 15, C["txt"])
    chip(464, 155, "r_ref ⊛ ref", C["ref"], C["ref_s"])
    t(632, 165, "Any field is read back by", 12, C["mut"]); t(632, 181, "unbinding its role from the pattern.", 12, C["mut"])

    # ── grid helper ──
    def grid(ox, oy, states, qrow, ncol_px):
        cw, ch, g = 28, 22, 4
        for r, rowst in enumerate(states):
            yy = oy + r*(ch+g)
            t(ox-8, yy+ch/2+4, f"s{r+1}", 11, (C["txt"] if r == qrow else C["mut"]),
              ("bold" if r == qrow else "normal"), "end")
            for c, st in enumerate(rowst):
                xx = ox + c*(cw+g)
                base = st.split(":")[0]
                fill = {"obs": C["obs"], "ho": C["ho"], "lab": C["lab"], "ref": C["ref"]}[base]
                stroke = {"obs": C["obs_s"], "ho": C["ho_s"], "lab": C["lab_s"], "ref": C["ref_s"]}[base]
                P.append(f'<rect x="{xx}" y="{yy}" width="{cw}" height="{ch}" rx="3" fill="{fill}" stroke="{stroke}"/>')
                if base == "ho": t(xx+cw/2, yy+ch/2+4, "·", 13, C["ho_s"], "normal", "middle")
                if base == "ref": t(xx+cw/2, yy+ch/2+4, "id", 9, C["ref_s"], "normal", "middle")
                if ":t" in st:
                    P.append(f'<rect x="{xx-2}" y="{yy-2}" width="{cw+4}" height="{ch+4}" rx="4" fill="none" stroke="{C["tgt"]}" stroke-width="3"/>')
                    t(xx+cw/2, yy+ch/2+4, "?", 12, C["tgt"], "bold", "middle")
        # column captions
        lc = ox + ncol_px*(cw+g)
        t(lc+cw/2, oy-8, "label", 10, C["lab_s"], "bold", "middle")
        t(lc+(cw+g)+cw/2, oy-8, "ref", 10, C["ref_s"], "bold", "middle")

    # ── 2 · full training context ──
    t(28, 224, "2 · The full training context = every sample's pattern, held together in one memory", 15, C["txt"], "bold")
    ctx = [
        ["obs","obs","ho","obs","obs","ho","lab","ref"],
        ["obs","ho","obs","obs","ho","obs","lab","ref"],
        ["obs","obs","obs","ho","obs","obs","lab","ref"],
        ["ho","obs","obs","obs","obs","ho","lab","ref"],
    ]
    grid(60, 246, ctx, -1, 6)
    t(360, 262, "Each row = one sample sᵢ.", 12.5, C["txt"])
    t(360, 282, "• blue = an observed pixel-token (r_pos ⊛ value)", 12, C["mut"])
    t(360, 300, "• gray · = a pixel NOT observed (~50% per row)", 12, C["mut"])
    t(360, 318, "• purple = the label token   • teal = a random ref tag", 12, C["mut"])
    t(360, 340, "There is no feature matrix — the context is just this bag of", 12, C["mut"])
    t(360, 356, "per-sample patterns (positions and ref assignment are random).", 12, C["mut"])

    # ── 3 · what is predicted (label held out) ──
    t(28, 402, "3 · What is predicted — the query's label (and target pixels) are HELD OUT", 15, C["txt"], "bold")

    tr = [
        ["obs","obs","ho","obs","obs","ho","lab","ref"],
        ["obs","ho","obs","obs","ho","obs","lab","ref"],
        ["obs","ho:t","obs","ho","obs","obs","lab:t","ref"],   # query s3: pixel + label held out
        ["ho","obs","obs","obs","obs","ho","lab","ref"],
    ]
    grid(60, 448, tr, 2, 6)
    t(60, 448+4*26+14, "TRAINING", 14, C["txt"], "bold")
    for i, ln in enumerate([
        "Query = one sample of the SAME episode. Its label and a masked",
        "pixel (red ?) are held out and predicted from the OTHER samples",
        "(leave-self-out cross read) → cross-entropy loss. Targets are",
        "OBSERVED values (known); the label is never used to predict itself."]):
        t(60, 448+4*26+34+i*16, ln, 12, C["mut"])

    te = [
        ["obs","obs","ho","obs","obs","ho","lab","ref"],
        ["obs","ho","obs","obs","ho","obs","lab","ref"],
        ["ho","obs","obs","obs","obs","ho","lab","ref"],
        ["obs","ho","obs","obs","obs","ho","lab","ref"],
    ]
    grid(600, 448, te, -1, 6)
    t(600, 448+4*26+14, "TEST", 14, C["txt"], "bold")
    # query strip (a separate TEST sample)
    qy = 448+4*26+40
    for c in range(8):
        base = ["obs","obs","obs","obs","obs","obs","lab","ref"][c]
        fill = {"obs":C["obs"],"lab":C["lab"],"ref":C["ref"]}[base]; stroke={"obs":C["obs_s"],"lab":C["lab_s"],"ref":C["ref_s"]}[base]
        xx = 600 + c*32
        P.append(f'<rect x="{xx}" y="{qy}" width="28" height="22" rx="3" fill="{fill}" stroke="{stroke}"/>')
        if base == "lab":
            P.append(f'<rect x="{xx-2}" y="{qy-2}" width="32" height="26" rx="4" fill="none" stroke="{C["tgt"]}" stroke-width="3"/>')
            t(xx+14, qy+15, "?", 12, C["tgt"], "bold", "middle")
        if base == "ref": t(xx+14, qy+15, "id", 9, C["ref_s"], "normal", "middle")
    t(592, qy+15, "q*", 11, C["tgt"], "bold", "end")
    for i, ln in enumerate([
        "Support (s1–s4) drawn from the TRAIN split; query q* drawn from",
        "the TEST split (q* ∉ support). q* can be a FULL image (supervised)",
        "or partial; predict its held-out label (red ?) and pixels from the",
        "support. Measures generalization to unseen samples."]):
        t(600, qy+40+i*16, ln, 12, C["mut"])
    P.append("</svg>")
    return "".join(P)


def main():
    arch_url = save_media("universal-ar_cross_completion_arch.svg",
                          io.BytesIO(arch_svg().encode("utf-8")), "image/svg+xml")

    e2 = row("exp2"); h = e2["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(h["step"], h["loss"], color="#2b6cb0"); ax[0].set_title("exp2 training loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[1].plot(h["step"], h["label_attn"], color="#2b6cb0", label="attention read")
    ax[1].plot(h["step"], h["label_lin"], color="#b0413a", label="linear read")
    ax[1].axhline(0.10, ls="--", color="#999", lw=1, label="chance (0.10)")
    ax[1].set_title("exp2 label generalization"); ax[1].set_xlabel("step"); ax[1].set_ylabel("accuracy"); ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout()
    curves_url = save_matplotlib_figure("universal-ar_cross_completion_exp2", fig, format="svg"); plt.close(fig)

    e6 = row("exp6")
    e6_line = (f"**exp6 (variant B, S=256)** confirms it: label **{e6['label_attn']:.2f}** and content-ink "
               f"**{e6['ink_attn']:.2f}** — longer context *helps* variant B (vs variant A, where it hurt)."
               ) if e6 else "exp6 (variant B at S=256) is the pending long-context test."

    md = f"""# Universal AR — Cross-Completion (variant B)

**Cross-completion** trains the memory to complete an entry of one sample from the
*other* samples in the episode, over the (samples × positions) matrix. Its explicit
cross-sample loss is what makes label/content *generalize* (variant A, self-recall,
never trains the cross kernel). See [concepts.md](concepts.md) §2.

## Tokens, context, and what is held out

![Cross-completion: tokens, context, held-out targets]({arch_url})

**A token** is one pixel bound as `r_pos[p] ⊛ ψ(value)` — a role⊗filler pair
(`⊛` = HRR bind). A **sample's pattern** superposes its observed pixel-tokens and
adds a label token (`r_label ⊛ label`) and a random ref tag (`r_ref ⊛ ref`):
```
P_s = Σ_observed (r_pos ⊛ value)  +  r_label ⊛ label  +  r_ref ⊛ ref
```
Any field is read back by **unbinding its role**. There is no feature matrix — the
**full context is just this bag of per-sample patterns** (positions observed and
ref tags are randomized per episode).

**What is predicted — the query's label is held out.** For a query sample `q`, its
**label** and a masked **pixel** are held out and predicted from the *other*
samples via a leave-self-out cross read → cross-entropy loss; `q`'s own label is
never used to predict itself.
- **Training:** query is a sample of the same (train) episode; targets are
  *observed* values, held out only from `q`'s own contribution.
- **Testing:** the support set is drawn from **train**, the query `q*` from
  **test** (`q* ∉ support`); `q*` may be a **full** image (supervised) or partial,
  and we predict its held-out label and pixels — measuring generalization.

## Result — exp2 (variant B, S=32)

| metric | attn | linear | chance / baseline |
|---|---|---|---|
| retrieval (recall) | {e2['recall']:.3f} | — | 0.03 |
| **label generalization** | **{e2['label_attn']:.3f}** | {e2['label_lin']:.3f} | 0.10 |
| content (raw) | {e2.get('ho_attn', float('nan')):.3f} | {e2.get('ho_lin', float('nan')):.3f} | 0.03 (bg ≈ 0.81) |

![exp2 loss and label curves]({curves_url})

- Label generalization reaches {e2['label_attn']:.2f} (attention); linear stays at
  chance ({e2['label_lin']:.2f}) — softmax is needed for the sharp label retrieval.
- Raw content (~0.83) is background-confounded (~81% bin-0); treat as unmeasured here.

{e6_line}

## Next steps
- **Longer training + supervised / supervised+ eval** (train support, test query) —
  in progress.
- Add the **ink-only content metric** to the S=32 variant B (exp2 lacked it).
- **Anonymized/permuted labels** → true in-context-learning test (fixed labels ⇒
  only the weak form).
"""
    url = save_report("universal-ar_report_cross_completion", md)
    print("ARCH_SVG:", arch_url); print("CURVES_SVG:", curves_url); print("REPORT:", url)


if __name__ == "__main__":
    main()
