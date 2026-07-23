"""
Generate the dedicated Cross-Completion (variant B) report for universal-ar:
an architecture diagram of how the cross-completion data is constructed for
training vs testing, plus the exp2 results. Publishes to R2.

Usage (on the server): uv run python projects/universal-ar/scripts/gen_report_cross_completion.py
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
def row(name): return next(r for r in ROWS if r["experiment"] == name)


# ── architecture diagram (hand-authored SVG) ──────────────────────────────────
def arch_svg() -> str:
    W, H = 1000, 660
    C = dict(obs="#bcd7f0", obs_s="#5a8fc0", ho="#ededed", ho_s="#cfcfcf",
             lab="#cdbff0", lab_s="#8f79c8", tgt_tr="#e08a1e", tgt_te="#d0453f",
             txt="#1c2733", mut="#6b7785", box="#f3f6fa", box_s="#c3ccd6")
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">']
    p.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    p.append('<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
             f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["mut"]}"/></marker></defs>')
    t = lambda x,y,s,sz=13,col=C["txt"],w="normal",anc="start": p.append(
        f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col}" font-weight="{w}" text-anchor="{anc}">{s}</text>')

    t(30, 34, "Cross-completion (variant B) — how the data is constructed", 19, C["txt"], "bold")
    t(30, 56, "Dataset = one (samples × positions) matrix; each row (sample) observes ~50% of its entries; one extra column is the label.", 13, C["mut"])

    # ── read mechanism flow ──
    y0 = 82
    t(30, y0+14, "Predicting one target entry of query row q  (leave-self-out cross read):", 14, C["txt"], "bold")
    boxes = [("q's observed\nentries", 40), ("build key\nP_q", 210), ("softmax sim to\nOTHER rows' P_s", 360),
             ("Σ wₛ · P_s\n(exclude q)", 560), ("unbind\ntarget field", 720), ("predicted\nvalue", 872)]
    by, bw, bh = y0+30, 138, 52
    for i,(lab,x) in enumerate(boxes):
        p.append(f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="7" fill="{C["box"]}" stroke="{C["box_s"]}"/>')
        for j,ln in enumerate(lab.split("\n")):
            t(x+bw/2, by+22+j*16, ln, 12.5, C["txt"], "normal", "middle")
        if i < len(boxes)-1:
            xa = x+bw
            p.append(f'<line x1="{xa}" y1="{by+bh/2}" x2="{boxes[i+1][1]}" y2="{by+bh/2}" stroke="{C["mut"]}" marker-end="url(#a)"/>')

    # ── grid helper ──
    def grid(ox, oy, states, qrow, title, cap):
        cw, ch, g = 30, 24, 4
        ncol = len(states[0])
        t(ox, oy-12, title, 15, C["txt"], "bold")
        # column header for label
        t(ox+(ncol-1)*(cw+g)+cw/2, oy-12, "label", 11, C["lab_s"], "bold", "middle")
        for r,rowst in enumerate(states):
            yy = oy + r*(ch+g)
            t(ox-10, yy+ch/2+4, f"s{r+1}", 11, (C["txt"] if r==qrow else C["mut"]), ("bold" if r==qrow else "normal"), "end")
            for c,st in enumerate(rowst):
                xx = ox + c*(cw+g)
                fill = {"obs":C["obs"],"ho":C["ho"],"lab":C["lab"]}.get(st.split(":")[0], C["obs"])
                p.append(f'<rect x="{xx}" y="{yy}" width="{cw}" height="{ch}" rx="3" fill="{fill}" stroke="{C["obs_s"] if st.startswith("obs") else (C["lab_s"] if st.startswith("lab") else C["ho_s"])}"/>')
                if st.split(":")[0]=="ho":
                    t(xx+cw/2, yy+ch/2+4, "·", 14, C["ho_s"], "normal", "middle")
                # target overlays
                if ":tr" in st:
                    p.append(f'<rect x="{xx-2}" y="{yy-2}" width="{cw+4}" height="{ch+4}" rx="4" fill="none" stroke="{C["tgt_tr"]}" stroke-width="3"/>')
                if ":te" in st:
                    p.append(f'<rect x="{xx-2}" y="{yy-2}" width="{cw+4}" height="{ch+4}" rx="4" fill="none" stroke="{C["tgt_te"]}" stroke-width="3"/>')
                    t(xx+cw/2, yy+ch/2+4, "?", 13, C["tgt_te"], "bold", "middle")
        for i,ln in enumerate(cap):
            t(ox-2, oy+len(states)*(ch+g)+18+i*16, ln, 12, C["mut"])

    gy = 330
    # TRAIN: query row s3, target = an OBSERVED pixel (orange). context rows give labels.
    train = [
        ["obs","obs","ho","obs","obs","ho","lab"],
        ["obs","ho","obs","obs","ho","obs","lab"],
        ["obs","obs:tr","obs","ho","obs","obs","lab"],   # s3 query; target = observed pixel
        ["ho","obs","obs","obs","obs","ho","lab"],
    ]
    grid(60, gy, train, 2, "TRAINING",
         ["Target = an OBSERVED entry (value is known).", "Masked from q's own key, predicted from the OTHER rows.",
          "→ cross-entropy loss. This TRAINS the cross-sample kernel.", "(no held-out / unseen data enters the loss)"])
    # TEST: query row s3, targets = a held-out pixel + the label (red ?).
    test = [
        ["obs","obs","ho","obs","obs","ho","lab"],
        ["obs","ho","obs","obs","ho","obs","lab"],
        ["obs","ho:te","obs","ho","obs","obs","lab:te"],  # s3 query; held-out pixel + label queried
        ["ho","obs","obs","obs","obs","ho","lab"],
    ]
    grid(600, gy, test, 2, "TEST",
         ["Target = a HELD-OUT entry never observed:", "• the label column  → label generalization",
          "• an unseen pixel   → content generalization", "Same read; measures generalization (not trained)."])

    # legend
    ly = H-40
    items = [("observed", C["obs"], C["obs_s"]), ("held-out", C["ho"], C["ho_s"]), ("label col", C["lab"], C["lab_s"])]
    lx = 60
    for lab,f,s in items:
        p.append(f'<rect x="{lx}" y="{ly-12}" width="20" height="14" rx="3" fill="{f}" stroke="{s}"/>'); t(lx+26, ly, lab, 12, C["mut"]); lx += 120
    p.append(f'<rect x="{lx}" y="{ly-12}" width="20" height="14" rx="3" fill="none" stroke="{C["tgt_tr"]}" stroke-width="3"/>'); t(lx+26, ly, "train target (observed)", 12, C["mut"]); lx += 190
    p.append(f'<rect x="{lx}" y="{ly-12}" width="20" height="14" rx="3" fill="none" stroke="{C["tgt_te"]}" stroke-width="3"/>'); t(lx+26, ly, "test target (held-out)", 12, C["mut"])
    p.append("</svg>")
    return "".join(p)


def main():
    arch_url = save_media("universal-ar_cross_completion_arch.svg",
                          io.BytesIO(arch_svg().encode("utf-8")), "image/svg+xml")

    e2 = row("exp2"); h = e2["history"]
    # exp2 loss + label curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(h["step"], h["loss"], color="#2b6cb0"); ax[0].set_title("exp2 training loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[1].plot(h["step"], h["label_attn"], color="#2b6cb0", label="attention read")
    ax[1].plot(h["step"], h["label_lin"], color="#b0413a", label="linear read")
    ax[1].axhline(0.10, ls="--", color="#999", lw=1, label="chance (0.10)")
    ax[1].set_title("exp2 label generalization"); ax[1].set_xlabel("step"); ax[1].set_ylabel("accuracy"); ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout()
    curves_url = save_matplotlib_figure("universal-ar_cross_completion_exp2", fig, format="svg")
    plt.close(fig)

    md = f"""# Universal AR — Cross-Completion (variant B)

**Cross-completion** trains the memory to complete an entry of one sample from the
*other* samples in the episode (leave-one-out), over the (samples × positions)
matrix. This is the variant whose explicit cross-sample loss makes label/content
*generalize* — in contrast to variant A (self-recall only), where the cross-sample
kernel is never trained. See [concepts.md](concepts.md) §2 for the auto-associative
memory this sits on.

## How the data is constructed (train vs test)

![Cross-completion data construction]({arch_url})

- An **episode** is one (samples × positions) matrix: each row (sample) exposes
  ~50% of its entries; one extra column holds the label.
- Each row is summarized into a pattern `P_s` from its **observed** entries
  (an amortized key — no per-sample lookup).
- To predict a target entry of query row `q`: a **leave-self-out** read —
  `softmax(sim(P_q, P_s))` over the *other* rows, aggregate their patterns,
  unbind the target field.
- **Training targets = OBSERVED entries** (value known): masked from `q`'s own key
  and predicted from the other rows → cross-entropy loss. This is what *trains the
  cross-sample kernel*; no held-out/unseen data enters the loss.
- **Test targets = HELD-OUT entries** never observed: the **label** column
  (label generalization) and **unseen pixels** (content generalization). Same read,
  used only to *measure*.

## Result — exp2 (variant B, S=32)

| metric | attn | linear | chance / baseline |
|---|---|---|---|
| retrieval (recall) | {e2['recall']:.3f} | — | 0.03 |
| **label generalization** | **{e2['label_attn']:.3f}** | {e2['label_lin']:.3f} | 0.10 |
| content (raw) | {e2.get('ho_attn', float('nan')):.3f} | {e2.get('ho_lin', float('nan')):.3f} | 0.03 (bg ≈ 0.81) |

![exp2 loss and label curves]({curves_url})

- **Label generalization reaches {e2['label_attn']:.2f}** with the attention read
  (linear read stays at chance, {e2['label_lin']:.2f}) — softmax is needed for the
  sharp single-column retrieval.
- The raw content number (~0.83) is **background-confounded** (~81% of MNIST pixels
  are bin-0); exp2 predates the ink-only metric, so treat content as *unmeasured*
  here.

## Why it matters (vs variant A)

Variant A (self-recall only) reaches label ≈ 0.40; **cross-completion roughly
doubles it (0.74)**. The cross term is what makes label/content generalize — and,
by extension, it should let *longer context help* rather than hurt (variant A's
label dropped 0.40 → 0.15 as S grew 32 → 256, precisely because its kernel is
untrained).

## Next steps
- **exp6 = variant B at long context (S=256)** — does a *trained* kernel flip
  longer context from liability to asset?
- Add the **ink-only content metric** to variant B (exp2 lacked it).
- **Anonymized/permuted labels** to turn label-gen into a true in-context-learning
  test (fixed labels ⇒ only the weak form).
"""
    url = save_report("universal-ar_report_cross_completion", md)
    print("ARCH_SVG:", arch_url)
    print("CURVES_SVG:", curves_url)
    print("REPORT:", url)


if __name__ == "__main__":
    main()
