"""
Hand-authored SVG explainer diagrams, uploaded to R2.

These carry the things a data plot cannot: how the task is built, what the search looks
like, why the dense column is degenerate, and how the experiments depend on each other.

Every diagram paints its own white background — report pages render in the viewer's colour
scheme, and a transparent SVG with dark strokes disappears in dark mode.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_diagrams.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_lib.media import save_media          # noqa: E402

BLUE, RED, GREEN, PURPLE = "#4a7ebb", "#c0504d", "#9bbb59", "#674ea7"
INK, MUTED, LINE = "#1a1a1a", "#666", "#bbb"
FONT = "ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "system-ui, -apple-system, Segoe UI, sans-serif"


def svg(w, h, body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
            f"width='100%' role='img' aria-label='{title}'>"
            f"<rect width='{w}' height='{h}' fill='#ffffff'/>"
            f"<style>text{{font-family:{SANS};fill:{INK}}} .m{{font-family:{FONT}}}"
            f".s{{font-size:11px}} .xs{{font-size:10px}} .lbl{{font-size:13px;font-weight:600}}"
            f".mut{{fill:{MUTED}}}</style>{body}</svg>")


def upload(name, s):
    url = save_media(f"{name}.svg", io.BytesIO(s.encode()), "image/svg+xml")
    print(f"  {name:<28} {url}")
    return url


# ── 1. the task ───────────────────────────────────────────────────────────────
def task_diagram():
    S, cell = 8, 26
    x0 = [1, 0, 1, 1, 0, 0, 1, 0]
    row = [0, 1, 0, 0, 1, 0, 1, 0]          # support of row 3 of A: positions 1,4,6
    supp = [i for i, v in enumerate(row) if v]
    parity = sum(x0[i] for i in supp) % 2
    b = [f"<text class='lbl' x='20' y='26'>The linear map task — the correct attention "
         f"pattern is known by construction</text>"]

    # x0 row
    b.append(f"<text class='m s' x='20' y='68'>x₀</text>")
    for i, v in enumerate(x0):
        f = BLUE if i in supp else "#eef2f7"
        t = "#fff" if i in supp else MUTED
        b.append(f"<rect x='{50 + i*cell}' y='52' width='{cell-3}' height='{cell-3}' "
                 f"fill='{f}' stroke='{LINE}'/>"
                 f"<text class='m xs' x='{50 + i*cell + 8}' y='70' fill='{t}'>{v}</text>")
    b.append(f"<text class='s mut' x='{60 + S*cell}' y='70'>random bits — "
             f"unpredictable, CE = ln 2</text>")

    # matrix row
    b.append(f"<text class='m s' x='20' y='118'>A[3]</text>")
    for i, v in enumerate(row):
        f = PURPLE if v else "#f6f6f8"
        t = "#fff" if v else MUTED
        b.append(f"<rect x='{50 + i*cell}' y='102' width='{cell-3}' height='{cell-3}' "
                 f"fill='{f}' stroke='{LINE}'/>"
                 f"<text class='m xs' x='{50 + i*cell + 8}' y='120' fill='{t}'>{v}</text>")
    b.append(f"<text class='s mut' x='{60 + S*cell}' y='120'>row 3 of the secret matrix — "
             f"exactly s = 3 ones</text>")

    # arrows from the selected x0 cells down to the output
    out_x = 50 + 3 * cell
    for i in supp:
        b.append(f"<path d='M{50 + i*cell + 11} 78 C{50 + i*cell + 11} 150, "
                 f"{out_x + 11} 130, {out_x + 11} 168' fill='none' stroke='{BLUE}' "
                 f"stroke-width='1.4' opacity='0.75'/>")

    # x1 row
    b.append(f"<text class='m s' x='20' y='188'>x₁</text>")
    for i in range(S):
        hi = i == 3
        b.append(f"<rect x='{50 + i*cell}' y='172' width='{cell-3}' height='{cell-3}' "
                 f"fill='{RED if hi else '#eef2f7'}' stroke='{LINE}'/>")
        if hi:
            b.append(f"<text class='m xs' x='{50 + i*cell + 8}' y='190' "
                     f"fill='#fff'>{parity}</text>")
        else:
            b.append(f"<text class='m xs' x='{50 + i*cell + 9}' y='190' fill='{MUTED}'>?</text>")
    b.append(f"<text class='s' x='{60 + S*cell}' y='190'>x₁[3] = parity of the "
             f"{len(supp)} highlighted bits</text>")

    b.append(f"<text class='s mut' x='20' y='232'>The model sees the flattened sequence "
             f"x₀ then x₁ and predicts each token from the ones before it. To get x₁[i] right "
             f"it must attend</text>")
    b.append(f"<text class='s mut' x='20' y='250'>to exactly the s positions where row i of A "
             f"is 1 — so \"did a head find the right pattern?\" is measurable, not "
             f"interpretive.</text>")
    b.append(f"<text class='s' x='20' y='282'>Candidate patterns per row: "
             f"<tspan class='m'>C(S, s)</tspan> — 56 here, but "
             f"<tspan class='m'>C(16,8) = 12,870</tspan> and "
             f"<tspan class='m'>C(32,16) ≈ 6×10⁸</tspan>. That count is what sets "
             f"difficulty.</text>")
    return svg(700, 300, "".join(b), "the linear map task")


# ── 2. what emergence looks like ──────────────────────────────────────────────
def emergence_diagram():
    x0, y0, w, h = 60, 60, 380, 150
    plateau = y0 + 28
    b = [f"<text class='lbl' x='20' y='26'>What emergence looks like — and what is "
         f"happening underneath</text>"]
    b.append(f"<line x1='{x0}' y1='{y0+h}' x2='{x0+w}' y2='{y0+h}' stroke='{INK}'/>"
             f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0+h}' stroke='{INK}'/>")
    b.append(f"<text class='s mut' x='{x0+w/2-24}' y='{y0+h+24}'>training step</text>")
    b.append(f"<text class='s mut' transform='rotate(-90 20 {y0+h/2})' "
             f"x='20' y='{y0+h/2}'>loss</text>")
    b.append(f"<line x1='{x0}' y1='{plateau}' x2='{x0+w}' y2='{plateau}' "
             f"stroke='{LINE}' stroke-dasharray='4 3'/>"
             f"<text class='m xs mut' x='{x0+w+6}' y='{plateau+4}'>ln 2</text>")
    b.append(f"<text class='m xs mut' x='{x0+w+6}' y='{y0+h+4}'>0</text>")

    for i, (drop, col) in enumerate([(0.30, BLUE), (0.46, PURPLE), (0.72, RED)]):
        dx = x0 + w * drop
        b.append(f"<path d='M{x0} {plateau} L{dx} {plateau} C{dx+18} {plateau}, "
                 f"{dx+22} {y0+h-6}, {dx+46} {y0+h-4} L{x0+w} {y0+h-4}' fill='none' "
                 f"stroke='{col}' stroke-width='1.8' opacity='0.9'/>")
        b.append(f"<text class='m xs' x='{dx-4}' y='{y0-6+i*0}' fill='{col}'>seed {i+1}</text>")

    b.append(f"<text class='s' x='{x0+8}' y='{plateau-10}'>flat at chance — the search "
             f"is running</text>")
    b.append(f"<text class='s' x='{x0+w*0.30+52}' y='{y0+h-18}'>then it finds it</text>")

    # attention states
    def grid(ox, oy, sparse, label, col):
        out = [f"<text class='s' x='{ox}' y='{oy-8}'>{label}</text>"]
        n, c = 6, 15
        vals = ([[0.05]*n for _ in range(n)] if not sparse
                else [[0.0]*n for _ in range(n)])
        for r in range(n):
            for k in range(n):
                v = 0.28 if not sparse else (0.95 if (k == (r + 2) % n or k == (r + 4) % n)
                                             else 0.04)
                out.append(f"<rect x='{ox + k*c}' y='{oy + r*c}' width='{c-2}' height='{c-2}' "
                           f"fill='{col}' opacity='{v:.2f}' stroke='#e6e6e6' "
                           f"stroke-width='0.5'/>")
        return "".join(out)

    b.append(grid(500, 88, False, "attention before", BLUE))
    b.append(grid(610, 88, True, "attention after", RED))
    b.append(f"<text class='s mut' x='500' y='196'>near-uniform</text>")
    b.append(f"<text class='s mut' x='610' y='196'>sparse, correct</text>")

    b.append(f"<text class='s mut' x='20' y='248'>Loss sits at the value you would get by "
             f"guessing, then falls within a few hundred steps. The step at which that happens "
             f"varies</text>")
    b.append(f"<text class='s mut' x='20' y='266'>by 4–5× across otherwise identical runs "
             f"— our 16 seeds emerged between step 469 and step 2521. The drop coincides with "
             f"one head</text>")
    b.append(f"<text class='s mut' x='20' y='284'>switching from diffuse to the correct "
             f"support; deleting that head afterwards costs 4.23 nats, deleting an unaligned "
             f"one costs 0.08.</text>")
    return svg(700, 300, "".join(b), "what emergence looks like")


# ── 3. the degenerate dense column ────────────────────────────────────────────
def artifact_diagram():
    cell, S = 26, 8
    b = [f"<text class='lbl' x='20' y='26'>Why the densest setting is not a solve</text>"]
    b.append(f"<text class='s mut' x='20' y='50'>At s = S every row of A is all-ones, so "
             f"every output token is the same value: parity(x₀).</text>")

    b.append(f"<text class='m s' x='20' y='96'>x₁</text>")
    for i in range(S):
        first = i == 0
        b.append(f"<rect x='{50 + i*cell}' y='80' width='{cell-3}' height='{cell-3}' "
                 f"fill='{RED if first else GREEN}' stroke='{LINE}'/>"
                 f"<text class='m xs' x='{50 + i*cell + 8}' y='98' fill='#fff'>1</text>")
    b.append(f"<text class='s' x='{60 + S*cell}' y='98'>one value, repeated S times</text>")

    b.append(f"<rect x='47' y='77' width='{cell+2}' height='{cell+3}' fill='none' "
             f"stroke='{RED}' stroke-width='2'/>")
    b.append(f"<text class='s' x='44' y='132' fill='{RED}'>must be computed → chance, "
             f"50%</text>")
    for i in range(1, S):
        b.append(f"<path d='M{50 + (i-1)*cell + 11} 74 C{50 + (i-1)*cell + 14} 58, "
                 f"{50 + i*cell + 8} 58, {50 + i*cell + 11} 74' fill='none' "
                 f"stroke='{GREEN}' stroke-width='1.3'/>")
    b.append(f"<text class='s' x='{50 + 2*cell}' y='48' fill='#5f7a33'>copied from the "
             f"previous token — free</text>")

    rows = [("accuracy a model gets by copying", "1 − 0.5/S", "0.969 at S=16"),
            ("our emergence threshold", "0.95", "passed without learning"),
            ("residual loss from the one guessed token", "ln 2 / S", "0.0433 at S=16"),
            ("measured final loss, S=16 / S=32", "0.0433 / 0.0217", "ln2/16 / ln2/32")]
    y = 168
    b.append(f"<text class='s' x='20' y='{y-12}'>The arithmetic, and what we measured:</text>")
    for i, (a, c, d) in enumerate(rows):
        yy = y + i * 22
        b.append(f"<rect x='20' y='{yy-13}' width='660' height='20' "
                 f"fill='{'#f7f8fa' if i % 2 == 0 else '#ffffff'}'/>")
        b.append(f"<text class='s' x='28' y='{yy+2}'>{a}</text>"
                 f"<text class='m s' x='390' y='{yy+2}' fill='{PURPLE}'>{c}</text>"
                 f"<text class='m s mut' x='510' y='{yy+2}'>{d}</text>")

    b.append(f"<text class='s mut' x='20' y='282'>Verified per position: the first output "
             f"token sits at 0.488 accuracy, the other fifteen at 1.000. It is copying.</text>")
    return svg(700, 300, "".join(b), "the degenerate dense column")


# ── 4. the experiment map ─────────────────────────────────────────────────────
def map_diagram():
    b = [f"<text class='lbl' x='20' y='26'>How the experiments fit together</text>"]
    boxes = [
        ("exp1", "H1  abrupt & seed-random?", 20, 56, BLUE, "16 seeds, fixed A"),
        ("exp2", "H2  sparsity × context", 20, 116, BLUE, "24 cells, the difficulty surface"),
        ("exp4", "H3  is the jump the pattern?", 250, 56, GREEN, "dense probes + ablation"),
        ("exp3", "H4  heads vs head dim", 250, 116, GREEN, "config picked from exp2"),
        ("exp6", "H5  mixer vs transformer", 250, 176, RED, "3 regimes from exp2"),
        ("exp7", "H5  debugged", 480, 176, RED, "paper's cell + masking test"),
        ("exp5", "CA task, in context", 480, 116, PURPLE, "different rule per sequence"),
    ]
    for name, claim, x, y, col, sub in boxes:
        b.append(f"<rect x='{x}' y='{y}' width='200' height='46' rx='5' fill='#fff' "
                 f"stroke='{col}' stroke-width='1.6'/>")
        b.append(f"<text class='m s' x='{x+10}' y='{y+18}' fill='{col}'>{name}</text>"
                 f"<text class='s' x='{x+52}' y='{y+18}'>{claim}</text>"
                 f"<text class='xs mut' x='{x+10}' y='{y+35}'>{sub}</text>")

    def arrow(x1, y1, x2, y2, label=""):
        s = (f"<path d='M{x1} {y1} C{(x1+x2)/2} {y1}, {(x1+x2)/2} {y2}, {x2} {y2}' "
             f"fill='none' stroke='{MUTED}' stroke-width='1.2' "
             f"marker-end='url(#a)'/>")
        if label:
            s += f"<text class='xs mut' x='{(x1+x2)/2-18}' y='{(y1+y2)/2-4}'>{label}</text>"
        return s

    b.insert(0, "<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' "
                "markerWidth='6' markerHeight='6' orient='auto-start-reverse'>"
                f"<path d='M0 0 L10 5 L0 10 z' fill='{MUTED}'/></marker></defs>")
    b.append(arrow(220, 79, 250, 79, "same config"))
    b.append(arrow(220, 139, 250, 139, "picks cell"))
    b.append(arrow(220, 150, 250, 199, "picks cells"))
    b.append(arrow(450, 199, 480, 199, "re-tested"))

    notes = [("exp1 / exp4", "H1 holds — 4–5× spread, twice over", GREEN),
             ("exp4", "H3 holds — ablation 0.00 → 4.23 nats", GREEN),
             ("exp2", "H2 partly — band widens; dense column degenerate", "#b8860b"),
             ("exp3", "H4 partly — more heads help, then saturates", "#b8860b"),
             ("exp6 / exp7", "H5 under investigation — wrong cell first time", RED)]
    y = 254
    b.append(f"<text class='s' x='20' y='{y-10}'>Where it stands:</text>")
    for i, (who, what, col) in enumerate(notes):
        yy = y + i * 20
        b.append(f"<circle cx='28' cy='{yy+2}' r='4' fill='{col}'/>"
                 f"<text class='m xs' x='40' y='{yy+6}'>{who}</text>"
                 f"<text class='s' x='130' y='{yy+6}'>{what}</text>")
    return svg(700, 366, "".join(b), "how the experiments fit together")


if __name__ == "__main__":
    print("uploading diagrams:")
    upload("sparse_attn_emergence_diag_task", task_diagram())
    upload("sparse_attn_emergence_diag_emergence", emergence_diagram())
    upload("sparse_attn_emergence_diag_artifact", artifact_diagram())
    upload("sparse_attn_emergence_diag_map", map_diagram())
