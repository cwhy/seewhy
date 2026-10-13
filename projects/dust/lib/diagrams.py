"""Hand-authored SVG diagrams that explain the two algorithms.

These are not data figures, so they do not go through gribouille. They are plain SVG
strings: `scripts/tmp/render_diagrams.py` writes them to disk, and the report scripts
upload them with `shared_lib.media.save_media(name, svg, "image/svg+xml")`.

Colours follow the paper: DUST violet, EGGROLL grey, backprop black.
"""

INK = "#1f2937"
MUTE = "#6b7280"
FAINT = "#d1d5db"
DUST = "#7c3aed"
EGG = "#6b7280"
HUES = ["#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]
FONT = "font-family=\"Helvetica, Arial, sans-serif\""


def _svg(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT} '
            f'font-size="14" fill="{INK}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>'
            f'<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{MUTE}"/></marker></defs>'
            f'{body}</svg>')


def _t(x, y, s, size=14, anchor="middle", fill=INK, weight="normal", style="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" font-style="{style}">{s}</text>')


def _box(x, y, w, h, fill="#ffffff", stroke=INK, r=6, sw=1.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def _arrow(x1, y1, x2, y2, stroke=MUTE, sw=1.5):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" '
            f'marker-end="url(#ar)"/>')


def pass_value() -> str:
    """What one forward pass is worth: one population member (EGGROLL) against one per token (DUST)."""
    b = []
    n = 5
    for panel, (x0, name, col) in enumerate([(30, "EGGROLL: perturb the weights", EGG),
                                              (480, "DUST: perturb the activations", DUST)]):
        pw = 400
        b.append(_t(x0 + pw / 2, 34, name, 18, weight="bold", fill=col))
        # tokens along the bottom
        xs = [x0 + 40 + i * 70 for i in range(n)]
        for i, x in enumerate(xs):
            b.append(_box(x - 22, 376, 44, 30, "#f9fafb", MUTE, 5, 1.2))
            b.append(_t(x, 396, f"t{i+1}" if i < n - 1 else "tT", 13, fill=MUTE))
        b.append(_t(x0 + pw / 2, 428, "the 2048 tokens of a sequence, in one forward pass", 12, fill=MUTE, style="italic"))
        # token lines first, so the opaque layer bars sit on top of them
        for i, x in enumerate(xs):
            b.append(f'<line x1="{x}" y1="374" x2="{x}" y2="86" stroke="{FAINT}" stroke-width="1.4" marker-end="url(#ar)"/>')
        for k, (y, lab) in enumerate([(300, "block 1"), (210, "block l"), (120, "head")]):
            b.append(_box(x0 + 10, y, pw - 20, 44, "#f3f4f6", MUTE, 8, 1.2))
            b.append(_t(x0 + pw / 2, y + 27, lab, 13, fill=MUTE))
            if panel == 0:
                b.append(_t(x0 + 22, y + 27, "W+σabᵀ", 12, fill=EGG, anchor="start", weight="bold"))
            else:
                b.append(_t(x0 + 22, y + 27, "W", 13, fill=MUTE, anchor="start"))
        if panel == 1:
            # per-token noise dots on the output of block l, per-token losses on top
            for i, x in enumerate(xs):
                c = HUES[i % len(HUES)]
                b.append(f'<circle cx="{x}" cy="190" r="9" fill="{c}"/>')
                b.append(_t(x + 16, 186, f"σa{i+1}" if i < n - 1 else "σaT", 11, anchor="start", fill=c, weight="bold"))
                b.append(_box(x - 22, 54, 44, 28, "#ffffff", c, 5, 2))
                b.append(_t(x, 73, f"ℓ{i+1}" if i < n - 1 else "ℓT", 13, fill=c, weight="bold"))
            b.append(_t(x0 + pw / 2, 106, "T separate losses — one reward per token", 13, fill=DUST))
            b.append(_t(x0 + pw / 2, 448, "1 pass  =  T population members  =  T numbers", 16, weight="bold", fill=DUST))
        else:
            b.append(_box(x0 + pw / 2 - 70, 48, 140, 36, "#ffffff", EGG, 6, 2))
            b.append(_t(x0 + pw / 2, 72, "L = mean loss", 14, fill=EGG, weight="bold"))
            b.append(_t(x0 + pw / 2, 106, "all tokens collapse into one reward", 13, fill=EGG))
            b.append(_t(x0 + pw / 2, 448, "1 pass  =  1 population member  =  1 number", 16, weight="bold", fill=EGG))
    b.append(f'<line x1="450" y1="20" x2="450" y2="440" stroke="{FAINT}" stroke-width="1" stroke-dasharray="4 4"/>')
    return _svg(910, 462, "".join(b))


def reward_to_gradient() -> str:
    """How DUST turns per-token rewards into a weight gradient for one linear layer."""
    b = []
    b.append(_t(450, 26, "DUST on one linear layer  y = W x", 18, weight="bold", fill=DUST))
    # forward chain
    b.append(_box(30, 92, 120, 54, "#f9fafb", MUTE))
    b.append(_t(90, 116, "input xₜ", 14, weight="bold"))
    b.append(_t(90, 135, "cached, clean pass", 11, fill=MUTE))
    b.append(_arrow(150, 119, 200, 119))
    b.append(_box(200, 92, 90, 54, "#f3f4f6", MUTE))
    b.append(_t(245, 126, "W", 20, weight="bold"))
    b.append(_arrow(290, 119, 340, 119))
    b.append('<circle cx="360" cy="119" r="20" fill="#ffffff" stroke="%s" stroke-width="2"/>' % DUST)
    b.append(_t(360, 126, "+", 22, fill=DUST, weight="bold"))
    b.append(_t(360, 62, "σ aₜ", 15, fill=DUST, weight="bold"))
    b.append(_t(360, 78, "fresh N(0, I) at every token", 11, fill=DUST))
    b.append(f'<line x1="360" y1="84" x2="360" y2="97" stroke="{DUST}" stroke-width="2" marker-end="url(#ar)"/>')
    b.append(_arrow(380, 119, 430, 119))
    b.append(_box(430, 92, 170, 54, "#f3f4f6", MUTE))
    b.append(_t(515, 116, "rest of the network", 14))
    b.append(_t(515, 135, "rerun from this block", 11, fill=MUTE))
    b.append(_arrow(600, 119, 650, 119))
    b.append(_box(650, 92, 130, 54, "#ffffff", DUST, 6, 2))
    b.append(_t(715, 116, "loss ℓₜ", 15, fill=DUST, weight="bold"))
    b.append(_t(715, 135, "that token's own", 11, fill=DUST))
    b.append(_t(796, 112, "× K draws", 14, anchor="start", fill=MUTE, style="italic"))
    b.append(_t(796, 130, "n at a time", 12, anchor="start", fill=MUTE, style="italic"))
    # three steps
    steps = [
        ("1  reward", "rₜ = mean(ℓₜ) − ℓₜ", "positive when this draw's noise\nlowered token t's loss", 30),
        ("2  error at the output", "ĝₜ = −(1 / Kσ) · Σ rₜ aₜ", "noise that helped, averaged over\ndraws, points away from the gradient", 320),
        ("3  weight gradient", "Ĝ = Σₜ ĝₜ xₜᵀ", "same outer product as backprop —\nonly ĝₜ comes from the population", 610),
    ]
    for title, eq, note, x in steps:
        b.append(_box(x, 200, 270, 150, "#faf5ff", DUST, 10, 1.5))
        b.append(_t(x + 135, 228, title, 15, weight="bold", fill=DUST))
        b.append(_t(x + 135, 268, eq, 18, weight="bold"))
        for i, line in enumerate(note.split("\n")):
            b.append(_t(x + 135, 304 + i * 18, line, 12, fill=MUTE))
    b.append(_arrow(300, 275, 318, 275, DUST, 2))
    b.append(_arrow(590, 275, 608, 275, DUST, 2))
    b.append(f'<path d="M715 146 V176 H165 V197" fill="none" stroke="{DUST}" stroke-width="1.5" marker-end="url(#ar)"/>')
    # backprop contrast
    b.append(_box(30, 380, 840, 60, "#f9fafb", FAINT, 8, 1.2))
    b.append(_t(450, 406, "Backprop:  ĝₜ = ∂L/∂yₜ  from the chain rule.     DUST:  ĝₜ estimated from forward passes only.", 14))
    b.append(_t(450, 427, "Embeddings: xₜ is one-hot, so Σₜ ĝₜ xₜᵀ is a scatter-add of ĝₜ into token t's row.", 12, fill=MUTE))
    return _svg(900, 460, "".join(b))


def eggroll_update() -> str:
    """EGGROLL for one weight matrix: a rank-one perturbation, an antithetic pair, one number."""
    b = []
    b.append(_t(450, 26, "EGGROLL on one weight matrix W  (m × n)", 18, weight="bold", fill=EGG))

    # panel 1: the perturbation is an outer product, never stored as a matrix
    b.append(_t(150, 62, "1  perturb", 15, weight="bold", fill=EGG))
    b.append(_box(40, 84, 22, 100, "#e5e7eb", EGG, 3, 1.5))
    b.append(_t(51, 78, "a", 13, fill=EGG, weight="bold"))
    b.append(_t(51, 204, "m", 11, fill=MUTE))
    b.append(_t(80, 140, "×", 20, fill=MUTE))
    b.append(_box(100, 128, 100, 22, "#e5e7eb", EGG, 3, 1.5))
    b.append(_t(150, 122, "bᵀ", 13, fill=EGG, weight="bold"))
    b.append(_t(150, 166, "n", 11, fill=MUTE))
    b.append(_t(212, 140, "=", 20, fill=MUTE))
    # the product, drawn as the rank-one pattern it is: every row a multiple of bᵀ
    for i in range(6):
        shade = ["#f3f4f6", "#d1d5db", "#9ca3af", "#d1d5db", "#f3f4f6", "#e5e7eb"][i]
        b.append(f'<rect x="232" y="{84 + i * 16.6}" width="76" height="16" fill="{shade}" stroke="{EGG}" stroke-width="0.6"/>')
    b.append(_t(270, 204, "σ a bᵀ  (rank 1)", 12, fill=EGG))
    b.append(_t(150, 232, "m + n numbers per member,", 12, fill=MUTE))
    b.append(_t(150, 248, "never the m × n matrix", 12, fill=MUTE))

    # panel 2: an antithetic pair is two forward passes and one number
    b.append(_t(470, 62, "2  evaluate the pair", 15, weight="bold", fill=EGG))
    b.append(_box(350, 84, 240, 46, "#f9fafb", MUTE, 8, 1.2))
    b.append(_t(470, 112, "forward pass with W + σ a bᵀ   →   L⁺", 13))
    b.append(_box(350, 140, 240, 46, "#f9fafb", MUTE, 8, 1.2))
    b.append(_t(470, 168, "forward pass with W − σ a bᵀ   →   L⁻", 13))
    b.append(_box(380, 200, 180, 40, "#ffffff", EGG, 8, 2))
    b.append(_t(470, 225, "f = L⁺ − L⁻   one number", 13, weight="bold", fill=EGG))
    b.append(_arrow(332, 140, 346, 140))

    # panel 3: the estimate is a fitness-weighted sum of the perturbations
    b.append(_t(770, 62, "3  combine P members", 15, weight="bold", fill=EGG))
    b.append(_box(640, 84, 260, 90, "#f9fafb", MUTE, 8, 1.2))
    b.append(_t(770, 118, "ĝ = 1 / (P σ²) · Σᵢ fᵢ · σ aᵢ bᵢᵀ", 14, weight="bold"))
    b.append(_t(770, 142, "a sum of P rank-one matrices:", 12, fill=MUTE))
    b.append(_t(770, 158, "its rank can reach P", 12, fill=MUTE))
    b.append(_box(680, 190, 180, 40, "#ffffff", EGG, 8, 2))
    b.append(_t(770, 215, "Adam(ĝ)  →  new W", 13, weight="bold", fill=EGG))
    b.append(_arrow(600, 140, 636, 140))

    b.append(_box(30, 270, 870, 56, "#f9fafb", FAINT, 8, 1.2))
    b.append(_t(465, 294, "Every member scores the whole batch and returns ONE number. That number has to say which way to move all m × n entries,", 13))
    b.append(_t(465, 314, "of every matrix at once. With P pairs, the estimate's cosine to the true gradient grows only as √(P / d).", 13))
    return _svg(930, 340, "".join(b))


ALL = {"pass_value": pass_value, "reward_to_gradient": reward_to_gradient, "eggroll_update": eggroll_update}
