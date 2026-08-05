"""
SVG diagrams for the two synthetic tasks: how every token is produced, and what the
next-token prediction problem looks like on top of it.

Four diagrams per task. Uploaded to R2; embedded by reports/tasks.md.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_task_diagrams.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.svgkit import (AMBER, BLUE, COLORS4, FG4, GREEN, INK, LINE, MONO, MUTED,  # noqa: E402
                        PURPLE, RED, TEAL, WASH, arrow, cell, curve, note, svg, title)
from shared_lib.media import save_media                                            # noqa: E402

CW = 26          # cell width


def upload(name, s):
    url = save_media(f"{name}.svg", io.BytesIO(s.encode()), "image/svg+xml")
    print(f"  {name:<40} {url}")
    return url


# ══════════════════════════════════════ LINEAR MAP ══════════════════════════════════════

def lm1_matrix():
    """How A is sampled: exactly s ones per row."""
    S, s, x0, y0 = 6, 2, 260, 60
    rows = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]]
    b = [title(20, 28, "1 · The secret matrix A — exactly s ones in every row")]
    b.append(note(20, 50, "Sampled once per run and never shown to the model. Row i says "
                          "which input bits decide output bit i."))
    for i, r in enumerate(rows):
        b.append(f"<text class='m xs mut' x='{x0-26}' y='{y0 + i*CW + 17}'>row {i}</text>")
        for j, v in enumerate(r):
            b.append(cell(x0 + j * CW, y0 + i * CW, CW - 2, str(v),
                          PURPLE if v else "#f6f6f8", "#fff" if v else MUTED))
    for j in range(S):
        b.append(f"<text class='m xs mut' x='{x0 + j*CW + 11}' y='{y0 - 6}'>{j}</text>")
    b.append(note(x0 + S * CW + 20, y0 + 20, "s = 2 ones per row"))
    b.append(note(x0 + S * CW + 20, y0 + 38, "columns = input positions", "xs mut"))
    b.append(note(20, y0 + S * CW + 34,
                  "Each row is one choice out of C(S, s) — 15 here, but C(16,8) = 12,870 and "
                  "C(32,16) ≈ 6×10⁸."))
    b.append(note(20, y0 + S * CW + 52,
                  "That count is what the model's attention has to search through, and it is "
                  "what sets difficulty."))
    return svg(700, y0 + S * CW + 74, "".join(b), "sampling the matrix A")


def lm2_token():
    """How ONE output token is produced: parity of the selected bits."""
    x0v = [1, 0, 1, 1, 0, 1]
    row = [0, 1, 0, 0, 1, 0]
    supp = [j for j, v in enumerate(row) if v]
    bits = [x0v[j] for j in supp]
    out = sum(bits) % 2
    x0, y0 = 120, 76
    b = [title(20, 28, "2 · How one output token is produced")]
    b.append(note(20, 50, "Output bit i is the XOR (parity) of the input bits that row i "
                          "of A selects — nothing else."))

    b.append(f"<text class='m s' x='{x0-46}' y='{y0+17}'>x₀</text>")
    for j, v in enumerate(x0v):
        hit = j in supp
        b.append(cell(x0 + j * CW, y0, CW - 2, str(v), BLUE if hit else WASH,
                      "#fff" if hit else MUTED))
    b.append(note(x0 + 6 * CW + 16, y0 + 17, "random input bits", "xs mut"))

    b.append(f"<text class='m s' x='{x0-46}' y='{y0+CW+29}'>A[i]</text>")
    for j, v in enumerate(row):
        b.append(cell(x0 + j * CW, y0 + CW + 12, CW - 2, str(v),
                      PURPLE if v else "#f6f6f8", "#fff" if v else MUTED))
    b.append(note(x0 + 6 * CW + 16, y0 + CW + 29, "row i of A picks positions "
                  f"{supp[0]} and {supp[1]}", "xs mut"))

    eq = " ⊕ ".join(str(v) for v in bits)
    b.append(f"<text class='m s' x='{x0-46}' y='{y0+2*CW+50}'>x₁[i]</text>")
    b.append(cell(x0, y0 + 2 * CW + 33, CW - 2, str(out), RED, "#fff"))
    b.append(f"<text class='m s' x='{x0 + CW + 12}' y='{y0+2*CW+50}'>"
             f"= {eq} = {out}</text>")
    b.append(note(x0 + CW + 190, y0 + 2 * CW + 50, "(XOR of the two selected bits)", "xs mut"))

    for j in supp:
        b.append(curve(x0 + j * CW + 11, y0 + CW - 2, x0 + 11, y0 + 2 * CW + 31,
                       BLUE, 1.4, "arb", 0.32))
    b.append(note(20, y0 + 3 * CW + 44,
                  "To predict this token the model must attend to exactly those s positions. "
                  "The correct attention pattern is"))
    b.append(note(20, y0 + 3 * CW + 62,
                  "therefore known in advance — which is what makes \"did a head find it?\" "
                  "measurable instead of a judgement call."))
    return svg(700, y0 + 3 * CW + 84, "".join(b), "producing one output token")


def lm3_sequence():
    """The flattened sequence and the next-token prediction problem."""
    S = 6
    x0v, x1v = [1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 0]
    x0, y0 = 60, 92
    b = [title(20, 28, "3 · The sequence the model actually sees")]
    b.append(note(20, 50, "x₀ and x₁ are concatenated into one sequence of S·T tokens "
                          "(T = 2). The model predicts each token from the ones before it —"))
    b.append(note(20, 68, "ordinary autoregressive next-token prediction, no separator, no "
                          "special tokens."))

    seq = x0v + x1v
    for i, v in enumerate(seq):
        first = i < S
        b.append(cell(x0 + i * CW, y0, CW - 2, str(v), WASH if first else "#fdeaea",
                      INK if first else RED))
        b.append(f"<text class='m xs mut' x='{x0 + i*CW + 11}' y='{y0 - 8}' "
                 f"text-anchor='middle'>{i}</text>")
    b.append(f"<line x1='{x0 + S*CW - 2}' y1='{y0-20}' x2='{x0 + S*CW - 2}' "
             f"y2='{y0+CW+26}' stroke='{MUTED}' stroke-dasharray='3 3'/>")
    b.append(note(x0 + 8, y0 + CW + 20, "x₀ — random", "xs mut"))
    b.append(note(x0 + S * CW + 8, y0 + CW + 20, "x₁ = A x₀ mod 2", "xs mut"))

    # prediction arrows
    for i in (2, 7):
        b.append(curve(x0 + i * CW + 11, y0 - 4, x0 + (i + 1) * CW + 11, y0 - 4,
                       BLUE if i > 5 else MUTED, 1.3, "arb" if i > 5 else "ar", 0.55))
    b.append(note(x0 + 2 * CW, y0 - 34, "predicts →", "xs mut"))

    y1 = y0 + CW + 46
    rows = [("positions 1 … S−1", "impossible — x₀ is uniform noise", "CE = ln 2 exactly", MUTED),
            ("positions S … 2S−1", "determined by A and x₀", "CE → 0 once learned", RED)]
    for i, (a, c, d, col) in enumerate(rows):
        yy = y1 + i * 24
        b.append(f"<rect x='20' y='{yy-14}' width='660' height='21' fill='"
                 f"{'#f7f8fa' if i == 0 else '#fff'}'/>")
        b.append(f"<circle cx='30' cy='{yy-3}' r='4' fill='{col}'/>")
        b.append(note(44, yy + 1, a, "m s")
                 + note(190, yy + 1, c, "s") + note(450, yy + 1, d, "m s mut"))
    b.append(note(20, y1 + 60,
                  "Half of every sequence is unpredictable by construction, so we report loss "
                  "on the second half only — a full-sequence"))
    b.append(note(20, y1 + 78,
                  "average would be half noise floor. The paper reports the full-sequence "
                  "version, whose floor is (S−1)/ST · ln C."))
    return svg(700, y1 + 100, "".join(b), "the flattened sequence")


def lm4_attention():
    """The ground-truth attention pattern, as a query x key matrix."""
    S = 6
    rows = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]]
    x0, y0 = 150, 78
    b = [title(20, 28, "4 · The attention pattern the model has to find")]
    b.append(note(20, 50, "Query at position S+i−1 (predicting token S+i) must read the s "
                          "key positions marked below. Everything else is a distractor."))
    for j in range(S):
        b.append(f"<text class='m xs mut' x='{x0 + j*CW + 11}' y='{y0-8}' "
                 f"text-anchor='middle'>{j}</text>")
    b.append(note(x0, y0 - 26, "key position (inside x₀)", "xs mut"))
    for i, r in enumerate(rows):
        b.append(f"<text class='m xs mut' x='{x0-104}' y='{y0 + i*CW + 17}'>"
                 f"query {S+i-1} → x₁[{i}]</text>")
        for j, v in enumerate(r):
            b.append(cell(x0 + j * CW, y0 + i * CW, CW - 2, "●" if v else "",
                          BLUE if v else "#fafbfc", "#fff"))
    b.append(note(x0 + S * CW + 18, y0 + 16, "s cells per row", "xs mut"))
    b.append(note(x0 + S * CW + 18, y0 + 34, "= one attention pattern", "xs mut"))
    y1 = y0 + S * CW + 34
    b.append(note(20, y1, "Training starts from near-uniform attention — every key weighted "
                          "about equally. Emergence is the moment a head"))
    b.append(note(20, y1 + 18, "moves from that to this. We score it with IoU between a "
                               "head's top-s keys and the row's true support, and we test it"))
    b.append(note(20, y1 + 36, "causally by deleting the aligned head: loss goes 0.00 → 4.23 "
                               "nats, while deleting an unaligned head costs 0.08."))
    return svg(700, y1 + 58, "".join(b), "the ground truth attention pattern")


# ═════════════════════════════════ CELLULAR AUTOMATA ═════════════════════════════════

def ca1_rule():
    """The lookup table: a 3-cell window maps to the next colour."""
    entries = [((0, 0, 0), 2), ((0, 1, 2), 1), ((1, 2, 3), 0), ((2, 0, 1), 3),
               ((3, 3, 1), 2), ((1, 0, 0), 3)]
    x0, y0 = 40, 96
    b = [title(20, 28, "5 · A cellular-automaton rule is a lookup table")]
    b.append(note(20, 50, "With C = 4 colours and a window of W = 3 cells there are 4³ = 64 "
                          "possible windows. A rule assigns a colour to each one:"))
    b.append(note(20, 74, "R : (left, centre, right) → next colour", "m s"))
    for k, (win, out) in enumerate(entries):
        gx = x0 + k * 108
        for j, v in enumerate(win):
            b.append(cell(gx + j * 22, y0, 20, str(v), COLORS4[v], FG4[v]))
        b.append(f"<text class='m s mut' x='{gx + 70}' y='{y0 + 15}'>→</text>")
        b.append(cell(gx + 84, y0, 20, str(out), COLORS4[out], FG4[out]))
    b.append(note(20, y0 + 46, "… 58 more entries. The full table is 64 numbers, drawn "
                               "uniformly at random."))
    b.append(note(20, y0 + 76, "A pool of N = 256 such rules is drawn per run — and, per the "
                               "paper's appendix, one rule is sampled PER"))
    b.append(note(20, y0 + 94, "TRAINING EXAMPLE. So the rule changes every sequence: it "
                               "cannot be memorised into the weights, it has to be"))
    b.append(note(20, y0 + 112, "inferred from the sequence itself. This task is in-context "
                                "where the linear map is in-weights."))
    return svg(700, y0 + 134, "".join(b), "the cellular automaton rule table")


def ca2_transition():
    """One state transition: slide the window, wrap at the edges."""
    st = [2, 0, 1, 3, 1, 0, 2, 1]
    nxt = [1, 3, 0, 2, 3, 1, 0, 2]
    S = len(st)
    x0, y0 = 90, 92
    b = [title(20, 28, "6 · One transition: slide the window across the state")]
    b.append(note(20, 50, "Every cell of the next state is one table lookup on its own "
                          "neighbourhood. Boundaries wrap around (ours — unstated"))
    b.append(note(20, 68, "in the paper)."))
    b.append(f"<text class='m s' x='{x0-58}' y='{y0+17}'>state t</text>")
    for j, v in enumerate(st):
        hl = j in (2, 3, 4)
        b.append(cell(x0 + j * CW, y0, CW - 2, str(v), COLORS4[v], FG4[v],
                      stroke=AMBER if hl else LINE))
    b.append(f"<rect x='{x0 + 2*CW - 3}' y='{y0-4}' width='{3*CW}' height='{CW+6}' "
             f"fill='none' stroke='{AMBER}' stroke-width='2' rx='3'/>")
    b.append(note(x0 + S * CW + 16, y0 + 17, "window W = 3", "xs mut"))

    b.append(arrow(x0 + 3 * CW + 11, y0 + CW + 6, x0 + 3 * CW + 11, y0 + CW + 30, AMBER))
    b.append(f"<text class='m xs' x='{x0 + 3*CW + 22}' y='{y0+CW+24}' fill='{AMBER}'>"
             f"R(1, 3, 1) = 2</text>")

    b.append(f"<text class='m s' x='{x0-58}' y='{y0+CW+55}'>state t+1</text>")
    for j, v in enumerate(nxt):
        hl = j == 3
        b.append(cell(x0 + j * CW, y0 + CW + 38, CW - 2, str(v), COLORS4[v], FG4[v],
                      stroke=AMBER if hl else LINE))
    y1 = y0 + 2 * CW + 60
    b.append(note(20, y1, "Composing the rule k times per transition widens the neighbourhood "
                          "a cell depends on to 2k+1 —"))
    b.append(note(20, y1 + 18, "so k is a direct knob on how wide the required attention "
                               "pattern is:"))
    for i, (k, span) in enumerate([(1, 3), (2, 5), (3, 7)]):
        yy = y1 + 40 + i * 20
        b.append(note(44, yy, f"k = {k}", "m s"))
        b.append(note(110, yy, f"span {span} cells", "s"))
        for j in range(span):
            b.append(cell(230 + j * 15, yy - 10, 13, "", TEAL if j == span // 2 else "#cfe3e3",
                          "#fff"))
        res = {1: "4 / 8 seeds solved", 2: "0 / 8", 3: "0 / 8"}[k]
        b.append(note(360, yy, res, "s mut"))
    return svg(700, y1 + 118, "".join(b), "one cellular automaton transition")


def ca3_sequence():
    """The flattened trajectory and what is predictable when."""
    S, T = 8, 5
    grid = [[2, 0, 1, 3, 1, 0, 2, 1], [1, 3, 0, 2, 3, 1, 0, 2], [3, 1, 2, 0, 1, 3, 2, 0],
            [0, 2, 3, 1, 2, 0, 1, 3], [2, 0, 1, 3, 0, 2, 3, 1]]
    x0, y0 = 70, 92
    b = [title(20, 28, "7 · The trajectory, flattened into one sequence")]
    b.append(note(20, 50, "T = 16 states of S cells are laid end to end (T = 5 shown). The "
                          "model predicts every token from the ones before it —"))
    b.append(note(20, 68, "the same objective as the linear map, on a 256-token sequence."))
    for t, rowv in enumerate(grid):
        b.append(f"<text class='m xs mut' x='{x0-46}' y='{y0 + t*(CW+3) + 17}'>state {t}</text>")
        for j, v in enumerate(rowv):
            b.append(cell(x0 + j * CW, y0 + t * (CW + 3), CW - 2, str(v), COLORS4[v], FG4[v]))
        labels = {0: ("random — unpredictable, CE = ln 4", MUTED),
                  1: ("rule still ambiguous — 256 candidates", AMBER),
                  2: ("evidence accumulating", AMBER),
                  4: ("rule identified — predictable", GREEN)}
        if t in labels:
            txt, col = labels[t]
            b.append(f"<text class='xs' x='{x0 + S*CW + 16}' y='{y0 + t*(CW+3) + 17}' "
                     f"fill='{col}'>{txt}</text>")
    y1 = y0 + T * (CW + 3) + 26
    b.append(note(20, y1, "Because the rule differs per sequence, the early states are "
                          "genuinely ambiguous — no model can predict state 1 from"))
    b.append(note(20, y1 + 18, "state 0 alone. Loss therefore has to be read per state, and "
                               "the headline metric is the FINAL state, where all the"))
    b.append(note(20, y1 + 36, "in-context evidence is available."))
    b.append(note(20, y1 + 64, "Measured at k = 1: per-state loss falls 1.298 → 0.130 across "
                               "the sequence (plateau ln 4 = 1.386). That downward", "s"))
    b.append(note(20, y1 + 82, "curve within a single sequence is the signature of in-context "
                               "learning.", "s"))
    return svg(700, y1 + 104, "".join(b), "the flattened cellular automaton trajectory")


def ca4_compare():
    """Side-by-side summary of the two tasks."""
    b = [title(20, 28, "8 · The two tasks side by side")]
    cols = [("", "linear map", "cellular automata"),
            ("what is fixed per run", "one matrix A", "a pool of 256 rules"),
            ("what varies per sequence", "the input x₀ only", "the input AND the active rule"),
            ("where the map lives", "in the weights", "in the context"),
            ("vocabulary", "C = 2", "C = 4"),
            ("sequence length", "S·T = 32", "S·T = 256"),
            ("layers used", "1", "4"),
            ("required pattern", "s specific positions", "a window of 2k+1 cells"),
            ("plateau (no knowledge)", "ln 2 = 0.693", "ln 4 = 1.386"),
            ("unpredictable by design", "the whole first half", "state 0, plus early ambiguity"),
            ("difficulty knob", "sparsity s, context S", "composition depth k")]
    y = 62
    for i, (a, c, d) in enumerate(cols):
        yy = y + i * 23
        head = i == 0
        b.append(f"<rect x='20' y='{yy-15}' width='660' height='22' fill='"
                 f"{'#eef1f6' if head else ('#f8f9fb' if i % 2 else '#fff')}'/>")
        cls = "s" if head else "s"
        b.append(f"<text class='{cls}' x='30' y='{yy}' "
                 f"{'font-weight=\"600\"' if head else 'fill=\"#666\"'}>{a}</text>")
        b.append(f"<text class='m {cls}' x='250' y='{yy}' fill='{BLUE if not head else INK}'"
                 f"{' font-weight=\"600\"' if head else ''}>{c}</text>")
        b.append(f"<text class='m {cls}' x='450' y='{yy}' fill='{PURPLE if not head else INK}'"
                 f"{' font-weight=\"600\"' if head else ''}>{d}</text>")
    y1 = y + len(cols) * 23 + 16
    b.append(note(20, y1, "Both tasks hide a sparse, known-by-construction attention pattern "
                          "behind a next-token objective. That is the"))
    b.append(note(20, y1 + 18, "whole design: make the thing the paper cares about — finding "
                               "the pattern — the only hard part of the task."))
    return svg(700, y1 + 40, "".join(b), "the two tasks compared")


if __name__ == "__main__":
    print("uploading task diagrams:")
    for name, fn in [
        ("lm1_matrix", lm1_matrix), ("lm2_token", lm2_token),
        ("lm3_sequence", lm3_sequence), ("lm4_attention", lm4_attention),
        ("ca1_rule", ca1_rule), ("ca2_transition", ca2_transition),
        ("ca3_sequence", ca3_sequence), ("ca4_compare", ca4_compare),
    ]:
        upload(f"sparse_attn_emergence_task_{name}", fn())
