"""
Diagram: how retrieval actually works in the token-level model, and why the same
mechanism cannot answer a generalisation query.

Usage (server): uv run python projects/universal-ar/scripts/gen_diagram_retrieval.py
"""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_media
from shared_lib.report import save_report

C = dict(pos="#bcd7f0", pos_s="#5a8fc0", val="#c6e6c9", val_s="#5aa564",
         ref="#f2d4a8", ref_s="#c9924a", refb="#d9c7f0", refb_s="#8f79c8",
         mask="#f6c3c0", mask_s="#d0453f", ok="#1e8449", bad="#c0392b",
         box="#f4f7fb", box_s="#c3ccd6", txt="#1c2733", mut="#67727e", hi="#fff6d5")

P = []
def T(x, y, s, sz=12.5, col=None, w="normal", anc="start"):
    P.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col or C["txt"]}" font-weight="{w}" text-anchor="{anc}">{s}</text>')
def R(x, y, w, h, fill, stroke, rx=5, sw=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def L(x1, y1, x2, y2, col=None, dash="", arrow=True, sw=1.4):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a = ' marker-end="url(#ar)"' if arrow else ""
    P.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col or C["mut"]}" stroke-width="{sw}"{d}{a}/>')

def token(x, y, pos, val, ref, refc, masked=False, hl=None, scale=1.0):
    """A 3-field token chip: pos | value | ref."""
    w, h = 112 * scale, 46 * scale
    R(x, y, w, h, C["hi"] if hl == "hi" else C["box"], hl if hl and hl != "hi" else C["box_s"],
      6, 2.5 if hl else 1)
    fw = (w - 14) / 2
    R(x + 5, y + 5, fw, 15, C["pos"], C["pos_s"], 3)
    T(x + 5 + fw/2, y + 16, pos, 9.5 * scale, anc="middle")
    R(x + 9 + fw, y + 5, fw, 15, C["mask"] if masked else C["val"], C["mask_s"] if masked else C["val_s"], 3)
    T(x + 9 + fw + fw/2, y + 16, val, 9.5 * scale, C["bad"] if masked else C["txt"],
      "bold" if masked else "normal", "middle")
    R(x + 5, y + 24, w - 10, 15, refc[0], refc[1], 3)
    T(x + w/2, y + 35, ref, 9.5 * scale, anc="middle")

W, H = 1180, 1000
P.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">')
P.append(f'<rect width="{W}" height="{H}" fill="white"/>')
P.append('<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["mut"]}"/></marker>'
         '<marker id="arg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["ok"]}"/></marker>'
         '<marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["bad"]}"/></marker></defs>')

T(28, 34, "How retrieval works — and why the same mechanism cannot generalise", 20, w="bold")
T(28, 58, "Attention matches a query against every context token by dot product. What the query can address determines what it can answer.", 12.5, C["mut"])

# ── token anatomy ──
T(28, 92, "A token is a sum of three field embeddings", 14, w="bold")
token(30, 104, "pos p", "value v", "ref s", (C["ref"], C["ref_s"]))
T(155, 122, "emb(token) =  pos_emb[p]  +  val_emb[v]  +  ref_emb[s]", 13)
T(155, 141, "The ADDRESS (pos, ref) and the CONTENT (value) live in the same vector — additively.", 12, C["mut"])
T(155, 158, "A query masks the value: val_emb[MASK]. Its address fields are unchanged.", 12, C["mut"])

# ══════════ PANEL A: RETRIEVAL ══════════
y0 = 196
R(20, y0, W - 40, 300, "#f7fbf8", C["ok"], 10, 2)
T(40, y0 + 26, "A · RETRIEVAL  —  the answer is IN the context", 16, C["ok"], "bold")
T(40, y0 + 46, "Query: “what is the value at position p₇ of sample 🟠?”   →  token (p₇, MASK, 🟠)", 12.5)

# context row
T(40, y0 + 76, "context tokens", 11.5, C["mut"])
ctx = [(40, "p₃", "v=12", "🟠", (C["ref"], C["ref_s"]), None),
       (168, "p₇", "v=27", "🟠", (C["ref"], C["ref_s"]), C["ok"]),
       (296, "p₇", "v=04", "🟣", (C["refb"], C["refb_s"]), None),
       (424, "p₉", "v=27", "🟠", (C["ref"], C["ref_s"]), None),
       (552, "p₂", "v=31", "🟣", (C["refb"], C["refb_s"]), None)]
for x, p_, v_, r_, rc, hl in ctx:
    token(x, y0 + 84, p_, v_, r_, rc, hl=hl)
T(168 + 56, y0 + 146, "▲ the unique match", 10.5, C["ok"], "bold", "middle")

# query
T(760, y0 + 76, "query token", 11.5, C["bad"])
token(760, y0 + 84, "p₇", "MASK", "🟠", (C["ref"], C["ref_s"]), masked=True, hl=C["bad"])

# match table
T(40, y0 + 176, "dot-product against each context token — the address fields decide:", 12.5)
rows = [("p₃,🟠", "pos ✗   ref ✓", "partial", C["mut"]),
        ("p₇,🟠", "pos ✓   ref ✓", "MATCH — unique", C["ok"]),
        ("p₇,🟣", "pos ✓   ref ✗", "partial", C["mut"]),
        ("p₉,🟠", "pos ✗   ref ✓", "partial", C["mut"])]
for i, (a, b, c, col) in enumerate(rows):
    yy = y0 + 200 + i * 22
    T(56, yy, a, 11.5, col, "bold" if col == C["ok"] else "normal")
    T(130, yy, b, 11.5, col)
    T(250, yy, c, 11.5, col, "bold" if col == C["ok"] else "normal")

L(700, y0 + 232, 770, y0 + 232, C["ok"]); P[-1] = P[-1].replace('url(#ar)', 'url(#arg)')
R(778, y0 + 210, 172, 46, "#e8f6ed", C["ok"], 7, 2)
T(864, y0 + 231, "copy value → 27", 13, C["ok"], "bold", "middle")
T(864, y0 + 247, "ONE attention hop", 10.5, C["mut"], anc="middle")
T(978, y0 + 232, "measured: 1.000", 14, C["ok"], "bold")
T(978, y0 + 250, "(pixel and label, both digit pairs)", 10.5, C["mut"])

T(40, y0 + 290, "Why it works: the address (pos, ref) is present additively in BOTH the query and the answer token, so a single dot product isolates exactly one token.", 12, C["mut"])

# ══════════ PANEL B: GENERALISATION ══════════
y1 = 520
R(20, y1, W - 40, 430, "#fdf7f7", C["bad"], 10, 2)
T(40, y1 + 26, "B · GENERALISATION  —  the answer is NOT in the context", 16, C["bad"], "bold")
T(40, y1 + 46, "Query: “what is the label of sample 🔴?”   →  token (p_label, MASK, 🔴).  Sample 🔴's own label is never given.", 12.5)

T(40, y1 + 78, "context: each support sample has a label token, but all of them share the SAME position and none share the query's ref", 11.5, C["mut"])
sup = [(40, "p_label", "lab=A", "🟠", (C["ref"], C["ref_s"])),
       (168, "p_label", "lab=B", "🟣", (C["refb"], C["refb_s"])),
       (296, "p_label", "lab=A", "🟢", ("#bfe6de", "#4fa895")),
       (424, "p_label", "lab=B", "🔵", ("#cfe2f3", "#5a8fc0"))]
for x, p_, v_, r_, rc in sup:
    token(x, y1 + 88, p_, v_, r_, rc)
T(760, y1 + 78, "query token", 11.5, C["bad"])
token(760, y1 + 88, "p_label", "MASK", "🔴", ("#f5cccc", "#c0392b"), masked=True, hl=C["bad"])

T(40, y1 + 156, "the same address lookup now FAILS to disambiguate:", 12.5, C["bad"], "bold")
T(56, y1 + 178, "pos ✓ for every support label token   —   ref ✗ for every one of them (🔴 appears nowhere)", 12, C["mut"])
T(56, y1 + 196, "⇒ the address matches ALL of them equally. Attention has nothing to break the tie.", 12, C["bad"])

# required extra hops
T(40, y1 + 230, "so the answer requires two extra hops the address cannot provide:", 12.5, w="bold")
hops = [(56, "HOP 1  aggregate", "summarise 🔴's own pixel tokens,\nand each support sample's pixels,\ninto per-sample representations", C["mut"]),
        (400, "HOP 2  compare", "score similarity between 🔴 and each\nsupport summary → pick the closest", C["bad"]),
        (744, "HOP 3  copy", "copy that sample's label token\n(this part is the retrieval of panel A)", C["ok"])]
for x, title, body, col in hops:
    R(x, y1 + 244, 320, 86, "white", col, 7, 1.6)
    T(x + 12, y1 + 264, title, 12.5, col, "bold")
    for j, ln in enumerate(body.split("\n")):
        T(x + 12, y1 + 282 + j * 15, ln, 11, C["mut"])
L(376, y1 + 287, 400, y1 + 287, C["mut"])
L(720, y1 + 287, 744, y1 + 287, C["mut"])

T(40, y1 + 356, "HOP 2 is where it breaks.", 13.5, C["bad"], "bold")
T(200, y1 + 356, "Aggregation is a SUM of token embeddings, and with additive fields the sum is  Σ pos_emb + Σ val_emb  —", 12)
T(40, y1 + 374, "positions and values are summed separately, so a summary records “which positions were seen” and “which values occurred”, never “which value at which position”.", 12)

T(40, y1 + 402, "0 vs 1  →  differ in the VALUE MARGINAL (a 1 has far less ink) → survives the sum →", 12, C["mut"])
T(600, y1 + 402, "0.992", 13, C["ok"], "bold")
T(650, y1 + 402, "· 4 vs 9  →  differ only in the CONJUNCTION → destroyed →", 12, C["mut"])
T(1010, y1 + 402, "0.492 (chance)", 13, C["bad"], "bold")
T(40, y1 + 420, "Caveat: this mechanism was our best explanation, and a conjunctive MLP embedding built to fix it FAILED and regressed 0v1 to 0.523 — so HOP 2's true obstacle is still open.", 11, C["mut"])

P.append("</svg>")
svg = "".join(P)
url = save_media("universal-ar_retrieval_mechanism.svg", io.BytesIO(svg.encode()), "image/svg+xml")

md = f"""# Universal-AR — how retrieval works, and why generalisation doesn't

![retrieval mechanism]({url})

## The short version

Both tasks are answered by the *same* machinery: mask a token's value, let attention
find relevant context, copy. The difference is entirely in **what the query can
address**.

**Retrieval** — query `(p₇, MASK, 🟠)`. A token with exactly that address,
`(p₇, v=27, 🟠)`, is sitting in the context. Because the address fields are present
additively in both, one dot product isolates that single token and its value is
copied. **One hop. Measured accuracy 1.000** — pixels and labels, easy and hard
digit pairs alike.

**Generalisation** — query `(p_label, MASK, 🔴)`. Sample 🔴's label is *never* in the
context. Every support sample does have a label token, but they all share the same
position `p_label`, and none share 🔴's ref. So the address matches **all of them
equally** and attention has nothing to break the tie.

Answering therefore needs hops the address cannot supply:

| hop | what it must do | status |
|---|---|---|
| 1 · aggregate | summarise 🔴's pixels, and each support sample's pixels | works |
| 2 · **compare** | score 🔴 against each summary, pick the closest | **fails** |
| 3 · copy | copy the winner's label token | works (this *is* retrieval) |

Hops 1 and 3 are demonstrably fine — retrieval is perfect. **Hop 2 is the entire
failure.**

## Why hop 2 is hard here

Aggregation in a transformer is a weighted **sum** of token embeddings, and with
additive fields that sum is `Σ pos_emb + Σ val_emb`. Positions and values are summed
*separately*, so a per-sample summary can record which positions were observed and
which values occurred — but never **which value at which position**.

- **0 vs 1** differ in the value marginal (a 1 has far less ink) → survives the sum
  → **0.992**
- **4 vs 9** differ only in the conjunction → destroyed by the sum → **0.492**, exactly
  chance

## Important caveat

That explanation is *not confirmed*. It predicted a conjunctive token embedding
(`MLP(concat[pos, val, ref])`) would fix 4v9 — instead 4v9 stayed at chance **and the
0v1 control regressed from 0.992 to 0.523**. The regression does confirm the model
was leaning on a linear value-marginal readout, but the true obstacle at hop 2
remains open.

What is solid: **hop 2 is the failing step**, it is unmoved by depth, shared
positions, 20-shot support, retrieval-only training data, or conjunctive embeddings,
and the natural next suspect is the comparison *operation* itself — attention returns
a weighted sum and has no primitive for "compare my value at position p with yours at
position p, then pool agreement grouped by ref".
"""
rep = save_report("universal-ar_retrieval_mechanism", md)
print("DIAGRAM:", url)
print("REPORT:", rep)
