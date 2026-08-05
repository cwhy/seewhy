"""
Minimal helpers for hand-authored SVG diagrams.

Every diagram paints its own white background: report pages render in the viewer's colour
scheme, and a transparent SVG with dark strokes vanishes in dark mode.
"""

BLUE, RED, GREEN, PURPLE = "#4a7ebb", "#c0504d", "#9bbb59", "#674ea7"
AMBER, TEAL = "#d9902c", "#3c8f8f"
INK, MUTED, LINE, WASH = "#1a1a1a", "#666", "#bbb", "#f4f6f9"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "system-ui, -apple-system, Segoe UI, sans-serif"

# C = 4 colour vocabulary for the cellular automata
COLORS4 = ["#e8eaf0", BLUE, RED, GREEN]
FG4 = [MUTED, "#fff", "#fff", "#fff"]


def svg(w, h, body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='100%' "
            f"role='img' aria-label='{title}'>"
            f"<defs><marker id='ar' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='5' "
            f"markerHeight='5' orient='auto-start-reverse'>"
            f"<path d='M0 0 L10 5 L0 10 z' fill='{MUTED}'/></marker>"
            f"<marker id='arb' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='5' "
            f"markerHeight='5' orient='auto-start-reverse'>"
            f"<path d='M0 0 L10 5 L0 10 z' fill='{BLUE}'/></marker></defs>"
            f"<rect width='{w}' height='{h}' fill='#ffffff'/>"
            f"<style>text{{font-family:{SANS};fill:{INK}}} .m{{font-family:{MONO}}}"
            f".s{{font-size:11px}} .xs{{font-size:10px}} .t{{font-size:13px;font-weight:600}}"
            f".mut{{fill:{MUTED}}}</style>{body}</svg>")


def title(x, y, s):
    return f"<text class='t' x='{x}' y='{y}'>{s}</text>"


def note(x, y, s, cls="s mut"):
    return f"<text class='{cls}' x='{x}' y='{y}'>{s}</text>"


def cell(x, y, w, label="", fill=WASH, fg=INK, stroke=LINE, h=None, mono=True):
    h = h or w
    out = (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}' stroke='{stroke}' "
           f"rx='2'/>")
    if label != "":
        cls = "m xs" if mono else "xs"
        out += (f"<text class='{cls}' x='{x + w/2}' y='{y + h/2 + 3.5}' fill='{fg}' "
                f"text-anchor='middle'>{label}</text>")
    return out


def arrow(x1, y1, x2, y2, color=MUTED, dash="", marker="ar", width=1.2):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
            f"stroke-width='{width}'{d} marker-end='url(#{marker})'/>")


def curve(x1, y1, x2, y2, color=MUTED, width=1.2, marker="ar", lift=0.5):
    my = y1 - abs(x2 - x1) * lift
    m = f" marker-end='url(#{marker})'" if marker else ""
    return (f"<path d='M{x1} {y1} Q{(x1+x2)/2} {my} {x2} {y2}' fill='none' stroke='{color}' "
            f"stroke-width='{width}'{m}/>")
