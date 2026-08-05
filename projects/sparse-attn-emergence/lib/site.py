"""
Report minisite — a set of cross-linked HTML pages on R2.

R2 keys are date-foldered (`seewhy/yy-mm-dd/<name>`), so pages published in the SAME
run share a folder and can link to each other with plain relative hrefs.

The one rule that follows, and it matters: **never publish a single page on its own.**
Always republish the whole site (`scripts/gen_site.py`), or yesterday's pages will link
to siblings that live in yesterday's folder while the new page sits in today's.

Page content lives in `reports/<key>.md` — committed, diffable markdown. A page with no
`.md` yet still appears in the nav, greyed out, so the site shows its own roadmap.
"""

from datetime import datetime

import mistletoe

# key, nav label, page title
PAGES = [
    ("index",   "Overview",   "Emergent capabilities from sparse attention — a small-scale replication"),
    ("paper",   "The paper",  "The paper in plain terms — what it claims, and why it matters"),
    ("tasks",   "Task setup", "Task setup — how every token is produced, drawn out"),
    ("methods", "Methods",    "Methods — tasks, model, metrics, deviations"),
    ("exp1",    "exp1 · H1",  "exp1 — is emergence abrupt, and is its timing seed-random?"),
    ("exp2",    "exp2 · H2",  "exp2 — the sparsity × context-length difficulty window"),
    ("exp3",    "exp3 · H4",  "exp3 — heads versus head dimension"),
    ("exp4",    "exp4 · H3",  "exp4 — is the loss jump the attention pattern being found?"),
    ("exp5",    "exp5 · CA",  "exp5 — cellular automata, in context"),
    ("exp67",   "exp6/7 · H5", "exp6 & exp7 — mixer versus transformer, and what masking is worth"),
    ("findings", "Findings",  "Findings — the whole replication in one place"),
    ("mistakes", "Mistakes",  "Mistakes — what went wrong, how it surfaced, what it cost"),
]
PREFIX = "sparse_attn_emergence_"

_CSS = """
:root { --pico-font-size: 100%; }
main.container { max-width: 62rem; }
nav.site { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem;
  margin-bottom: 1.5rem; padding-bottom: .6rem; border-bottom: 1px solid var(--pico-muted-border-color); }
nav.site a, nav.site span { margin-right: .1rem; }
nav.site .here { font-weight: 700; text-decoration: underline; }
nav.site .pending { opacity: .38; }
nav.site .sep { opacity: .3; margin: 0 .35rem; }
img { max-width: 100%; height: auto; background: #fff; border-radius: 4px; }
table { font-size: .82rem; }
table td, table th { padding: .28rem .5rem; }
code { font-size: .82em; }
blockquote { border-left: 3px solid var(--pico-primary); }
.foot { margin-top: 3rem; padding-top: .8rem; border-top: 1px solid var(--pico-muted-border-color);
  font-size: .78rem; opacity: .7; display: flex; justify-content: space-between; gap: 1rem; }
"""


def page_name(key: str) -> str:
    return f"{PREFIX}{key}"


def page_file(key: str) -> str:
    return f"{PREFIX}{key}.html"


def _nav(current: str, available: set) -> str:
    bits = []
    for key, label, _ in PAGES:
        if key == current:
            bits.append(f"<span class='here'>{label}</span>")
        elif key in available:
            bits.append(f"<a href='{page_file(key)}'>{label}</a>")
        else:
            bits.append(f"<span class='pending' title='not run yet'>{label}</span>")
    return "<nav class='site'>" + "<span class='sep'>·</span>".join(bits) + "</nav>"


def _prev_next(current: str, available: set) -> str:
    order = [k for k, _, _ in PAGES if k in available]
    i = order.index(current)
    prev = f"<a href='{page_file(order[i-1])}'>← {order[i-1]}</a>" if i > 0 else "<span></span>"
    nxt = f"<a href='{page_file(order[i+1])}'>{order[i+1]} →</a>" if i < len(order) - 1 else "<span></span>"
    return f"{prev}{nxt}"


def render(key: str, body_md: str, available: set) -> str:
    title = dict((k, t) for k, _, t in PAGES)[key]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"  <title>{title}</title>\n"
        "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css'>\n"
        f"  <style>{_CSS}</style>\n"
        "</head>\n<body>\n<main class='container'>\n"
        f"{_nav(key, available)}\n"
        f"{mistletoe.markdown(body_md)}"
        f"<div class='foot'>{_prev_next(key, available)}</div>\n"
        f"<div class='foot'><span>sparse-attn-emergence · seewhy</span>"
        f"<span>generated {stamp}</span></div>\n"
        "</main>\n</body>\n</html>\n"
    )
