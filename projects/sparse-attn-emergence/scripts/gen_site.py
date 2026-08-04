"""
Publish the whole report minisite to R2.

Publishes EVERY reports/*.md in one run so cross-page relative links all resolve inside
the same dated R2 folder. Never publish one page alone — see lib/site.py.

Renders and validates first, uploads second: a body link to a page that has no .md yet
is a dead link (the nav greys those out, but hand-written prose can't know), so the run
aborts instead of shipping it.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_site.py
"""

import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.site import PAGES, PREFIX, page_name, render   # noqa: E402
from shared_lib.html import save_html                   # noqa: E402

REPORTS = PROJECT / "reports"
available = {key for key, _, _ in PAGES if (REPORTS / f"{key}.md").exists()}
missing = [key for key, _, _ in PAGES if key not in available]

# ── render ──
pages = {}
for key, _, _ in PAGES:
    if key in available:
        pages[key] = render(key, (REPORTS / f"{key}.md").read_text(), available)

# ── validate internal links ──
dead = []
for key, html in pages.items():
    body = html.split("</nav>", 1)[-1]          # nav intentionally names pending pages
    for href in re.findall(rf"href='({re.escape(PREFIX)}[a-z0-9_]+\.html)'", body):
        target = href[len(PREFIX) : -len(".html")]
        if target not in available:
            dead.append((key, href))

if dead:
    print("\nABORTED — body links to pages that do not exist yet:\n")
    for src, href in dead:
        print(f"  reports/{src}.md  →  {href}")
    print("\nDrop the link (or write the page) and re-run.\n")
    sys.exit(1)

# ── publish ──
urls = {key: save_html(page_name(key), html) for key, html in pages.items()}

print(f"\npublished {len(urls)} page(s); {len(missing)} pending: {', '.join(missing) or '—'}\n")
for key, url in urls.items():
    print(f"  {key:<8} {url}")
hub = urls.get("index", "(no index.md)")
print(f"\nhub → {hub}")
print(f"\nreminder: R2 keys are date-foldered, so this run minted NEW urls. Update the\n"
      f"sparse-attn-emergence row in projects/index.md to:\n  {hub}\n")
