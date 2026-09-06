"""
Markdown-to-HTML report utilities.

save_report(name, markdown_str)  — render a markdown string and upload
save_report_file(name, path)     — render a .md file and upload
"""

from pathlib import Path
import mistletoe
from .html import save_html

_PICO_CSS_CDN = "https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"

# Pico ships `img { max-width: 100% }`, which leaves an image narrower than the
# column at its intrinsic size. Typst emits a page sized to its content — a 17cm
# figure is ~640px — so figures rendered small while matplotlib's wider SVGs
# happened not to. Figures in these reports are always meant to fill the column,
# and every one is either vector or several thousand pixels wide, so scaling to
# the column width costs nothing and fixes both cases.
_REPORT_CSS = """
main.container { max-width: 68rem; }
main.container img { width: 100%; height: auto; display: block;
                     margin: 1.25rem auto; }
main.container table { font-variant-numeric: tabular-nums; }
"""


def _render(name: str, markdown_str: str, title: str = "") -> str:
    body = mistletoe.markdown(markdown_str)
    heading = f"<h1>{title}</h1>\n" if title else ""
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>{title or name}</title>\n"
        f"  <link rel='stylesheet' href='{_PICO_CSS_CDN}'>\n"
        f"  <style>{_REPORT_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<main class='container'>\n"
        f"{heading}{body}"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


def save_report(name: str, markdown_str: str, title: str = "") -> str:
    """Render a markdown string to HTML and upload. Returns URL."""
    return save_html(name, _render(name, markdown_str, title))


def save_report_file(name: str, path: str | Path, title: str = "") -> str:
    """Render a local .md file to HTML and upload. Returns URL."""
    markdown_str = Path(path).read_text(encoding="utf-8")
    return save_html(name, _render(name, markdown_str, title))
