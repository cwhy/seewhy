"""
HTML upload utilities.

save_html(name, html_str)                      — upload an HTML string to R2
save_html_file(name, path)                     — upload a local .html file to R2
save_figures_page(name, title, figures)        — build a Tailwind figure grid and upload
"""

import io
from pathlib import Path
from .media import save_media

_TAILWIND_CDN = "https://cdn.tailwindcss.com"
_UNSET = object()  # sentinel for "auto-generate detail page"

# Responsive grid: 1 → 2 → 3 → 4 → 6 → 8 columns across breakpoints (max 8)
_GRID_CLASSES = "grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 2xl:grid-cols-8 gap-4"


def save_html(name: str, html_str: str) -> str:
    """Upload an HTML string to R2 (or local fallback). Returns URL."""
    if not name.lower().endswith(".html"):
        name = f"{name}.html"
    return save_media(name, io.BytesIO(html_str.encode("utf-8")), "text/html; charset=utf-8")


def save_html_file(name: str, path: str | Path) -> str:
    """Upload a local HTML file to R2 (or local fallback). Returns URL."""
    if not name.lower().endswith(".html"):
        name = f"{name}.html"
    return save_media(name, str(path), "text/html; charset=utf-8")


def _detail_page_html(title: str, caption: str, svg_url: str) -> str:
    """Build a full-page detail view for a single figure."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>{caption} — {title}</title>\n"
        f"  <script src='{_TAILWIND_CDN}'></script>\n"
        "</head>\n"
        "<body class='bg-white p-6'>\n"
        "  <div class='mb-4'>\n"
        "    <button onclick='history.back()' "
        "class='text-sm font-mono text-blue-600 hover:underline'>← back</button>\n"
        "  </div>\n"
        f"  <h1 class='text-xl font-mono mb-4'>{caption}</h1>\n"
        f"  <img src='{svg_url}' class='w-full' alt='{caption}'>\n"
        "</body>\n"
        "</html>\n"
    )


def save_flat_grid(name: str, title: str, figures: list[tuple[str, str]]) -> str:
    """
    Build a responsive Tailwind figure grid with no detail page links.

    Args:
        name:    Filename (with or without .html).
        title:   Page heading.
        figures: List of (caption, svg_url) pairs.

    Returns:
        str: URL of the uploaded HTML page.
    """
    back_btn = "  <div class='mb-4'><button onclick='history.back()' class='text-sm font-mono text-blue-600 hover:underline'>← back</button></div>\n"
    cards = "\n".join(
        f"    <figure class='flex flex-col gap-1'>\n"
        f"      <img src='{url}' class='w-full' loading='lazy' alt='{caption}'>\n"
        f"      <figcaption class='text-xs font-mono text-gray-500'>{caption}</figcaption>\n"
        f"    </figure>"
        for caption, url in figures
    )
    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>{title}</title>\n"
        f"  <script src='{_TAILWIND_CDN}'></script>\n"
        "</head>\n"
        f"<body class='bg-white p-6'>\n{back_btn}"
        f"  <h1 class='text-2xl font-mono mb-6'>{title}</h1>\n"
        f"  <div class='{_GRID_CLASSES}'>\n{cards}\n  </div>\n"
        "</body>\n</html>\n"
    )
    return save_html(name, html)


def save_figures_page(name: str, title: str, figures: list[tuple]) -> str:
    """
    Build a responsive Tailwind figure grid page with per-figure detail pages.

    Each entry in figures can be:
      (caption, svg_url)               — auto-generates a detail page
      (caption, svg_url, detail_url)   — uses the provided detail URL directly
      (caption, svg_url, None)         — no link (thumbnail only)

    Returns the URL of the front (grid) page.
    """
    cards = []
    for entry in figures:
        caption, svg_url = entry[0], entry[1]
        custom_detail    = entry[2] if len(entry) > 2 else _UNSET

        if custom_detail is _UNSET:
            svg_filename = svg_url.rstrip("/").rsplit("/", 1)[-1]
            detail_name  = svg_filename.rsplit(".", 1)[0] + "_detail"
            detail_url   = save_html(detail_name, _detail_page_html(title, caption, svg_url))
        else:
            detail_url = custom_detail  # may be None (no link)

        if detail_url is not None:
            img_html = (f"      <a href='{detail_url}'>\n"
                        f"        <img src='{svg_url}' class='w-full' loading='lazy' alt='{caption}'>\n"
                        f"      </a>")
        else:
            img_html = f"      <img src='{svg_url}' class='w-full' loading='lazy' alt='{caption}'>"

        cards.append(
            f"    <figure class='flex flex-col gap-1'>\n"
            f"{img_html}\n"
            f"      <figcaption class='text-xs font-mono text-gray-500'>{caption}</figcaption>\n"
            f"    </figure>"
        )

    html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>{title}</title>\n"
        f"  <script src='{_TAILWIND_CDN}'></script>\n"
        "</head>\n"
        "<body class='bg-white p-6'>\n"
        f"  <h1 class='text-2xl font-mono mb-6'>{title}</h1>\n"
        f"  <div class='{_GRID_CLASSES}'>\n"
        + "\n".join(cards) + "\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )
    return save_html(name, html)
