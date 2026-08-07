"""
Python → Typst plotting, on top of gribouille (grammar of graphics).

Three layers, each usable on its own:

  values   `typ()` serialises a Python value to a Typst literal; `Raw` marks a
           string that is already Typst code. `pt`/`cm`/`rgb`/… build literals.

  grammar  thin wrappers over gribouille's API — `aes()`, `geom_line()`,
           `scale_continuous()`, `theme_minimal()`, … Each returns `Raw` Typst
           source, so they nest exactly like the Typst calls do. Anything not
           wrapped here is reachable with `g("geom-violin", ...)`.

  charts   `line_chart()`, `scatter_chart()`, `heatmap()`, `bar_chart()` return
           a `Figure`: a column-store of data plus the `plot(...)` call reading
           it. These cover the plots this repo actually makes; drop to the
           grammar layer for anything else.

`write_figures(report_dir, figures)` materialises figures into a report tree —
data to `assets/<name>.json`, source to `figures/<name>.typ` — so a section can
place one with

    #fig(include "/figures/<name>.typ", caption: [Learning curves.])

Data is always a *column store*: `{"step": [...], "acc": [...]}`, which is both
what `json()` yields in Typst and what gribouille's `_normalise-data` accepts.
Long (tidy) form is the norm — one row per observation, a column naming the
series. `long_form()` converts the wide dict-of-series shape into it.

Non-finite floats (NaN/inf) are written as `null`, which Typst reads as `none`
— gribouille's own missing-value convention.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

GRIBOUILLE = "@preview/gribouille:0.6.0"

# A colourblind-safe qualitative ramp (Okabe-Ito, reordered so the first two
# entries are the highest-contrast pair — most plots here have two series).
PALETTE = (
    "#0072b2", "#d55e00", "#009e73", "#cc79a7",
    "#e69f00", "#56b4e9", "#f0e442", "#000000",
)

__all__ = [
    "GRIBOUILLE", "PALETTE", "Raw", "Figure",
    "typ", "call", "pretty_call", "g",
    "pt", "cm", "mm", "em", "pct", "rgb", "luma",
    "aes", "labels", "scales", "annotate",
    "geom_point", "geom_line", "geom_step", "geom_area", "geom_ribbon",
    "geom_tile", "geom_col", "geom_bar", "geom_histogram", "geom_boxplot",
    "geom_smooth", "geom_errorbar", "geom_text", "geom_label",
    "geom_hline", "geom_vline", "geom_abline",
    "scale_continuous", "scale_discrete", "scale_log10", "scale_manual",
    "scale_viridis_c", "scale_viridis_d", "scale_okabe_ito", "scale_gradient",
    "theme_minimal", "theme_bw", "theme_classic", "theme_light", "theme_void",
    "coord_flip", "coord_cartesian", "facet_wrap", "facet_grid",
    "long_form", "line_chart", "scatter_chart", "heatmap", "bar_chart",
    "write_figures",
]


# ───────────────────────────── value serialisation ──────────────────────────


class Raw(str):
    """A string emitted verbatim as Typst code instead of being quoted."""

    __slots__ = ()


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _num(v: Any) -> str:
    """Format a number as a plain Typst numeric literal (never exponential)."""
    if isinstance(v, (bool, np.bool_)):
        raise TypeError("bool is not a Typst number")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"non-finite number {f!r} has no Typst literal")
    s = repr(f)
    if "e" in s or "E" in s:
        s = f"{f:.12f}".rstrip("0").rstrip(".") or "0"
    return s


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _key(k: str) -> str:
    """A Typst dictionary key: bare identifier when it can be, else quoted."""
    return k if _IDENT.match(k) else _quote(k)


def _kw(k: str) -> str:
    """Python kwarg → Typst named argument (`inherit_aes` → `inherit-aes`).

    A single trailing underscore is stripped first so Python keywords can be
    passed (`type_` → `type`).
    """
    return k.rstrip("_").replace("_", "-") if k.endswith("_") else k.replace("_", "-")


def typ(v: Any) -> str:
    """Serialise a Python value to a Typst literal."""
    if isinstance(v, Raw):
        return str(v)
    if v is None:
        return "none"
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _num(v)
    if isinstance(v, str):
        return _quote(v)
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, Mapping):
        if not v:
            return "(:)"
        return "(" + ", ".join(f"{_key(str(k))}: {typ(x)}" for k, x in v.items()) + ")"
    if isinstance(v, (list, tuple)):
        items = [typ(x) for x in v]
        if len(items) == 1:          # a 1-tuple needs the trailing comma
            return f"({items[0]},)"
        return "(" + ", ".join(items) + ")"
    raise TypeError(f"cannot serialise {type(v).__name__} to Typst")


def call(_name: str, /, *args: Any, **kwargs: Any) -> Raw:
    """Build a single-line Typst call.

    `call("aes", x="step")` → `aes(x: "step")`; positional arguments come first,
    as in `call("annotate", "text", x=1)` → `annotate("text", x: 1)`.
    """
    parts = [typ(a) for a in args]
    parts += [f"{_kw(k)}: {typ(v)}" for k, v in kwargs.items()]
    return Raw(f"{_name}({', '.join(parts)})")


def _pretty(v: Any, indent: int) -> str:
    """Like `typ()`, but arrays of two or more items break across lines."""
    if isinstance(v, Raw) or isinstance(v, str):
        return typ(v)
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)) and len(v) > 1:
        pad = "  " * indent
        inner = ",\n".join(f"{pad}  {_pretty(x, indent + 1)}" for x in v)
        return f"(\n{inner},\n{pad})"
    return typ(v)


def pretty_call(name: str, kwargs: Mapping[str, Any]) -> str:
    """Multi-line Typst call — one named argument per line, arrays expanded."""
    if not kwargs:
        return f"{name}()"
    body = "".join(f"  {_kw(k)}: {_pretty(v, 1)},\n" for k, v in kwargs.items())
    return f"{name}(\n{body})"


# ───────────────────────────── grammar wrappers ─────────────────────────────


def g(name: str, /, *args: Any, **kwargs: Any) -> Raw:
    """Escape hatch for any gribouille function this module does not wrap."""
    return call(name, *args, **kwargs)


def pt(v: float) -> Raw: return Raw(f"{_num(v)}pt")
def cm(v: float) -> Raw: return Raw(f"{_num(v)}cm")
def mm(v: float) -> Raw: return Raw(f"{_num(v)}mm")
def em(v: float) -> Raw: return Raw(f"{_num(v)}em")
def pct(v: float) -> Raw: return Raw(f"{_num(v)}%")
def rgb(hex_colour: str) -> Raw: return Raw(f"rgb({_quote(hex_colour)})")
def luma(v: float) -> Raw: return Raw(f"luma({_num(v)})")


def aes(**kw: Any) -> Raw: return call("aes", **kw)
def labels(**kw: Any) -> Raw: return call("labels", **kw)
def scales(**kw: Any) -> Raw: return call("scales", **kw)
def annotate(**kw: Any) -> Raw: return call("annotate", **kw)

def geom_point(**kw: Any) -> Raw: return call("geom-point", **kw)
def geom_line(**kw: Any) -> Raw: return call("geom-line", **kw)
def geom_step(**kw: Any) -> Raw: return call("geom-step", **kw)
def geom_area(**kw: Any) -> Raw: return call("geom-area", **kw)
def geom_ribbon(**kw: Any) -> Raw: return call("geom-ribbon", **kw)
def geom_tile(**kw: Any) -> Raw: return call("geom-tile", **kw)
def geom_col(**kw: Any) -> Raw: return call("geom-col", **kw)
def geom_bar(**kw: Any) -> Raw: return call("geom-bar", **kw)
def geom_histogram(**kw: Any) -> Raw: return call("geom-histogram", **kw)
def geom_boxplot(**kw: Any) -> Raw: return call("geom-boxplot", **kw)
def geom_smooth(**kw: Any) -> Raw: return call("geom-smooth", **kw)
def geom_errorbar(**kw: Any) -> Raw: return call("geom-errorbar", **kw)
def geom_text(**kw: Any) -> Raw: return call("geom-text", **kw)
def geom_label(**kw: Any) -> Raw: return call("geom-label", **kw)
def geom_hline(**kw: Any) -> Raw: return call("geom-hline", **kw)
def geom_vline(**kw: Any) -> Raw: return call("geom-vline", **kw)
def geom_abline(**kw: Any) -> Raw: return call("geom-abline", **kw)

def scale_continuous(**kw: Any) -> Raw: return call("scale-continuous", **kw)
def scale_discrete(**kw: Any) -> Raw: return call("scale-discrete", **kw)
def scale_log10(**kw: Any) -> Raw: return call("scale-log10", **kw)
def scale_manual(**kw: Any) -> Raw: return call("scale-manual", **kw)
def scale_viridis_c(**kw: Any) -> Raw: return call("scale-viridis-c", **kw)
def scale_viridis_d(**kw: Any) -> Raw: return call("scale-viridis-d", **kw)
def scale_okabe_ito(**kw: Any) -> Raw: return call("scale-okabe-ito", **kw)
def scale_gradient(**kw: Any) -> Raw: return call("scale-gradient", **kw)

def theme_minimal(**kw: Any) -> Raw: return call("theme-minimal", **kw)
def theme_bw(**kw: Any) -> Raw: return call("theme-bw", **kw)
def theme_classic(**kw: Any) -> Raw: return call("theme-classic", **kw)
def theme_light(**kw: Any) -> Raw: return call("theme-light", **kw)
def theme_void(**kw: Any) -> Raw: return call("theme-void", **kw)

def coord_flip(**kw: Any) -> Raw: return call("coord-flip", **kw)
def coord_cartesian(**kw: Any) -> Raw: return call("coord-cartesian", **kw)
def facet_wrap(**kw: Any) -> Raw: return call("facet-wrap", **kw)
def facet_grid(**kw: Any) -> Raw: return call("facet-grid", **kw)


# ───────────────────────────────── data ─────────────────────────────────────


def _column(values: Any) -> list:
    """Coerce one column to a JSON-safe list; non-finite floats become `None`."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out = arr.ravel().tolist()
    if arr.dtype.kind == "f":
        return [None if not math.isfinite(v) else float(v) for v in out]
    if arr.dtype.kind in "OUS":
        return [None if v is None else str(v) for v in out]
    return out


def _columns(data: Mapping[str, Any]) -> dict[str, list]:
    cols = {str(k): _column(v) for k, v in data.items()}
    lengths = {len(v) for v in cols.values()}
    if len(lengths) > 1:
        shape = {k: len(v) for k, v in cols.items()}
        raise ValueError(f"columns must be equal length, got {shape}")
    return cols


def long_form(
    x: Sequence[Any],
    series: Mapping[str, Sequence[Any]],
    *,
    x_name: str = "x",
    y_name: str = "y",
    series_name: str = "series",
) -> dict[str, list]:
    """Wide → long. `{"train": [...], "test": [...]}` over a shared `x` becomes
    three columns (`x`, `y`, `series`) — the shape gribouille maps to aesthetics.
    """
    xs = _column(x)
    out: dict[str, list] = {x_name: [], y_name: [], series_name: []}
    for label, ys in series.items():
        col = _column(ys)
        if len(col) != len(xs):
            raise ValueError(
                f"series {label!r} has {len(col)} points but x has {len(xs)}"
            )
        out[x_name].extend(xs)
        out[y_name].extend(col)
        out[series_name].extend([str(label)] * len(col))
    return out


# ──────────────────────────────── figures ───────────────────────────────────


@dataclass(frozen=True)
class Figure:
    """A plot: its data (column store) plus the `plot(...)` call reading it.

    `spec` holds every `plot()` argument except `data`, which is supplied at
    render time so the same figure can point at whatever asset path the report
    tree uses.
    """

    name: str
    data: dict[str, list] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    alt: str = ""

    def typst(self, data_expr: str) -> str:
        """Render the `plot(...)` call, reading data from `data_expr`."""
        spec = dict(self.spec)
        if self.alt:
            spec["alt"] = self.alt
        return pretty_call("plot", {"data": Raw(data_expr), **spec})


def _plot_spec(
    *,
    mapping: Raw,
    layers: Sequence[Raw],
    scale_args: Mapping[str, Raw],
    label_args: Mapping[str, Any],
    theme: Raw,
    width: Raw,
    height: Raw,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    spec: dict[str, Any] = {"mapping": mapping, "layers": list(layers)}
    if scale_args:
        spec["scales"] = scales(**scale_args)
    labs = {k: v for k, v in label_args.items() if v is not None}
    if labs:
        spec["labels"] = labels(**labs)
    spec["theme"] = theme
    spec["width"] = width
    spec["height"] = height
    spec.update(extra)
    return spec


def _discrete_colour(levels: Sequence[str], palette: Sequence[str] | None) -> Raw:
    colours = list(palette or PALETTE)
    if len(colours) < len(levels):                     # recycle rather than fail
        colours = (colours * (len(levels) // len(colours) + 1))[: len(levels)]
    return scale_discrete(
        limits=list(levels),
        palette=[rgb(c) for c in colours[: len(levels)]],
    )


def _span(values: Sequence[Any] | None) -> tuple[float, float] | None:
    """Numeric (min, max) of a column, or None if it has no numeric entries."""
    if not values:
        return None
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return (min(nums), max(nums)) if nums else None


def _reference_lines(
    hlines: Sequence[float | tuple[float, str]],
    x_values: Sequence[Any] | None = None,
    y_span: tuple[float, float] | None = None,
) -> list[Raw]:
    """Dashed horizontal reference lines, each optionally labelled.

    An entry is either a bare value or a `(value, label)` pair. Labelling needs
    both axes: an x to anchor to and a y range to offset by, since gribouille's
    text nudges are in data units. Without a numeric x — a categorical axis, say
    — the label is dropped rather than placed somewhere wrong.
    """
    x = _span(x_values)
    lift = 0.028 * (y_span[1] - y_span[0]) if y_span and y_span[1] > y_span[0] else None

    layers: list[Raw] = []
    for entry in hlines:
        value, label = entry if isinstance(entry, tuple) else (entry, None)
        layers.append(
            geom_hline(yintercept=float(value), linetype="dashed",
                       colour=luma(130), stroke=pt(0.6))
        )
        if label and x is not None and lift is not None:
            # A one-row layer with its own data, rather than `annotate`: that
            # routes every field through aesthetic-splitting, so `colour` there
            # means "a column to map through the colour scale", not a colour.
            layers.append(
                geom_text(
                    data={
                        "x": [x[0] + 0.88 * (x[1] - x[0])],
                        "y": [float(value)],
                        "label": [str(label)],
                    },
                    mapping=aes(x="x", y="y", label="label"),
                    size=pt(7), colour=luma(110), nudge_y=lift,
                    inherit_aes=False,
                )
            )
    return layers


def _levels(data: Mapping[str, list], column: str) -> list[str]:
    """Distinct values of a column, in first-appearance order."""
    seen: dict[str, None] = {}
    for v in data[column]:
        seen.setdefault(str(v), None)
    return list(seen)


def line_chart(
    name: str,
    data: Mapping[str, Any],
    *,
    x: str,
    y: str,
    colour: str | None = None,
    points: bool = False,
    size: Raw | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colour_label: str | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    log_x: bool = False,
    hlines: Sequence[float | tuple[float, str]] = (),
    palette: Sequence[str] | None = None,
    theme: Raw | None = None,
    width: Raw | None = None,
    height: Raw | None = None,
    alt: str = "",
    **plot_kwargs: Any,
) -> Figure:
    """A line chart over long-form data.

    `hlines` draws dashed reference lines — `[(0.2, "chance")]` — the annotation
    most of these experiments need. `colour` names the series column.
    """
    cols = _columns(data)
    mapping = aes(x=x, y=y, **({"colour": colour, "fill": colour} if colour else {}))

    layers: list[Raw] = [geom_line(size=size or pt(1.1))]
    if points:
        layers.append(geom_point(size=pt(2.6)))
    layers += _reference_lines(
        hlines, cols.get(x), y_limits or _span(cols.get(y))
    )

    scale_args: dict[str, Raw] = {}
    scale_args["x"] = scale_log10() if log_x else scale_continuous(
        **({"limits": [float(v) for v in x_limits]} if x_limits else {})
    )
    scale_args["y"] = scale_continuous(
        **({"limits": [float(v) for v in y_limits]} if y_limits else {})
    )
    if colour:
        levels = _levels(cols, colour)
        scale_args["colour"] = _discrete_colour(levels, palette)
        scale_args["fill"] = _discrete_colour(levels, palette)

    spec = _plot_spec(
        mapping=mapping,
        layers=layers,
        scale_args=scale_args,
        label_args={"title": title, "subtitle": subtitle, "caption": caption,
                    "x": x_label or x, "y": y_label or y,
                    "colour": colour_label, "fill": colour_label},
        theme=theme or theme_minimal(),
        width=width or cm(12),
        height=height or cm(7),
        extra=plot_kwargs,
    )
    return Figure(name=name, data=cols, spec=spec, alt=alt)


def scatter_chart(
    name: str,
    data: Mapping[str, Any],
    *,
    x: str,
    y: str,
    colour: str | None = None,
    smooth: str | None = None,
    size: Raw | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colour_label: str | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    palette: Sequence[str] | None = None,
    theme: Raw | None = None,
    width: Raw | None = None,
    height: Raw | None = None,
    alt: str = "",
    **plot_kwargs: Any,
) -> Figure:
    """A scatter plot; `smooth` (e.g. `"lm"`) adds a fitted `geom-smooth` layer."""
    cols = _columns(data)
    mapping = aes(x=x, y=y, **({"colour": colour, "fill": colour} if colour else {}))

    layers: list[Raw] = [geom_point(size=size or pt(3))]
    if smooth:
        layers.append(geom_smooth(method=smooth, se=True, alpha=0.18))

    scale_args: dict[str, Raw] = {
        "x": scale_continuous(**({"limits": [float(v) for v in x_limits]} if x_limits else {})),
        "y": scale_continuous(**({"limits": [float(v) for v in y_limits]} if y_limits else {})),
    }
    if colour:
        levels = _levels(cols, colour)
        scale_args["colour"] = _discrete_colour(levels, palette)
        scale_args["fill"] = _discrete_colour(levels, palette)

    spec = _plot_spec(
        mapping=mapping,
        layers=layers,
        scale_args=scale_args,
        label_args={"title": title, "subtitle": subtitle, "caption": caption,
                    "x": x_label or x, "y": y_label or y,
                    "colour": colour_label, "fill": colour_label},
        theme=theme or theme_minimal(),
        width=width or cm(11),
        height=height or cm(8),
        extra=plot_kwargs,
    )
    return Figure(name=name, data=cols, spec=spec, alt=alt)


def heatmap(
    name: str,
    z: Any,
    *,
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    x_name: str = "x",
    y_name: str = "y",
    z_name: str = "value",
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    z_label: str | None = None,
    theme: Raw | None = None,
    width: Raw | None = None,
    height: Raw | None = None,
    alt: str = "",
    **plot_kwargs: Any,
) -> Figure:
    """A `geom-tile` heatmap of a 2-D array indexed `z[row=y, col=x]`.

    Both axes are treated as *discrete*: a sweep over `[1, 5, 20]` shots wants
    three equal columns, not three tiles at numeric positions 1, 5 and 20 with
    gaps between them. Axis order follows `x_values`/`y_values` as given.

    NaN cells are dropped rather than drawn, so a cell that was never run reads
    as a hole in the grid instead of a dark tile at the bottom of the scale.
    """
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"heatmap expects a 2-D array, got shape {arr.shape}")
    if arr.shape != (len(y_values), len(x_values)):
        raise ValueError(
            f"z has shape {arr.shape} but axes are "
            f"({len(y_values)}, {len(x_values)})"
        )

    x_levels = [str(v) for v in x_values]
    y_levels = [str(v) for v in y_values]
    xs, ys, vs = [], [], []
    for i, yv in enumerate(y_levels):
        for j, xv in enumerate(x_levels):
            if not math.isfinite(float(arr[i, j])):
                continue
            xs.append(xv)
            ys.append(yv)
            vs.append(float(arr[i, j]))
    cols = _columns({x_name: xs, y_name: ys, z_name: vs})

    spec = _plot_spec(
        mapping=aes(x=x_name, y=y_name, fill=z_name),
        layers=[geom_tile(stroke=None)],
        scale_args={
            "x": scale_discrete(limits=x_levels),
            "y": scale_discrete(limits=y_levels),
            "fill": scale_viridis_c(),
        },
        label_args={"title": title, "subtitle": subtitle, "caption": caption,
                    "x": x_label or x_name, "y": y_label or y_name,
                    "fill": z_label or z_name},
        theme=theme or theme_minimal(),
        width=width or cm(11),
        height=height or cm(8),
        extra=plot_kwargs,
    )
    return Figure(name=name, data=cols, spec=spec, alt=alt)


def bar_chart(
    name: str,
    data: Mapping[str, Any],
    *,
    x: str,
    y: str,
    fill: str | None = None,
    x_order: Sequence[str] | None = None,
    horizontal: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    fill_label: str | None = None,
    y_limits: tuple[float, float] | None = None,
    hlines: Sequence[float | tuple[float, str]] = (),
    palette: Sequence[str] | None = None,
    theme: Raw | None = None,
    width: Raw | None = None,
    height: Raw | None = None,
    alt: str = "",
    **plot_kwargs: Any,
) -> Figure:
    """A `geom-col` bar chart of pre-aggregated values (no counting stat).

    A discrete x axis is ordered alphabetically unless `x_order` says otherwise,
    which scrambles any grouping the labels imply. Pass the categories in the
    order they should appear — usually just the order they were built in.
    """
    cols = _columns(data)
    mapping = aes(x=x, y=y, **({"fill": fill} if fill else {}))

    layers: list[Raw] = [
        geom_col(width=0.72, stroke=None,
                 **({} if fill else {"fill": rgb((palette or PALETTE)[0])}))
    ]
    layers += _reference_lines(hlines)   # x is categorical here, so no labels

    scale_args: dict[str, Raw] = {
        "y": scale_continuous(**({"limits": [float(v) for v in y_limits]} if y_limits else {}))
    }
    if x_order:
        scale_args["x"] = scale_discrete(limits=[str(v) for v in x_order])
    if fill:
        scale_args["fill"] = _discrete_colour(_levels(cols, fill), palette)

    spec = _plot_spec(
        mapping=mapping,
        layers=layers,
        scale_args=scale_args,
        label_args={"title": title, "subtitle": subtitle, "caption": caption,
                    "x": x_label or x, "y": y_label or y, "fill": fill_label},
        theme=theme or theme_minimal(),
        width=width or cm(11),
        height=height or cm(7),
        extra=plot_kwargs,
    )
    if horizontal:
        spec["coord"] = coord_flip()
    return Figure(name=name, data=cols, spec=spec, alt=alt)


# ──────────────────────────── materialisation ───────────────────────────────

_HEADER = (
    "// Generated by shared_lib.typst_plot — do not edit by hand.\n"
    "// Regenerate with the project's scripts/gen_report.py.\n"
)


def write_figures(
    report_dir: str | Path,
    figures: Sequence[Figure],
    *,
    assets_dir: str = "assets",
    figures_dir: str = "figures",
) -> list[Path]:
    """Write each figure's data and Typst source into a report tree.

    Data goes to `<report>/<assets_dir>/<name>.json`, source to
    `<report>/<figures_dir>/<name>.typ`. The source refers to its data by a
    root-relative path, so it renders the same wherever it is included from.

    Returns every path written. Files whose content is unchanged are left
    alone, so a partial regeneration only touches the figures that moved.
    """
    root = Path(report_dir)
    assets = root / assets_dir
    figs = root / figures_dir
    assets.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fig in figures:
        data_rel = f"/{assets_dir}/{fig.name}.json"
        data_path = assets / f"{fig.name}.json"
        src_path = figs / f"{fig.name}.typ"

        data_text = json.dumps(fig.data, indent=1) + "\n"
        src_text = (
            f"{_HEADER}#import {typ(GRIBOUILLE)}: *\n\n"
            f"#{fig.typst(f'json({typ(data_rel)})')}\n"
        )
        for path, text in ((data_path, data_text), (src_path, src_text)):
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
                written.append(path)
    return written
