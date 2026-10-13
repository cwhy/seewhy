"""Report figures for DUST vs EGGROLL, built on `shared_lib.typst_plot`.

Each function takes the list of result rows and returns a `Figure` -- data plus a
declarative gribouille spec -- or None when the rows it needs do not exist yet. Nothing
here writes to disk; `scripts/gen_report.py` collects them and calls `write_figures()`.

Colours follow the paper: DUST violet, EGGROLL grey, backprop black. The paper's own
published points are drawn as lighter series beside ours, never mixed with them.
"""

from lib import paper_ref as ref
from lib import results as R
from shared_lib.typst_plot import Figure, bar_chart, cm, line_chart

VIOLET, VIOLET_L = "#7c3aed", "#c4b5fd"
GREY, GREY_L = "#4b5563", "#cbd5e1"
POPS = (64, 256, 1024, 4096, 16384)


def _pop_label(p: int) -> str:
    return f"{p // 1024}k" if p >= 1024 and p % 1024 == 0 else str(p)


def ladder(rows: list[dict], budget: str, name: str | None = None) -> Figure | None:
    """Test loss against population, ours beside the paper's, with backprop and the unigram floor."""
    xs, ys, series = [], [], []
    for method, label, colour_key in (("dust", "DUST", 0), ("eggroll", "EGGROLL", 1)):
        for pop in POPS:
            c = R.cell(rows, method=method, budget=budget, pop=pop)
            if c:
                xs.append(pop); ys.append(c["mean"]); series.append(f"{label} (ours)")
        for pop in POPS:
            v = ref.TABLE1[method].get(pop, {}).get(budget)
            if v is not None:
                xs.append(pop); ys.append(v); series.append(f"{label} (paper)")
    if not xs:
        return None
    bp = R.cell(rows, method="bp", budget=budget)
    hl = [(ref.UNIGRAM_BY_BUDGET[budget], "unigram floor")]
    if bp:
        hl.append((bp["mean"], "backprop (ours)"))
    hl.append((ref.TABLE1["bp"][budget], "backprop (paper)"))
    order = ["DUST (ours)", "DUST (paper)", "EGGROLL (ours)", "EGGROLL (paper)"]
    data = {"population": xs, "test loss": ys, "method": series}
    levels = [s for s in order if s in series]
    palette = {"DUST (ours)": VIOLET, "DUST (paper)": VIOLET_L, "EGGROLL (ours)": GREY, "EGGROLL (paper)": GREY_L}
    return line_chart(
        name or f"ladder_{budget}", data, x="population", y="test loss", colour="method", points=True,
        log_x=True, x_label="population (DUST draws / EGGROLL forward passes)", y_label="test loss",
        colour_label="", hlines=hl, palette=[palette[l] for l in levels],
        width=cm(14), height=cm(8.5),
        alt=f"Test loss at {budget} tokens against population for DUST and EGGROLL, ours and the paper's, "
            "with backprop and the unigram floor as dashed lines.",
    )


LAYER_ORDER = ["head", "token_emb", "attn_out", "mlp_out", "mlp_in", "gate", "v", "q", "k", "value_emb"]
LAYER_NAME = dict(head="LM head", token_emb="token emb", attn_out="attn out", mlp_out="MLP out", mlp_in="MLP in",
                  gate="value gate", v="V", q="Q", k="K", value_emb="value emb")


def parity_cosines(rows: list[dict], name: str = "parity_cosines") -> Figure | None:
    """Cosine to backprop, authors' PyTorch DUST against this JAX port, same weights and batch."""
    row = next((r for r in rows if r.get("kind") == "parity"), None)
    if not row:
        return None
    xs, ys, impl = [], [], []
    for t in LAYER_ORDER:
        for label, key in (("authors' PyTorch", "dust_cos_reference"), ("this JAX port", "dust_cos_jax")):
            xs.append(LAYER_NAME[t]); ys.append(row[key][t]); impl.append(label)
    return bar_chart(
        name, {"layer type": xs, "cosine to backprop": ys, "implementation": impl},
        x="layer type", y="cosine to backprop", fill="implementation", position="dodge",
        x_order=[LAYER_NAME[t] for t in reversed(LAYER_ORDER)], palette=[VIOLET_L, VIOLET], horizontal=True,
        x_label="", y_label="cosine to backprop gradient", fill_label="", width=cm(12), height=cm(9),
        alt="Cosine between the DUST gradient estimate and the autograd gradient for each layer type, "
            "for the authors' PyTorch implementation and this JAX port on identical weights and batch.",
    )


def cost(rows: list[dict], budget: str = "1M", name: str | None = None) -> Figure | None:
    """Test loss against measured seconds per update, DUST and EGGROLL, one point per population."""
    xs, ys, series = [], [], []
    for method, label in (("dust", "DUST"), ("eggroll", "EGGROLL")):
        for pop in POPS:
            c = R.cell(rows, method=method, budget=budget, pop=pop)
            if c:
                xs.append(c["step_s"]); ys.append(c["mean"]); series.append(label)
    if not xs:
        return None
    bp = R.cell(rows, method="bp", budget=budget)
    hl = [(ref.UNIGRAM_BY_BUDGET[budget], "unigram floor")]
    if bp:
        hl.append((bp["mean"], f"backprop ({bp['step_s']:.2f} s/update)"))
    levels = [s for s in ("DUST", "EGGROLL") if s in series]
    palette = {"DUST": VIOLET, "EGGROLL": GREY}
    return line_chart(
        name or f"cost_{budget}", {"seconds per update": xs, "test loss": ys, "method": series},
        x="seconds per update", y="test loss", colour="method", points=True, log_x=True,
        x_label="measured seconds per update (log scale)", y_label="test loss", colour_label="",
        hlines=hl, palette=[palette[l] for l in levels], width=cm(14), height=cm(8.5),
        alt=f"Test loss at {budget} tokens against measured wall-clock seconds per update, for DUST and EGGROLL "
            "at each population, with backprop and the unigram floor as dashed lines.",
    )


def curves(rows: list[dict], budget: str = "1M", name: str | None = None) -> Figure | None:
    """Validation loss over training, mean over seeds, for the populations that have run."""
    xs, ys, series = [], [], []
    spec = [("bp", 0, "backprop"), ("dust", 256, "DUST 256"), ("dust", 1024, "DUST 1k"),
            ("eggroll", 256, "EGGROLL 256"), ("eggroll", 1024, "EGGROLL 1k")]
    for method, pop, label in spec:
        rs = R.train_rows(rows, exps=R.LADDER_EXPS, method=method, budget=budget, pop=pop)
        if not rs:
            continue
        steps = rs[0]["history"]["step"]
        for i, s in enumerate(steps):
            vals = [r["history"]["val_loss"][i] for r in rs if len(r["history"]["val_loss"]) == len(steps)]
            if vals:
                xs.append(s / max(steps)); ys.append(sum(vals) / len(vals)); series.append(label)
    if not xs:
        return None
    palette = {"backprop": "#111827", "DUST 256": VIOLET_L, "DUST 1k": VIOLET, "EGGROLL 256": GREY_L, "EGGROLL 1k": GREY}
    levels = list(dict.fromkeys(series))
    return line_chart(
        name or f"curves_{budget}", {"fraction of training": xs, "validation loss": ys, "run": series},
        x="fraction of training", y="validation loss", colour="run", x_label="fraction of the token budget consumed",
        y_label="validation loss (mean over seeds)", colour_label="", palette=[palette[l] for l in levels],
        hlines=[(ref.UNIGRAM_BY_BUDGET[budget], "unigram floor")], width=cm(14), height=cm(8.5),
        alt=f"Validation loss over training at {budget} tokens for backprop, DUST and EGGROLL.",
    )


def eggroll_tuning(rows: list[dict], name: str = "eggroll_tuning") -> Figure | None:
    """EGGROLL's tuning grid at 1M tokens: test loss against Adam rate, one line per (population, sigma, Adam)."""
    xs, ys, series = [], [], []
    for r in R.train_rows(rows, method="eggroll", budget="1M", seed=42):
        if r["experiment"] not in ("exp5", "exp8", "exp14") or r["pop"] not in (64, 256):
            continue
        adam = "paper Adam" if r["momentum"] == 0.5 else "default Adam"
        xs.append(r["lr"]); ys.append(r["test_at_best_val"]); series.append(f"{r['pop']} passes, σ={r['sigma']:g}, {adam}")
    if not xs:
        return None
    levels = sorted(set(series))
    pal = ["#9ca3af", "#4b5563", "#c4b5fd", "#7c3aed", "#fbbf24", "#d97706", "#34d399", "#059669", "#38bdf8", "#0369a1"]
    return line_chart(
        name, {"Adam learning rate": xs, "test loss": ys, "setting": series}, x="Adam learning rate", y="test loss",
        colour="setting", points=True, log_x=True, x_label="Adam learning rate (log scale)", y_label="test loss, 1M tokens",
        colour_label="", y_limits=(7.0, 8.4),
        hlines=[(ref.TABLE1["eggroll"][64]["1M"], "paper, 64"), (ref.TABLE1["eggroll"][256]["1M"], "paper, 256")],
        palette=pal[:len(levels)], width=cm(14), height=cm(9),
        alt="Test loss at 1M tokens against Adam learning rate for each EGGROLL tuning setting at 64 and 256 forward passes, "
            "with the paper's published EGGROLL losses at those populations as dashed lines.",
    )


OPT_EXPS = ("exp24", "exp25", "exp26", "exp27")


def opt_cell(rows: list[dict], method: str, pop: int, opt: str) -> dict | None:
    """Mean over seeds of test loss for one (method, population, optimiser) at 1M tokens, final runs only.

    SGD is the original ladder; EGGROLL's "Adam" is the ladder arm that was always run with Adam (paper's EGGROLL constants).
    """
    if opt == "sgd" and method == "eggroll":
        return None
    if opt == "adam" and method == "eggroll":
        return R.cell(rows, method="eggroll", budget="1M", pop=pop)
    if opt == "sgd":
        return R.cell(rows, method=method, budget="1M", pop=pop)
    return R.cell(rows, exps=OPT_EXPS, method=method, budget="1M", pop=pop, opt=opt)


def optimiser_ladder(rows: list[dict], name: str = "optimiser_ladder") -> Figure | None:
    """Test loss against population for DUST under SGD, Adam and Muon, EGGROLL under Adam and Muon, backprop as lines."""
    xs, ys, series = [], [], []
    spec = [("dust", "sgd", "DUST, SGD"), ("dust", "adam", "DUST, Adam"), ("dust", "muon", "DUST, Muon"),
            ("eggroll", "adam", "EGGROLL, Adam"), ("eggroll", "muon", "EGGROLL, Muon")]
    for method, opt, label in spec:
        for pop in (64, 256, 1024, 4096):
            c = opt_cell(rows, method, pop, opt)
            if c:
                xs.append(pop); ys.append(c["mean"]); series.append(label)
    if not xs:
        return None
    for opt in ("sgd", "adam", "muon"):                    # backprop in the legend, as a flat line between the ends of the ladder
        c = opt_cell(rows, "bp", 0, opt)
        if c:
            for pop in (64, 4096):
                xs.append(pop); ys.append(c["mean"]); series.append(f"backprop, {OPT_LABEL[opt]}")
    pal = {"DUST, SGD": "#c4b5fd", "DUST, Adam": "#7c3aed", "DUST, Muon": "#3b0764", "EGGROLL, Adam": "#d1d5db", "EGGROLL, Muon": "#6b7280",
           "backprop, SGD": "#fca5a5", "backprop, Adam": "#ef4444", "backprop, Muon": "#7f1d1d"}
    levels = list(dict.fromkeys(series))
    return line_chart(
        name, {"population": xs, "test loss": ys, "method and optimiser": series}, x="population", y="test loss",
        colour="method and optimiser", points=True, log_x=True, x_label="population (DUST draws / EGGROLL forward passes)",
        y_label="test loss, 1M tokens", colour_label="", palette=[pal[l] for l in levels], width=cm(14), height=cm(9),
        alt="Test loss at 1M tokens against population for DUST under SGD, Adam and Muon and EGGROLL under Adam and Muon, "
            "with backprop under each optimiser as a flat line.",
    )


OPT_LABEL = {"sgd": "SGD", "adam": "Adam", "muon": "Muon"}
DUST_PAL = {"sgd": "#c4b5fd", "adam": "#7c3aed", "muon": "#3b0764"}
BP_PAL = {"sgd": "#fca5a5", "adam": "#ef4444", "muon": "#7f1d1d"}


def opt_rows(rows: list[dict], method: str, pop: int, opt: str) -> list[dict]:
    """The runs behind `opt_cell`: one per seed, for one (method, population, optimiser) at 1M tokens."""
    if opt == "sgd":
        return R.train_rows(rows, exps=R.LADDER_EXPS, method=method, budget="1M", pop=pop)
    if opt == "adam" and method == "eggroll":
        return R.train_rows(rows, exps=R.LADDER_EXPS, method=method, budget="1M", pop=pop)
    return R.train_rows(rows, exps=OPT_EXPS, method=method, budget="1M", pop=pop, opt=opt)


def optimiser_zoom(rows: list[dict], name: str = "optimiser_zoom") -> Figure | None:
    """DUST under SGD, Adam and Muon against backprop under the same three, without EGGROLL so the range is readable."""
    xs, ys, series, order = [], [], [], []
    pops = (64, 256, 1024, 4096)
    for opt in ("sgd", "adam", "muon"):
        label = f"DUST, {OPT_LABEL[opt]}"
        for pop in pops:
            c = opt_cell(rows, "dust", pop, opt)
            if c:
                xs.append(pop); ys.append(c["mean"]); series.append(label)
        if label in series:
            order.append((label, DUST_PAL[opt]))
    for opt in ("sgd", "adam", "muon"):
        c = opt_cell(rows, "bp", 0, opt)
        if c:
            label = f"backprop, {OPT_LABEL[opt]}"
            for pop in (pops[0], pops[-1]):                     # a flat line: its two ends
                xs.append(pop); ys.append(c["mean"]); series.append(label)
            order.append((label, BP_PAL[opt]))
    if not xs:
        return None
    levels = list(dict.fromkeys(series))
    pal = dict(order)
    return line_chart(
        name, {"population (DUST draws)": xs, "test loss": ys, "method and optimiser": series}, x="population (DUST draws)", y="test loss",
        colour="method and optimiser", points=True, log_x=True, x_label="DUST draws per update (log scale)",
        y_label="test loss, 1M tokens", colour_label="", palette=[pal[l] for l in levels], width=cm(14), height=cm(9),
        alt="Test loss at 1M tokens against DUST population under SGD, Adam and Muon, with backprop under each as a flat line.",
    )


def optimiser_gap(rows: list[dict], name: str = "optimiser_gap") -> Figure | None:
    """DUST's test loss minus backprop's under the same optimiser, against population: zero is level."""
    xs, ys, series = [], [], []
    for opt in ("sgd", "adam", "muon"):
        bp = opt_cell(rows, "bp", 0, opt)
        for pop in (64, 256, 1024, 4096):
            c = opt_cell(rows, "dust", pop, opt)
            if c and bp:
                xs.append(pop); ys.append(c["mean"] - bp["mean"]); series.append(f"{OPT_LABEL[opt]}")
    if not xs:
        return None
    levels = list(dict.fromkeys(series))
    pal = {OPT_LABEL[o]: DUST_PAL[o] for o in DUST_PAL}
    return line_chart(
        name, {"population (DUST draws)": xs, "DUST minus backprop": ys, "optimiser": series}, x="population (DUST draws)", y="DUST minus backprop",
        colour="optimiser", points=True, log_x=True, x_label="DUST draws per update (log scale)",
        y_label="DUST minus backprop, test loss (nats)", colour_label="optimiser", hlines=[0.0],
        palette=[pal[l] for l in levels], width=cm(14), height=cm(8),
        alt="DUST's test loss minus backprop's under the same optimiser against population at 1M tokens; zero, dashed, is level with backprop.",
    )


def optimiser_curves(rows: list[dict], name: str = "optimiser_curves") -> Figure | None:
    """Validation loss over the 61 updates at 1M tokens: backprop and DUST at 256 draws under SGD, Adam and Muon."""
    xs, ys, series, order = [], [], [], []
    for method, pop, tag, pal in (("dust", 256, "DUST 256", DUST_PAL), ("bp", 0, "backprop", BP_PAL)):
        for opt in ("sgd", "adam", "muon"):
            rs = opt_rows(rows, method, pop, opt)
            if not rs:
                continue
            steps = rs[0]["history"]["step"]
            label = f"{tag}, {OPT_LABEL[opt]}"
            for i, s in enumerate(steps):
                vals = [r["history"]["val_loss"][i] for r in rs if len(r["history"]["val_loss"]) == len(steps)]
                if vals:
                    xs.append(s / max(steps)); ys.append(sum(vals) / len(vals)); series.append(label)
            order.append((label, pal[opt]))
    if not xs:
        return None
    levels = list(dict.fromkeys(series))
    pal = dict(order)
    return line_chart(
        name, {"fraction of training": xs, "validation loss": ys, "run": series}, x="fraction of training", y="validation loss",
        colour="run", x_label="fraction of the 1M-token budget consumed", y_label="validation loss (mean over seeds)", colour_label="",
        palette=[pal[l] for l in levels], y_limits=(5.2, 7.4), width=cm(14), height=cm(9),
        alt="Validation loss over training at 1M tokens for backprop and for DUST at 256 draws, each under SGD, Adam and Muon.",
    )


def _cos_value(rows: list[dict], ckpt: str, method: str, pop: int, layer: str) -> float | None:
    """Mean over noise draws of the cosine for one layer type ('mlp' averages MLP in and out)."""
    rs = [r for r in rows if r.get("kind") == "cosine" and r["ckpt"] == ckpt and r["method"] == method and r["pop"] == pop]
    if not rs:
        return None
    keys = ("mlp_in", "mlp_out") if layer == "mlp" else (layer,)
    vals = [sum(r["cos"][k] for k in keys) / len(keys) for r in rs]
    return sum(vals) / len(vals)


CKPT_NAME = {"init": "initialisation", "bp100M_step610": "10M-token checkpoint", "bp100M_step6100": "100M-token checkpoint"}
COS_PALETTE = ["#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#6b7280"]


def cosine_dust(rows: list[dict], ckpt: str, name: str | None = None) -> Figure | None:
    """DUST's cosine to backprop against population, per layer type, at one checkpoint."""
    xs, ys, series = [], [], []
    types = [("head", "LM head"), ("attn_out", "attention out"), ("mlp", "MLP"), ("token_emb", "token embedding"),
             ("v", "V"), ("q", "Q"), ("k", "K")]
    for key, label in types:
        for pop in POPS:
            v = _cos_value(rows, ckpt, "dust", pop, key)
            if v is not None:
                xs.append(pop); ys.append(v); series.append(label)
    if not xs:
        return None
    levels = list(dict.fromkeys(series))
    return line_chart(
        name or f"cosine_dust_{ckpt}", {"population K (draws)": xs, "cosine to backprop": ys, "layer type": series},
        x="population K (draws)", y="cosine to backprop", colour="layer type", points=True, log_x=True,
        x_label="DUST population K (draws, log scale)", y_label="cosine to backprop gradient", colour_label="",
        y_limits=(-0.05, 1.0), palette=(COS_PALETTE + ["#a78bfa"])[:len(levels)], width=cm(14), height=cm(8.5),
        alt=f"DUST's gradient cosine to backprop against population at the {CKPT_NAME.get(ckpt, ckpt)}, per layer type.",
    )


def cosine_vs_eggroll(rows: list[dict], ckpt: str, name: str | None = None) -> Figure | None:
    """DUST against EGGROLL on the same axes for three layer types."""
    xs, ys, series = [], [], []
    for key, label in (("head", "head"), ("attn_out", "attention out"), ("mlp", "MLP")):
        for method, mlabel in (("dust", "DUST"), ("eggroll", "EGGROLL")):
            for pop in POPS:
                v = _cos_value(rows, ckpt, method, pop, key)
                if v is not None:
                    xs.append(pop); ys.append(v); series.append(f"{mlabel} {label}")
    if not xs:
        return None
    order = [s for s in ("DUST head", "EGGROLL head", "DUST attention out", "EGGROLL attention out", "DUST MLP", "EGGROLL MLP")
             if s in series]
    pal = {"DUST head": "#7c3aed", "EGGROLL head": "#c4b5fd", "DUST attention out": "#0ea5e9", "EGGROLL attention out": "#bae6fd",
           "DUST MLP": "#10b981", "EGGROLL MLP": "#a7f3d0"}
    return line_chart(
        name or f"cosine_vs_eggroll_{ckpt}", {"population": xs, "cosine to backprop": ys, "estimator and layer": series},
        x="population", y="cosine to backprop", colour="estimator and layer", points=True, log_x=True,
        x_label="population (DUST draws / EGGROLL forward passes, log scale)", y_label="cosine to backprop gradient",
        colour_label="", y_limits=(-0.05, 1.0), palette=[pal[s] for s in order], width=cm(14), height=cm(8.5),
        alt=f"Cosine to the backprop gradient against population at the {CKPT_NAME.get(ckpt, ckpt)} for DUST and EGGROLL.",
    )


def build_figures(rows: list[dict]) -> list[Figure]:
    """Every figure the paper needs. Guards on what exists: results.jsonl is partial for most of the project."""
    figs: list[Figure | None] = []
    figs.append(parity_cosines(rows))
    figs.append(eggroll_tuning(rows))
    for budget in ("100K", "1M", "10M"):
        figs.append(ladder(rows, budget))
        figs.append(cost(rows, budget))
        figs.append(curves(rows, budget))
    for ckpt in ("init", "bp100M_step610", "bp100M_step6100"):
        figs.append(cosine_dust(rows, ckpt))
        figs.append(cosine_vs_eggroll(rows, ckpt))
    figs.append(optimiser_ladder(rows))
    figs.append(optimiser_zoom(rows))
    figs.append(optimiser_gap(rows))
    figs.append(optimiser_curves(rows))
    return [f for f in figs if f is not None]
