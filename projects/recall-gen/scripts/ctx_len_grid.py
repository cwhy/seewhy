"""Does more context help a bounded memory? Train length against test length.

The delta rule writes every context item into one fixed dk x dk matrix per head
and reads a linear combination back. At d_model=512 with 8 heads of dk=64 that
state is 32,768 floats. Sixteen items of 832 numbers is 13,312 floats of content
— comfortably under. Sixty-four is 53,248 — comfortably over. So M=64 is the
first setting in this project where the memory is asked to hold more than it can.

Two checkpoints, trained at M=16 and M=64 on the same pool, each scored at both
lengths and at two more in between. Test length costs nothing, so the extra
columns come free; the 2x2 is the four cells where train and test length are
each 16 or 64.

Reported per cell, on hidden coordinates and normalised by the mean-item error:

    R           recall quality — how much the returned item's mistake costs,
                against what a randomly grabbed context item would cost
    committed   distance from the output to the NEAREST context item, divided by
                how far apart that episode's items are; low means the output is
                sitting on something actually in memory
    nmse        error against the true answer
    present/absent
                the same queries with the answer removed, so the gap is recall

    .venv/bin/python projects/recall-gen/scripts/ctx_len_grid.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT_DIR))

from shared_lib.typst_plot import bar_chart, cm, line_chart, long_form
from shared_lib.typst_report import save_figure
from recall_quality import analyse
from rescore import rows as read_rows

PROJ = "recall-gen"
V = "v1"
TRAIN = [("trained at M=16", "exp58"), ("trained at M=64", "exp59")]
TEST_M = [8, 16, 32, 64]
CELLS = (16, 64)          # the 2x2 the question asks for
# The synthetic held-out band and real datasets pulled in OPPOSITE directions in
# exp57: early-stopping on the former cost 0.335 of recall quality on
# Fashion-MNIST. So a ranking established on novel synthetic worlds is not
# allowed to stand as a ranking, and every cell is scored on real data too.
TRANSFER = [("MNIST", "synth_long_to_mnist"),
            ("Fashion-MNIST", "synth_long_to_fashion_mnist"),
            ("chess", "synth_long_to_chess")]
STATE_FLOATS = 32768
D_IN = 832


def main():
    by_exp = {r["experiment"]: r for r in read_rows()}
    R = {}
    for lab, e in TRAIN:
        for m in TEST_M:
            R[(e, m)] = analyse(e, by_exp[e], M=m)
            a = R[(e, m)]["A_seen_present"]
            b = R[(e, m)]["B_novel_present"]
            print(f"{lab:<18} test M={m:<3d}  seen R={a['R']:.3f} committed={a['committed']:.3f} "
                  f"nmse={a['nmse']:.4f}   novel R={b['R']:.3f} committed={b['committed']:.3f} "
                  f"nmse={b['nmse']:.4f}   absent={R[(e, m)]['C_seen_absent']['nmse']:.4f}")

    print("\nThe 2x2, worlds never seen:")
    print(f"{'':<18}" + "".join(f"  test M={m:<10d}" for m in CELLS))
    for lab, e in TRAIN:
        print(f"{lab:<18}" + "".join(
            f"  R {R[(e, m)]['B_novel_present']['R']:.3f} / cmt "
            f"{R[(e, m)]['B_novel_present']['committed']:.3f}" for m in CELLS))

    T = {}
    print("\nThe same 2x2 on real datasets (B band = the real data):")
    hdr = f"{'':<18}{'':<12}" + "".join(f"  test M={m:<8d}" for m in CELLS)
    print(hdr)
    for dlab, dom in TRANSFER:
        for lab, e in TRAIN:
            cells = []
            for m in CELLS:
                T[(e, dom, m)] = analyse(e, by_exp[e], as_domain=dom, M=m)
                b = T[(e, dom, m)]["B_novel_present"]
                cells.append(f"  R {b['R']:.3f} / cmt {b['committed']:.3f}")
            print(f"{dlab:<18}{lab:<12}" + "".join(cells))

    series = {}
    for lab, e in TRAIN:
        for m in CELLS:
            series[f"{lab.replace('trained at ', 'train ')}, test M={m}"] = [
                T[(e, dom, m)]["B_novel_present"]["R"] for _, dom in TRANSFER]
    u = save_figure(bar_chart(
        f"{PROJ}_r25_transfer",
        long_form([d for d, _ in TRANSFER], series,
                  x_name="dataset", y_name="R", series_name="cell"),
        x="dataset", y="R", fill="cell", x_order=[d for d, _ in TRANSFER],
        position="dodge",
        title="The same four cells, scored on real data",
        subtitle=("Both networks trained on the synthetic prior alone. Recall "
                  "quality on datasets neither has seen."),
        x_label="", y_label="recall quality", y_limits=(0.0, 1.05),
        caption=("The synthetic held-out band is not a reliable proxy for this, "
                 "so the ordering here is the one that decides the question."),
        width=cm(17), height=cm(9)), name=f"{PROJ}_r25_transfer_{V}")
    print("fig:", u)

    for band, key in (("worlds seen in training", "A_seen_present"),
                      ("worlds never seen", "B_novel_present")):
        tag = "seen" if key.startswith("A") else "novel"
        d = long_form(TEST_M,
                      {lab: [R[(e, m)][key]["R"] for m in TEST_M] for lab, e in TRAIN},
                      x_name="test_M", y_name="R", series_name="training")
        u = save_figure(line_chart(
            f"{PROJ}_r25_R_{tag}", d, x="test_M", y="R", colour="training", points=True,
            title=f"Recall quality against context length — {band}",
            subtitle=("The delta rule holds 32,768 floats of state. Context content is "
                      "832 floats per item, so it exceeds the state somewhere between "
                      "M=32 and M=64."),
            x_label="context items at test time", y_label="recall quality",
            y_limits=(0.0, 1.05),
            caption=("Both networks are scored at every length, so the diagonal of "
                     "the 2x2 is the matched case and the off-diagonal is transfer "
                     "across context length."),
            width=cm(16), height=cm(9)), name=f"{PROJ}_r25_R_{tag}_{V}")
        print("fig:", u)

    d = long_form(TEST_M,
                  {lab: [R[(e, m)]["B_novel_present"]["committed"] for m in TEST_M]
                   for lab, e in TRAIN},
                  x_name="test_M", y_name="committed", series_name="training")
    u = save_figure(line_chart(
        f"{PROJ}_r25_committed", d, x="test_M", y="committed", colour="training",
        points=True,
        title="Does the output still land on a stored item as the context grows?",
        subtitle="Worlds never seen. Lower is better.",
        x_label="context items at test time",
        y_label="distance to nearest stored item, at episode scale",
        caption=("A memory that saturates should drift upward here as more is "
                 "written into the same fixed state."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r25_committed_{V}")
    print("fig:", u)

    (PROJECT_DIR / "ctx_len_grid.json").write_text(json.dumps(
        {"synthetic": {f"{e}@M{m}": v for (e, m), v in R.items()},
         "transfer": {f"{e}@{d}@M{m}": v for (e, d, m), v in T.items()}}, indent=2))


if __name__ == "__main__":
    main()
