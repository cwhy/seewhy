"""
gen_tsne_anim.py — Generate a t-SNE optimisation-drift animation for arbitrary embeddings.

Loads embeddings + labels from a file, runs openTSNE in phases while capturing
intermediate states, and produces a self-contained animated HTML file.

Input formats
-------------
  --npz  PATH          .npz file with keys 'embeddings' (N,D) and 'labels' (N,)
  --emb  PATH          .npy or .pkl file containing the (N,D) embedding array
  --labels PATH        .npy or .pkl file containing the (N,) label array
                       (required when using --emb; ignored with --npz)
  --pkl  PATH          .pkl file containing a dict; use --emb-key / --labels-key
                       to specify which keys to read (default: 'embeddings'/'labels')

Usage examples
--------------
  uv run python projects/ema-viz/scripts/gen_tsne_anim.py \\
      --npz results/embeddings.npz --out my_exp_tsne

  uv run python projects/ema-viz/scripts/gen_tsne_anim.py \\
      --emb embeddings.npy --labels labels.npy --out my_exp_tsne --title "MNIST ViT"

  uv run python projects/ema-viz/scripts/gen_tsne_anim.py \\
      --pkl results.pkl --emb-key E_test --labels-key Y_test --subsample 3000

Options
-------
  --out NAME           output filename stem (default: tsne_anim)
  --title TEXT         title shown above the animation (default: "t-SNE animation")
  --fps FLOAT          default animation speed in frames/second (default: 2.0)
  --perplexity INT     t-SNE perplexity (default: 40)
  --early-steps INT    snapshots during early-exaggeration phase (default: 10)
  --early-iters INT    optimisation iters per early snapshot (default: 25)
  --refine-steps INT   snapshots during refinement phase (default: 15)
  --refine-iters INT   optimisation iters per refinement snapshot (default: 50)
  --subsample INT      randomly subsample to N points before t-SNE (default: all)
  --seed INT           random seed (default: 42)
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # project root
from shared_lib.media import save_media

# lib is one level up from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.tsne import capture_snapshots
from lib.viz  import tsne_animation_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_npz(path):
    data = np.load(path)
    return data["embeddings"], data["labels"]


def load_npy(emb_path, labels_path):
    E = np.load(emb_path)
    Y = np.load(labels_path)
    return E, Y


def load_pkl(path, emb_key="embeddings", labels_key="labels"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        return np.array(d[emb_key]), np.array(d[labels_key])
    raise ValueError(f"pkl file must contain a dict, got {type(d)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate a t-SNE drift animation HTML for arbitrary embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz",    metavar="PATH", help=".npz with embeddings/labels keys")
    g.add_argument("--emb",    metavar="PATH", help=".npy or .pkl embedding array")
    g.add_argument("--pkl",    metavar="PATH", help=".pkl dict file")
    p.add_argument("--labels",    metavar="PATH", help=".npy or .pkl label array (with --emb)")
    p.add_argument("--emb-key",   default="embeddings", help="dict key for embeddings (with --pkl)")
    p.add_argument("--labels-key",default="labels",     help="dict key for labels (with --pkl)")

    # Output
    p.add_argument("--out",   default="tsne_anim", help="output filename stem")
    p.add_argument("--title", default="t-SNE optimisation drift", help="animation title")

    # Animation
    p.add_argument("--fps",           type=float, default=2.0)

    # t-SNE
    p.add_argument("--perplexity",    type=int,   default=40)
    p.add_argument("--early-steps",   type=int,   default=10)
    p.add_argument("--early-iters",   type=int,   default=25)
    p.add_argument("--refine-steps",  type=int,   default=15)
    p.add_argument("--refine-iters",  type=int,   default=50)
    p.add_argument("--subsample",     type=int,   default=None,
                   help="randomly subsample to N points before t-SNE")
    p.add_argument("--seed",          type=int,   default=42)

    args = p.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.npz:
        E, Y = load_npz(args.npz)
    elif args.emb:
        if args.labels is None:
            p.error("--labels is required when using --emb")
        E, Y = load_npy(args.emb, args.labels)
    else:  # --pkl
        E, Y = load_pkl(args.pkl, args.emb_key, args.labels_key)

    E = np.array(E, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32).ravel()
    logging.info(f"Loaded embeddings {E.shape}, labels {Y.shape}")

    # ── Subsample ─────────────────────────────────────────────────────────────
    if args.subsample is not None and args.subsample < len(E):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(E), size=args.subsample, replace=False)
        idx.sort()
        E, Y = E[idx], Y[idx]
        logging.info(f"Subsampled to {len(E)} points")

    # ── t-SNE snapshots ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    snapshots = capture_snapshots(
        E,
        perplexity   = args.perplexity,
        n_early      = args.early_steps,
        early_iters  = args.early_iters,
        n_refine     = args.refine_steps,
        refine_iters = args.refine_iters,
        seed         = args.seed,
        log_fn       = logging.info,
    )
    logging.info(f"t-SNE done — {len(snapshots)} frames in {time.perf_counter()-t0:.1f}s")

    # ── Build HTML ────────────────────────────────────────────────────────────
    html = tsne_animation_html(snapshots, Y, title=args.title, fps=args.fps)

    # ── Save / upload ─────────────────────────────────────────────────────────
    out_name = args.out if args.out.endswith(".html") else f"{args.out}.html"
    url = save_media(out_name, html.encode(), content_type="text/html")
    logging.info(f"Animation → {url}")


if __name__ == "__main__":
    main()
