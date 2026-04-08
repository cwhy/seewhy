"""
Visualization utilities for small_lm.

Plots: learning curves (loss, perplexity), generation samples.
All figures go to R2 via save_matplotlib_figure or local fallback.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure


def plot_loss_curve(history: dict, exp_name: str) -> str:
    """Plot CE loss and perplexity curves. Returns URL/path."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    steps = history.get("steps", list(range(len(history["loss"]))))
    ax1.plot(steps, history["loss"])
    ax1.set_xlabel("step")
    ax1.set_ylabel("CE loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)

    if "eval_loss" in history:
        eval_steps, eval_losses = zip(*history["eval_loss"])
        eval_ppl = [np.exp(l) for l in eval_losses]
        ax2.plot(eval_steps, eval_ppl, marker="o", ms=4)
        ax2.set_xlabel("step")
        ax2.set_ylabel("perplexity")
        ax2.set_title("Eval perplexity")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_visible(False)

    fig.suptitle(exp_name, fontsize=12)
    fig.tight_layout()
    url = save_matplotlib_figure(f"small_lm_{exp_name}_loss", fig, format="svg")
    plt.close(fig)
    return url
