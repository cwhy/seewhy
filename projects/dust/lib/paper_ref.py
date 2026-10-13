"""The DUST paper's published numbers, transcribed from https://qlabs.sh/research/dust
(the post and its appendix page, read 2026-10-07).

Everything here is *theirs*, never ours. It is kept in one place so that every
comparison in reports and the paper cites a single source, and so that a transcription
error can be fixed once. Losses are test loss at the validation-selected checkpoint,
three-seed means unless noted.
"""

import numpy as np

POPS = (64, 256, 1024, 4096, 16384)
BUDGETS = ("100K", "1M", "10M", "20M")

# Appendix C, Table 1. Population = draws for DUST, forward passes of the batch for EGGROLL.
TABLE1 = {
    "dust": {
        64:    dict(zip(BUDGETS, (7.240, 6.290, 5.397, 5.265))),
        256:   dict(zip(BUDGETS, (7.165, 6.067, 5.231, 5.139))),
        1024:  dict(zip(BUDGETS, (7.151, 5.934, 5.111, 4.960))),
        4096:  dict(zip(BUDGETS, (7.164, 5.911, 5.072, 4.912))),
        16384: dict(zip(BUDGETS, (7.170, 5.916, 5.049, 4.802))),
    },
    "eggroll": {
        64:    dict(zip(BUDGETS, (7.961, 7.428, 7.056, 6.987))),
        256:   dict(zip(BUDGETS, (7.605, 7.200, 6.906, 6.811))),
        1024:  dict(zip(BUDGETS, (7.386, 7.046, 6.692, 6.517))),
        4096:  dict(zip(BUDGETS, (7.302, 6.892, 6.352, 6.200))),
        16384: dict(zip(BUDGETS, (7.263, 6.708, 6.033, 5.865))),
    },
    "bp": dict(zip(BUDGETS, (7.216, 5.959, 4.989, 4.633))),
}
DUST_LIMIT = dict(zip(BUDGETS, (7.161, 5.892, 5.013, 4.431)))     # power-law fit, population -> infinity

# Section 4.1, Figure 4: 10M tokens, four model sizes. Earlier training settings (Table 6), so
# the 38M column differs from Table 1's 10M column.
SIZES = ("L2d128", "L4d256", "L8d512", "L16d1024")
SIZE_PARAMS = dict(zip(SIZES, ("2.0M", "7.3M", "38M", "243M")))
FIG4 = {
    64:    dict(zip(SIZES, (5.705, 5.556, 5.558, 5.719))),
    256:   dict(zip(SIZES, (5.486, 5.362, 5.358, 5.419))),
    1024:  dict(zip(SIZES, (5.265, 5.161, 5.158, 5.214))),
    4096:  dict(zip(SIZES, (5.189, 5.095, 5.053, 5.124))),
    16384: dict(zip(SIZES, (5.171, 5.065, 5.036, 5.086))),
    "bp":  dict(zip(SIZES, (5.180, 5.066, 5.015, 5.048))),
}

# Appendix G, Table 8: cos(K) = c_max / sqrt(1 + c / K) at the 100M-token backprop checkpoint.
COSINE_FIT = {                         # type: (c_max, c, rmse)
    "mlp_out":   (0.8779, 31623, 0.0081),
    "mlp_in":    (0.8577, 48417, 0.0065),
    "attn_out":  (0.8729, 21627, 0.0093),
    "q":         (1.0000, 76736, 0.0356),
    "k":         (1.0000, 616595, 0.0191),
    "v":         (0.8776, 173780, 0.0269),
    "gate":      (0.6657, 47863, 0.0367),
    "value_emb": (1.0000, 495450, 0.0247),
    "token_emb": (0.7318, 19498, 0.0083),
    "head":      (1.0000, 759, 0.0032),
}


def cosine_law(layer_type: str, K) -> np.ndarray:
    """The paper's fitted cosine to backprop at population K (100M-token checkpoint)."""
    c_max, c, _ = COSINE_FIT[layer_type]
    return c_max / np.sqrt(1.0 + c / np.asarray(K, dtype=float))


def fit_cosine_law(K, cos) -> tuple[float, float, float]:
    """Least-squares (c_max, c, rmse) for cos = c_max / sqrt(1 + c/K), as the paper fits it."""
    from scipy.optimize import least_squares

    K, cos = np.asarray(K, float), np.asarray(cos, float)

    def resid(p):
        return p[0] / np.sqrt(1.0 + np.exp(p[1]) / K) - cos

    best = None
    for c0 in (1e2, 1e4, 1e6):
        r = least_squares(resid, [min(max(cos.max(), 0.05), 1.0), np.log(c0)], bounds=([0, -5], [1.0, 25]))   # a cosine cannot exceed 1
        if best is None or r.cost < best.cost:
            best = r
    c_max, c = float(best.x[0]), float(np.exp(best.x[1]))
    return c_max, c, float(np.sqrt(np.mean(resid(best.x) ** 2)))


# The reference points every loss chart should carry (ours, measured: scripts/tmp/unigram.py).
UNIGRAM_TEST_LOSS = 6.994           # token frequencies from the 1M-token prefix
# The same, from the prefix each budget actually trains on (constants row of results.jsonl, written by scripts/append_constants.py).
UNIGRAM_BY_BUDGET = {"100K": 7.027, "1M": 6.994, "10M": 6.978}
CHANCE_LOSS = float(np.log(4096))


# The DUST paper's Section 3 ("Dust Under Adam"): 1M tokens, Adam, five seeds for DUST and backprop, three for EGGROLL.
# Read from the page's text, not a summary. It reports backprop, and a power-law fit to DUST's ladder with its limit;
# the ladder's own values are in a figure and were not read.
ADAM_PAPER = dict(bp_1M=5.361, dust_limit=5.248, dust_limit_lo=5.17, dust_limit_hi=5.30, draw_factor=1.4)
