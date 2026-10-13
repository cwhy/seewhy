"""dust-history: DUST with the population spread over time instead of over parallel draws.

This project reuses the model, the data and the DUST estimator pieces of `projects/dust` rather than copying them: putting
that project's directory first on the path makes `lib` mean *its* library. This package is therefore not called `lib`.
"""

import sys
from pathlib import Path

_DUST = Path(__file__).resolve().parents[2] / "dust"
if str(_DUST) not in sys.path:
    sys.path.insert(0, str(_DUST))
_ROOT = _DUST.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

import lib  # noqa: E402,F401  -- turns on the persistent XLA compilation cache
