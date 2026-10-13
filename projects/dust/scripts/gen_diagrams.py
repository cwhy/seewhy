"""Write the hand-authored explainer diagrams (lib/diagrams.py) to paper/diagrams/*.svg.

Pure standard library, so it runs on the Mac. The SVGs are committed: they are authored
here, not generated from results, so unlike paper/assets they flow Mac -> box.

    python3 projects/dust/scripts/gen_diagrams.py
"""

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagrams", PROJECT / "lib" / "diagrams.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out = PROJECT / "paper" / "diagrams"
out.mkdir(exist_ok=True)
for name, fn in mod.ALL.items():
    (out / f"{name}.svg").write_text(fn())
    print("wrote", (out / f"{name}.svg").relative_to(PROJECT))
