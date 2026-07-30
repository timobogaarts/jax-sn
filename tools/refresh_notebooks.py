#!/usr/bin/env python
"""Re-execute the example notebooks that can't run on hosted CI.

`docs/source/_notebook_manifest.py` lists which notebooks are `ci: False`
(e.g. need a GPU backend, or externally generated data like OpenMC cross
sections that hosted CI can't produce). Those keep whatever outputs are
committed in the notebook; this script is how you refresh them -- run it
locally on a machine with the right hardware/data, then review the diff and
commit (or open a PR).

Usage: python tools/refresh_notebooks.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "docs" / "source"))

from _notebook_manifest import NOTEBOOKS  # noqa: E402


def main() -> None:
    targets = [meta["source"] for meta in NOTEBOOKS.values() if not meta["ci"]]
    if not targets:
        print("No ci: False notebooks in docs/source/_notebook_manifest.py -- nothing to do.")
        return

    # Matches build_docs.sh's renderer pin (see the comment there): plotly figures
    # embed differently depending on the renderer plotly's autodetection happens to
    # land on, and these notebooks' outputs are committed as-is (never re-executed
    # by the docs build), so this has to be set here rather than relying on
    # whoever invokes this script to have exported it themselves.
    env = {**os.environ, "PLOTLY_RENDERER": "sphinx_gallery"}

    for rel_path in targets:
        path = REPO_ROOT / rel_path
        print(f"Executing {path} ...")
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
