"""Registry of example notebooks rendered into the docs.

Single source of truth, consumed by three places:
- ``build_docs.sh`` (run as a script here) to know which notebooks to copy
  from ``examples/`` into the docs tree, and where.
- ``conf.py`` to exclude ``ci=False`` notebooks from myst-nb execution (they
  keep whatever outputs are committed in the notebook instead).
- ``tools/refresh_notebooks.py`` to know which notebooks it's responsible
  for re-executing.

Add a notebook here when adding it to ``examples/``. If ``ci=False`` (e.g. it
needs a GPU backend, or external data such as OpenMC-generated cross
sections that hosted CI can't produce), also add a matching
``-filter`` line to ``.gitattributes`` so the nbstripout pre-commit hook
does not strip its committed outputs.
"""

NOTEBOOKS = {
    # docs-tree path (relative to docs/source): {"source": examples/ path, "ci": bool}
    "user_guide/first_operator.ipynb": {"source": "examples/first_operator.ipynb", "ci": True},
}

NON_CI_NOTEBOOKS = [docs_path for docs_path, meta in NOTEBOOKS.items() if not meta["ci"]]

if __name__ == "__main__":
    # Emitted for build_docs.sh: one "<source>\t<docs_path>" pair per line.
    for _docs_path, _meta in NOTEBOOKS.items():
        print(f"{_meta['source']}\t{_docs_path}")
