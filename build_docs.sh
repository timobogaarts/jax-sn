#!/usr/bin/env bash
set -euo pipefail

# ./build_docs.sh              -- execute CI-runnable notebooks fresh (matches CI)
# ./build_docs.sh --no-execute -- skip execution, render whatever outputs are
#                                 already sitting in the docs copies (fast
#                                 local iteration on prose only)
if [ "${1:-}" = "--no-execute" ]; then
  export JAXSN_DOCS_NO_EXECUTE=1
fi

# The example notebooks live in examples/ (the copies users open and run); each
# one is copied into the docs tree, where myst_nb renders it as a page. The
# copies are generated, hence gitignored -- edit the original under examples/.
# The list of notebooks and their destinations lives in
# docs/source/_notebook_manifest.py (single source of truth, also used by
# conf.py and tools/refresh_notebooks.py) -- add a notebook there, not here.
copy_notebook() {  # <source notebook> <destination inside docs/source>
  mkdir -p "$(dirname "docs/source/$2")"
  cp "$1" "docs/source/$2"
}

while IFS=$'\t' read -r src dest; do
  copy_notebook "$src" "$dest"
done < <(python docs/source/_notebook_manifest.py)

python -m sphinx.ext.apidoc \
  -o docs/source/api \
  src/jax_sn \
  --force \
  --module-first \
  --separate\
  -t docs/source/_templates \
  -d 2\
  "src/jax_sn/tests/*" \
  "src/jax_sn/examples/*"

# depth of 2 for module documentation


python -m sphinx -b html docs/source docs/build/html
