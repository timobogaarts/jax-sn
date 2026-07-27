#!/usr/bin/env bash
set -euo pipefail

# ./build_docs.sh              -- render notebooks from their committed outputs
# ./build_docs.sh --execute    -- re-run every notebook during the build instead
#                                 (fails the build on the first cell error)
if [ "${1:-}" = "--execute" ]; then
  export JAXSN_DOCS_EXECUTE=1
fi

# The example notebooks live in examples/ (the copies users open and run); each
# one is copied into the docs tree, where myst_nb renders it as a page. The
# copies are generated, hence gitignored -- edit the original under examples/.
copy_notebook() {  # <source notebook> <destination inside docs/source>
  mkdir -p "$(dirname "docs/source/$2")"
  cp "$1" "docs/source/$2"
}

copy_notebook examples/first_operator.ipynb user_guide/first_operator.ipynb

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
