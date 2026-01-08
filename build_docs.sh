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