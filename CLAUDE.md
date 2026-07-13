# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`jax-sn` is a JAX-based discrete-ordinates (S_N) neutron transport solver: finite-element spatial
discretisation (via `fenics-basix`) x discrete-ordinates angular quadrature x multigroup energy, built
around composable JAX operators (`lineax.AbstractLinearOperator`) that are differentiable end-to-end.
Optional GPU backends (Pallas, NVIDIA Warp) provide fused transport-sweep kernels. Docs (theory +
API reference) are at https://timobogaarts.github.io/jax-sn/, built from `docs/source`.

## Environment

Developed against a conda env (see `.vscode/settings.json` — `python-envs.defaultEnvManager: conda`).
Install editable with test extras: `pip install -e ".[test]"`. Optional extras:
- `warp` — GPU warp-kernel transport sweep (`transport_operator_warp`, `linear_tetrahedron_warp_kernel`); guarded by try/except at import, so it's silently absent without the extra.
- `geometry` — mesh/BVH + Cartesian domain-setup helpers (`mesh_tools`, `domain_generation`); depends on `jax_sbgeom`, distributed via git rather than PyPI.

## Commands

Tests (from repo root):
```
pytest                                  # fast tests only
pytest --runslow                        # include slow-marked tests
pytest --runpallas                      # include Pallas-marked tests (needs a Pallas-capable backend)
pytest --jax-platform=cpu               # force JAX platform before jax is imported
pytest tests/finite_elements/test_transport_operator.py::test_name   # single test
```
`tests/conftest.py` sets `JAX_ENABLE_X64=True` and points `JAX_COMPILATION_CACHE_DIR` at
`<repo>/.jax_cache` (persistent XLA compilation cache — most test runtime is compilation, so a warm
cache matters much more than usual). Reference numerics live in `tests/reference_data/*.npz`,
regenerated via `tests/reference_data/regenerate.py`.

Docs: `./build_docs.sh` regenerates `docs/source/api` via `sphinx-apidoc` then builds
`docs/build/html` via `sphinx`.

## Architecture

### Package layout (`src/jax_sn`)
- `domain/` — geometry, mesh, cross-sections (`geometric_domain`, `domain_generation`, `graph_generation`, `cross_sections`).
- `solution_domain/` — ties a `domain` to a FEM basis (`base_fem`/`basix_fem`) into a `solution_domain.SolutionDomain`, the object most operators are built from.
- `quadrature_set/` — angular discretisation (T_N quadrature in `tn.py`).
- `energy_set/` — multigroup energy structure/overlap.
- `operators/` — the S_N operators; see below.
- `parametrisation/` — single-group/multigroup/transport parametrisations for optimisation/gradient workflows (topologically sorted, see `test_parametrisation_toposort.py`).
- `variance_reduction/` — FW-CADIS.
- `tools/` — mesh/graph/permutation utilities, memory analysis.
- `analytical_solutions/` — closed-form references (e.g. `slab.py`) used by tests/examples.

### Operator composition: strategy objects, not subclassing
Every operator axis (mass, moment, discrete, transport forward/inversion, scatter, multi-group) is an
`eqx.Module` **strategy** passed as a keyword to a factory, not a subclass of the operator. This is the
load-bearing pattern in the codebase — a new kernel/algorithm is a new strategy class, never a new
operator subclass. The central factory is
`operators/multi_group_operator/multi_group_operator_creation.py:create_multi_group_operator`, which
takes `forward_strategy`, `inversion_strategy`, `angle_lifting`, `moment_strategy`, `discrete_strategy`,
`mass_strategy`, `gather_scatter_operator_cls`, `multi_group_operator_cls` and wires them into a
`MultiGroupOperator`.

Transport is the richest axis (`operators/transport_operator/transport_operator_strategies.py`):
- `ForwardStrategy.forward` — single-angle forward apply.
- `InversionStrategy.invert` — single-angle sweep-inversion; also declares `vmappable()` (can an outer
  `vmap` see through this kernel, e.g. `False` for opaque Warp calls) and
  `required_capabilities()`/`default_sweep_data_type()` — the `SweepData` (toposort) it needs, checked
  via `check_compatible` at construction and on every `eqx.tree_at`.
- `AngleLifting.forward`/`.invert` — lifts a single-angle operator over the angle axis; declares
  `vmaps_inner()`. A `vmaps_inner()` lift over a non-`vmappable()` inversion silently serialises
  (`perf_tests/_discovery.py:serializes` flags this combination).
- Concrete implementations live under `operators/transport_operator/implementations/` next to their
  kernel (`transport_operator_jax.py`, `transport_operator_pallas.py`, `transport_operator_warp.py`,
  `transport_operator_forward.py`, `multi_angle_inversions.py`, `linear_tetrahedron_warp_kernel.py`) — optional
  backends (pallas/warp) are only importable/registered when their dependency is present.

All operators derive from `operators/base_linear_operator.py:AbstractLinearOperator_sn` (a `lineax`
operator restricted to real operators, with `as_matrix()` intentionally throwing rather than
materialising a potentially huge dense matrix).

## Performance testing (`perf_tests/`)

Standalone, **not part of the installed package** — scripts run from inside `perf_tests/` (its `_*`
helpers and `bench_*` modules import each other via relative `sys.path`, not package imports). Purpose:
measure predicted peak device memory and steady-state wall time per operator-strategy variant, and
catch regressions against a recorded "golden" file per machine. This is the harness intended for
evaluating new/faster kernel implementations, so know how to extend it, not just run it:

- `run.py` — orchestrator. Discovers every `bench_*.py`'s `BENCHES` list and runs the selected group.
  ```
  python run.py                               # cheap group (mv-style benches), default problem size
  python run.py --group all --save rtx3080     # include expensive solve/grad benches, write/merge a golden
  python run.py --benches mass.mv,moment.mv    # just these bench names
  python run.py --compare rtx3080              # exit 1 on >5% peak-mem drift vs golden/rtx3080.json
  ```
  Each `bench_x.py` is also directly runnable standalone (`python bench_mass.py`) with the same flags.
- `dev_bench.py` — fast dev loop for iterating on **one** strategy at a custom (small) mesh size
  instead of sweeping every variant at a fixed preset:
  ```
  python dev_bench.py --list
  python dev_bench.py --bench transport.inv --strategy MySweep --n-other 2 --tn-order 2
  python dev_bench.py --bench transport.inv.multiangle --strategy MySweep --lifting VmapAngles
  ```
- `_bench.py` — `Bench` dataclass: one operator axis (`base` strategy class, `kwarg` to
  `create_multi_group_operator`, `attr`/`extract` to pull the sub-operator out, `make_call` — default
  `mv_call`, also `inv_call`). Add a new axis by adding a `Bench(...)` to a `bench_*.py`'s `BENCHES`.
- `_discovery.py` — **never hand-list the strategy matrix**. `discover_variants` walks
  `__subclasses__()` of the strategy base to find every concrete, importable, default-constructible
  variant — so a new strategy class is benchmarked automatically just by existing on the machine, and
  an unavailable backend (no GPU/warp) silently drops out. A strategy needing a non-default constructor
  arg (batch size, level cap) is supplied via a `Bench.overrides` map, not a special case in discovery.
- `_common.py` — measurement primitives: `peak_mb` (from the XLA compiler via `.lower().compile()`, no
  execution — deterministic, never OOMs, works even for configs too large to run), `time_ms` (adaptive
  warm-up + min-of-N), OOM-safe (`measure` turns build/compile/run failures into a self-describing
  "skip" row rather than aborting the sweep), plus golden JSON load/save/compare
  (`perf_tests/golden/<machine>.json`, 5% peak-mem tolerance = fail, 25% time drift = advisory warning).
- `_create_problem.py` / `problems.py` — named problem-size presets (`PROBLEMS` dict: dim, FEM degree,
  element count, T_N order); `mv-*` sizes for single-application benches, `solve-*` for iterative
  solves, `tiny-d1` for smoke tests.

When adding a new kernel/strategy for performance work: implement it as a concrete strategy subclass
under the relevant `implementations/` (or operator) module — it will be auto-discovered by
`perf_tests`, so `python run.py` / `dev_bench.py` picks it up with no perf_tests-side changes needed
beyond, if it needs non-default construction args, one `overrides` entry on the relevant `Bench`.
