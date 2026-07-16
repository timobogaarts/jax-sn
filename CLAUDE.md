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
- `operator_creation/` — the factory layer that builds operators from a `SolutionDomain` + `QuadratureSetBase`: `multi_group_operator_creation.py` (`create_operator_data`, `create_multi_group_operator`) and `default_strategies.py` (`DefaultMultiGroupStrategy` — ready-made vmap/pallas/warp strategy trees).
- `parametrisation/` — single-group/multigroup/transport parametrisations for optimisation/gradient workflows (topologically sorted, see `test_parametrisation_toposort.py`).
- `solvers/` — direct solvers built on the operators: block forward/back-substitution for (block) lower-triangular multigroup systems (`lower_triangular_solver.py`) and configurable right-hand-sides (`right_hand_side.py`).
- `variance_reduction/` — FW-CADIS; also holds the adjoint multigroup operator (`adjoint_multigroup_operator.py`).
- `tools/` — mesh/graph/permutation utilities, memory analysis.
- `analytical_solutions/` — closed-form references (e.g. `slab.py`) used by tests/examples.

### Operator composition: data + strategy, not subclassing
Every operator is a **data pytree + strategy object**, not a `forward`-implementing subclass. The
`data` (an `eqx.Module` under `BaseOperatorData`) holds everything needed to apply the operator; the
`strategy` (an `eqx.Module`) is stateless and says *how* to apply it — so a new kernel/algorithm is a
new strategy class, never a new operator subclass, and composite operators store their data once
instead of duplicating a copy per sub-operator. This is the load-bearing pattern; the rationale
(avoiding duplicated data under composition, enabling fused kernels that never reference sub-operators)
is written out at the top of `operators/base_linear_operator.py`.

`base_linear_operator.py` defines the whole taxonomy. Strategies: `Strategy` (forward only) →
`TransposableStrategy` (implements `transpose` too) → `SymmetricStrategy`. Operators: everything derives
from `AbstractLinearOperator_sn` (a `lineax` operator restricted to real operators, with `as_matrix()`
intentionally throwing rather than materialising a potentially huge dense matrix). `DataStrategyLinearOperator`
is the data+strategy base; direction is handled two ways — `DataTransposeDataStrategyLinearOperator`
transposes by transforming the data (`TransposableOperatorData.transposed`, e.g. transport negates the
angle + reverses the sweep), while `TransposeBoolDataStrategyLinearOperator` carries a `transpose_bool`
flag and lets the `TransposableStrategy` supply both directions. `DirectlyInvertibleOperator` adds a
direct `inv()` via a dummy `lineax` solver (used by transport sweeps).

Building blocks: `create_operator_data(solution_domain, quadrature_set, strategy)` assembles the shared
`OperatorData` = `DiscretisationData(spatial, angular)` (differentiable: FEM/geometry/cross-sections +
`AngularData` quadrature & harmonics) + `NonDifferentiableData(sweep_data)`. The sweep data type is
chosen by the strategy's `default_sweep_data()`. `create_multi_group_operator(solution_domain,
quadrature_set, multi_group_strategy, sweep_data_kwargs={})` then wraps it in a `MultiGroupOperator`.
Note the factory signature is now **one composed `multi_group_strategy`**, not the old flat list of
per-axis kwargs — the composition happens by nesting strategy objects (see `DefaultMultiGroupStrategy`
in `operator_creation/default_strategies.py` for canonical trees).

The multigroup strategy nests down through the axes:
- **Multigroup** (`operators/multi_group_operator/`): `MultiGroupStrategyBase` → `ComposedMultiGroupStrategy`,
  which holds a `SingleGroupChainStrategy` + a `SingleGroupGatherScatterStrategy` and lifts the per-group
  body over the energy-group axis. Two lifts: `MultiGroupScanStrategy` (serialising `lax.scan`, optional
  `checkpoint`) and `MultiGroupVmapStrategy` (full `vmap`, memory-hungry, small problems only).
- **Single group** (`operators/single_group_operator/`): a `SingleGroupData(operator_data, energy_group)`
  view + one of three strategy flavours — `SingleGroupStrategy` (full `I − M T⁻¹ H D S`),
  `SingleGroupChainStrategy` (`M T⁻¹ H D`), `SingleGroupSourceStrategy` (`M T⁻¹ H`). The `Composed*`
  variants wire the mass / moment / discrete / transport / scatter sub-operators together and expose
  `all_operators()`.
- **Gather-scatter** (`operators/scatter_operator/`): now Data+Strategy+Operator like the rest. A per-group
  kernel `SingleGroupGatherScatterStrategy` (`BroadcastedSingleGroupGather`) with `incoming_gather`/
  `outgoing_gather`, lifted over groups by a `ComposedGatherScatterStrategy` (`VmapGatherScatterStrategy`)
  into the full `GatherScatterOperator`. (The old `gather_scatter_operator_cls` factory kwarg is gone.)

Transport is the richest axis. Strategy contracts live in
`operators/transport_operator/transport_operator_base.py`, combinators in
`transport_operator_strategies.py`:
- Forward: `SingleAngleForwardStrategy.forward` (single angle) and `MultiAngleForwardStrategy.forward`
  (angle axis leading); `ComposedMultiAngleForwardStrategy` lifts a single-angle kernel and names it via
  `single_angle_strategy()`.
- Inversion: `AbstractInversionStrategy.invert` is the shared sweep contract and declares the sweep-data
  precondition — `required_capabilities()`/`default_sweep_data_type()`, enforced by `check_compatible`
  from the operator's `__check_init__` (so it fires at construction and on every `eqx.tree_at`).
  `SingleAngleInversionStrategy` adds `vmappable()` (can an outer `vmap` see through this kernel — `False`
  for opaque Warp calls); `MultiAngleInversionStrategy` is the angle-axis form, with
  `ComposedMultiAngleInversionStrategy` lifting a single-angle sweep.
- A single-angle strategy can advertise a fused multi-angle kernel via `native_multi_angle()` (e.g. the
  Warp level-sweep returns `WarpLevelAngles`) instead of being lifted by a combinator.
- Concrete lifts: `VmapForward`/`BatchedForward` (forward), `VmapSweep`/`BatchedSweep` (jax inversion),
  `VmapAngles`/`ScanAngles` + `VmapAnglesForward`/`ScanAnglesForward` (multi-angle), plus pallas/warp
  variants. Concrete kernels live under `operators/transport_operator/implementations/`
  (`transport_operator_jax.py`, `transport_operator_pallas.py`, `transport_operator_warp.py`,
  `transport_operator_forward.py`, `multi_angle_inversions.py`, `linear_tetrahedron_warp_kernel.py`) —
  optional pallas/warp backends are only importable/registered when their dependency is present.
- `SweepData` (the per-angle topological ordering) is defined in `operators/sweep_data.py`, with concrete
  types (`BasicSweepData`, `LeveledSweepData`, …) in
  `operators/transport_operator/sweep_data_implementations.py`; it exposes `reverse` (transpose partner),
  `__getitem__` (unbatch a single angle) and `reconstruct` (rebuild after a domain deformation).

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
