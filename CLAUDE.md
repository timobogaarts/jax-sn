# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`jax-sn` is a JAX-based discrete-ordinates (S_N) neutron transport solver: finite-element spatial
discretisation (via `fenics-basix`) x discrete-ordinates angular quadrature x multigroup energy, built
around composable JAX operators (`lineax.AbstractLinearOperator`) that are differentiable end-to-end.
Optional GPU backends (Pallas, NVIDIA Warp) provide fused transport-sweep kernels. Docs (theory +
API reference) are at https://timobogaarts.github.io/jax-sn/, built from `docs/source`.

## Environment

Python >= 3.12, developed against a conda env. Published on PyPI as `jax-sn` (imported as `jax_sn`).
Install editable for the test suite with `pip install -e ".[test,geometry,warp]"` — the tests import the
Warp and geometry modules directly. Optional extras:
- `warp` — Warp transport-sweep and mass kernels (`transport_operator_warp`, `simplex_warp_kernel`, `mass_operator_warp`); the package `__init__`s guard the import with try/except, so without the extra the `Warp*` strategies are simply absent. `warp-lang>=1.16` also runs on CPU, which the docs CI relies on.
- `geometry` — mesh/BVH + Cartesian domain-setup helpers (`mesh_tools`, `domain_generation`); depends on `jax_sbgeom` (on PyPI).
- `viz` — `tools/plotly.py` (not imported eagerly); `docs` — `./build_docs.sh`; `dev` — pre-commit/nbstripout.

## Commands

Tests (from repo root):
```
pytest                                  # fast tests only
pytest --runslow                        # include slow-marked tests
pytest --runpallas                      # include Pallas-marked tests (needs a Pallas-capable backend)
pytest --jax-platform=cpu               # force JAX platform before jax is imported
pytest tests/operator_tests/test_operator_transport_inversion.py::test_name   # single test
```
`tests/conftest.py` sets `JAX_ENABLE_X64=True` in `pytest_configure` (before collection imports the test
modules), so the whole suite is keyed on one dtype configuration. Note most test runtime is XLA
compilation, so a warm compilation cache matters much more than usual — conftest no longer sets
`JAX_COMPILATION_CACHE_DIR` itself, so set it in the environment if you want one across runs.
Reference numerics live in `tests/reference_data/*.npz`, regenerated via
`tests/reference_data/regenerate.py`.

Release: pushing a tag `vX.Y.Z` (must equal `version` in `pyproject.toml`) runs `tests.yml` (CPU, fast
tests only), builds, and publishes to PyPI via Trusted Publishing (`release.yml`, environment `pypi`).

Docs: `./build_docs.sh` regenerates `docs/source/api` via `sphinx-apidoc` then builds
`docs/build/html` via `sphinx`. Example notebooks (`examples/*.ipynb`) are copied into the docs tree
and executed fresh on every build (`--no-execute` skips execution for a fast prose-only preview);
`docs/source/_notebook_manifest.py` is the single source of truth for which notebooks exist and where
they go. A notebook marked `ci: False` there (needs a GPU backend or external data hosted CI can't
produce, e.g. an OpenMC-based FW-CADIS example) is excluded from execution and instead renders its
committed outputs — refresh those manually via `tools/refresh_notebooks.py`. `examples/*.ipynb` have
their outputs stripped pre-commit (`.pre-commit-config.yaml`, `nbstripout`); `ci: False` notebooks
need a matching exclusion there so their committed outputs survive.

## Architecture

### Package layout (`src/jax_sn`)
- `domain/` — geometry, mesh, cross-sections (`geometric_domain`, `domain_generation`, `graph_generation`, `cross_sections`).
- `solution_domain/` — ties a `domain` to a FEM basis (`base_fem`/`basix_fem`) into a `solution_domain.SolutionDomain`, the object most operators are built from.
- `quadrature_set/` — angular discretisation (T_N quadrature in `tn.py`).
- `energy_set/` — multigroup energy structure/overlap.
- `operators/` — the S_N operators; see below.
- `operator_creation/` — `builders.py` (the builder ladder: one rung per operator kind, the only path from primitives to an operator; see below) plus `default_strategies.py` (`DefaultMultiGroupStrategy` — ready-made vmap/pallas/warp strategy trees).
- `solvers/` — direct solvers built on the operators: block forward/back-substitution for (block) lower-triangular multigroup systems (`MultiGroupLowerTriangularBlockSolver` in `base_solvers.py`) and configurable right-hand-sides (`right_hand_side.py`).
- `variance_reduction/` — FW-CADIS (`fw_cadis.py`); also holds the continuous adjoint (`continuous_adjoint.py`), which is the forward `MultiGroupOperator` applied to transformed data rather than a separate operator.
- `tools/` — mesh/graph/permutation utilities, memory analysis, and `parametrisation.py` (the `ParametrisedModule` machinery the builders are written in). Also `parametrisation_tools.py`, an unrelated leftover: scipy/objective glue for optimisation loops, imported by nothing in-package.
- `analytical_solutions/` — closed-form references (e.g. `slab.py`) used by tests/examples.

### Problem specification: primitives in, everything else derived
The builders exist because derived data used to be *cached* in the operator pytree and re-derived by
hand wherever a primitive changed (the `parametrisation/` package, since removed, held one
reconstruct function per parametrised quantity), so the derivation lived in two places and could
drift. Now a class declares only *what depends on what* and the storage policy is generated from that
graph.

`tools/parametrisation.py` is the mechanism. A `ParametrisedModule` subclass sets `DERIVATIONS = {quantity:
function}`, where a derivation's **parameter names are its dependencies** and anything without a
derivation is a **primitive**. A concrete subclass declares as fields the primitives plus whichever
derived quantities it stores; accessors for the rest are injected, so `x.face_normals` is a field read
or a recomputation depending on the policy, and nothing at the call site changes. Three more registries
refine it: `PROJECTIONS` (quantity → how to read it back out of a container that already holds it, for
the derivations that repackage rather than compute), `NON_DIFFERENTIABLE` (marked on the registry,
not inside the derivation, so no derivation *can* forget the `stop_gradient` and it holds on the stored
and projected paths too), and `RETAINED_UNDER` (below). Intermediate classes must be declared
`abstract=True` — a field cannot override an injected accessor, so only an abstract class is safe to
inherit from or generate from. A quantity that is a `PROJECTIONS` target of one of its own
dependencies is a *view* (the derivation and the projection are the same edge) and is never stored —
otherwise "most derived first" would store it *and* its own source beside it.

Storage is stated as intent, not as a field list: `.varying(*primitives)` generates the subclass
keeping those updatable and storing as much of the rest as the graph allows, `.freezing(*primitives)`
the converse, `.storing(*quantities)` states it directly (to seed from data you already have, then
`.materialise(cls)` into the policy you want). Each has a past-tense instance-side twin that converts
rather than generates — `.frozen(...)` / `.varied(...)` / `.stored(...)`, i.e.
`builder.frozen("element")` is `builder.materialise(Builder.freezing("element"))`.
`.updated(**primitives)` replaces primitives and everything downstream follows. Generated classes are memoised on `(base, stored)` — a class object is
part of a treedef, so two structurally identical but distinct classes would retrace.

`RETAINED_UNDER = {quantity: {primitive: predicate(held_value, new_primitive)}}` handles edges that
are coarser than the graph can see. A dependency edge records only *that* a quantity is a function of
a primitive, so any policy holding the sweep tier freezes `strategy` outright — yet the strategy
reaches that tier only through what it asks of it, and a tier at least as rich as the new strategy
needs is still correct. It has two halves, and they are the same statement:
- `updated` may replace such a primitive even though it is `frozen`, when the predicate holds. It
  raises naming the refusing quantity when it does not — never a silent rebuild of a tier the policy
  promised to keep. These primitives are `graph().conditionally_free`.
- `_frontier` may *store* a quantity that only such a primitive held back, so `.varying(X)` stores
  more than before. Only a quantity's own retentions count: one declared further down is on a value
  nothing will hold, so nothing could check it at update time. Note this means `.varying(X)` no
  longer promises X is unconditionally updatable.

The predicate must be sound (it asserts correctness, not convenience) and runs in Python, so it may
read only static structure, never a traced leaf. Three are declared: `DiscretisationBuilder` for
`cross_sections_indexed` (the harmonics read them only for `l_max`, so `.varying("cross_sections_indexed")`
— the canonical θ policy — now *stores* `angular_data`), `SweptBuilder` for `strategy`
(`strategy.accepts(tier)`), `TransportBuilder` for `inversion_strategy`. The forward strategy never
touches the sort, so it is plain `free` whatever the policy.

`operator_creation/builders.py` is the ladder over that mechanism, along two axes. The **operator**
axis is the class ladder: `SpatialBuilder` (mesh + element + materials) → `DiscretisationBuilder`
(adds `quadrature_set`) → `SweptBuilder` (adds the `non_differentiable_data` tier, with `sweep_data` a
view of it, and marks it non-differentiable)
→ the operator rungs (`MultiGroupBuilder`, `SingleGroupBuilder` / `SingleGroupChainBuilder` /
`SingleGroupSourceBuilder`, `MassBuilder`, `MomentBuilder`, `DiscreteBuilder`, `ScatterBuilder`,
`SingleAngleTransportBuilder` / `MultiAngleTransportBuilder`), each adding the primitives that operator
is built from and the quantities that follow. The **storage** axis is the generated one above. Which
tier an operator needs is visible in its registry rather than declared: only the rungs whose operator
actually sweeps carry `sweep_data`, so the others have no sweep to skip and no placeholder to invent.
The transport rungs hang off `SpatialBuilder` rather than
`DiscretisationBuilder` — they sweep a given `angle`, so they need no quadrature set — and single- vs
multi-angle are separate rungs so the shape/strategy compatibility checks become `__check_init__`
invariants that re-fire on every update.

The load-bearing fact underneath all of it: nothing untraceable (FEM matrices, `etoe`/`etof`/
`face_perm_idx`) depends on the vertices or the cross sections, so a builder can freeze the structural
tier once and leave the whole θ-dependent layer pure JAX.

`from_solution_domain` is the adapter for code that already has a `SolutionDomain`: nothing is
recomputed, the solution domain is stored whole and the spatial tier projected back out.

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

Building blocks: every operator is built by asking a builder rung for it. From an existing
`SolutionDomain` that is

```python
MultiGroupBuilder.from_solution_domain(
    solution_domain, quadrature_set=qs, strategy=strategy).multi_group_operator
```

and the same shape on the other rungs (`.operator_data`, `.transport_operator`, `.mass_operator`, …).
A builder takes no build-time knobs: every one of them is a static field on whichever strategy wants
it, so it rides in on `strategy` and no rung has to pass anything through. The load-bearing case is
the leveled sweep (`LeveledSweepData`, what the Warp strategies want): it pads its level arrays to a
fixed width, and left to itself reads that width off the levels — a tracer once the primitive it
derives from is one, so `builder.updated(vertices=...)` under `jit` dies with a
`ConcretizationTypeError`. Pinning it is `WarpLevelSweep(maximum_number_of_levels=n)` /
`WarpLevelAngles(maximum_number_of_levels=n)`, with `n` read off a concretely-built sweep as
`sweep_data.level_fwd.n_levels`; too small a value is a clean equinox runtime error, not silent
corruption. `perf_tests/_bench.create_multi_group_builder` does this, rewriting the built strategy
tree via `_bench.pin_level_width` (the knob is static, so it is a dataclass-field walk — `eqx.tree_at`
only reaches leaves).
The shared `OperatorData` a rung assembles is
`DiscretisationData(spatial, angular)` (differentiable: FEM/geometry/cross-sections + `AngularData`
quadrature & harmonics) + `NonDifferentiableData(sweep_data)`. The non-differentiable tier is built
whole by the strategy (`RequiresNonDifferentiableData` in `operators/operator_data.py`:
`build_default_non_differentiable_data(geometric_domain, angles)` builds it, `accepts(tier)`
says whether an existing one already serves; both abstract) — stated over the tier rather than over
the sweep data, so a strategy that comes to need more than a sweep ordering changes nothing
builder-side. It *builds* rather than naming a type because the tier is an assembly whose members are
chosen by different sub-strategies — and for the same reason the transport level below it builds too
(`build_default_sweep_data(geometric_domain, angle)`), so a knob-carrying strategy can apply its own
knobs without anything upstream knowing they exist. Composed strategies answer both by delegating to
the sub-strategy that consumes each part, bottoming out in the transport inversion's
`build_default_sweep_data()`/`accepts(sweep_data)`. What a strategy asks *of* an existing sweep is a
separate, cheap question (`required_capabilities()`), which is what `accepts` is written in terms of.
Note the strategy argument is **one composed `multi_group_strategy`**,
not a flat list of per-axis kwargs — the composition happens by nesting strategy objects (see
`DefaultMultiGroupStrategy` in `operator_creation/default_strategies.py` for canonical trees).

The multigroup strategy nests down through the axes:
- **Multigroup** (`operators/multi_group_operator/`): `MultiGroupStrategy` → `MultiGroupChainGatherStrategy`,
  which holds a `SingleGroupChainStrategy` + a `SingleGroupGatherScatterStrategy` and lifts the per-group
  body over the energy-group axis via `_lift_over_groups`. Two lifts, mixed in as `_ScanLift`/`_VmapLift`:
  `MultiGroupChainGatherScanStrategy` (serialising `lax.scan`, optional `checkpoint`) and
  `MultiGroupChainGatherVmapStrategy` (full `vmap`, memory-hungry, small problems only).
  `ComposedMultiGroupChainStrategy` (+ its `MultiGroupChainScanStrategy`/`MultiGroupChainVmapStrategy`
  lifts) is the chain *without* the gather-scatter, used to precondition a per-group right-hand side;
  `from_composed` builds it with the same lift as the operator it accompanies.
- **Single group** (`operators/single_group_operator/`): a `SingleGroupData(operator_data, energy_group)`
  view + one of three strategy flavours — `SingleGroupStrategy` (full `I − M T⁻¹ H D S`),
  `SingleGroupChainStrategy` (`M T⁻¹ H D`), `SingleGroupSourceStrategy` (`M T⁻¹ H`). The `Composed*`
  variants wire the mass / moment / discrete / transport / scatter sub-operators together and expose
  `all_operators()`.
- **Gather-scatter** (`operators/scatter_operator/`): now Data+Strategy+Operator like the rest. A per-group
  kernel `SingleGroupGatherScatterStrategy` (`BroadcastedSingleGroupGather`) with `incoming_gather`/
  `outgoing_gather`, lifted over groups by a `ComposedGatherScatterStrategy` (`VmapGatherScatterStrategy`)
  into the full `GatherScatterOperator`.

Transport is the richest axis. Strategy contracts live in
`operators/transport_operator/transport_operator_base.py`, combinators in
`transport_operator_strategies.py`:
- Forward: `SingleAngleForwardStrategy.forward` (single angle) and `MultiAngleForwardStrategy.forward`
  (angle axis leading); `ComposedMultiAngleForwardStrategy` lifts a single-angle kernel and names it via
  `single_angle_strategy()`.
- Inversion: `AbstractInversionStrategy.invert` is the shared sweep contract and declares the sweep-data
  precondition — `required_capabilities()`/`build_default_sweep_data()`, enforced by `check_compatible`
  from the operator's `__check_init__`, so it fires when the operator is **constructed** — but *not* on
  `eqx.tree_at`, which rebuilds via `tree_unflatten` and bypasses `__check_init__`. Swapping a strategy
  in with `tree_at` (as several test fixtures do) skips the check entirely.
  `SingleAngleInversionStrategy` adds `vmappable()` (can an outer `vmap` see through this kernel — `False`
  for opaque Warp calls); `MultiAngleInversionStrategy` is the angle-axis form, with
  `ComposedMultiAngleInversionStrategy` lifting a single-angle sweep.
- A single-angle strategy can advertise a fused multi-angle kernel via `native_multi_angle()` (e.g. the
  Warp level-sweep returns `WarpLevelAngles`) instead of being lifted by a combinator.
- Concrete lifts: `VmapForward`/`BatchedForward` (forward), `VmapSweep`/`BatchedSweep` (jax inversion),
  `VmapAnglesInversion`/`ScanAnglesInversion` + `VmapAnglesForward`/`ScanAnglesForward` (multi-angle),
  plus pallas/warp variants. Concrete kernels live under `operators/transport_operator/implementations/`
  (`transport_operator_jax.py`, `transport_operator_pallas.py`, `transport_operator_warp.py`,
  `transport_operator_forward.py`, `multi_angle_inversions.py`, `simplex_warp_kernel.py`) —
  optional pallas/warp backends are only importable/registered when their dependency is present.
- The Warp kernels are **parametrised on the element**, not written per element type.
  `simplex_warp_kernel.build_kernels(n_basis, n_faces)` (memoised) compiles the struct/func/kernel set
  for one element shape; the strategies read that shape off `spatial_data` and dispatch at trace time,
  so there is one `WarpLevelSweep`, not one per element. `mass_operator_warp.build_kernel(n_basis)` does
  the same for mass. The only limit is Warp's fixed-size vector/matrix types, which stop at length 4 —
  hence `SUPPORTED = ((2,2), (3,2), (4,2), (3,3), (4,4))`: linear/quadratic/cubic interval, linear
  triangle, linear tetrahedron. Spatial types are deliberately *not* parametrised: angles and normals
  are 3-vectors at every dimension, and `vol_jacobians_inv`/`derivative_matrix` are zero-padded to 3
  host-side (a no-op in 3-D), which reproduces the JAX kernel's `angle[:n_dim]` truncation exactly and
  avoids needing a 1x1 matrix type Warp does not have. Note the file must **not** use
  `from __future__ import annotations` — Warp resolves `@wp.struct` annotations against module globals,
  where the per-shape types (locals of the factory) are invisible.
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
- `_bench.py` — `Bench` dataclass: one operator axis (`base` strategy class; `chain_field`/`top_level`/`lift`
  say where a discovered instance is composed into the default multi-group strategy tree, via
  `compose_chain_strategy`; `extract` pulls the sub-operator out of the built operator; `make_call` — default
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
