# Operator data-ownership refactor (architecture_review §1) — progress & handoff

*Working doc. Last updated 2026-06-12. Companion to `architecture_review.md` (§1) and
`transport_operator_refactor.md` (the pattern this generalises).*

## Goal

Kill the duplicated shared data flagged in `architecture_review.md` §1: `CrossSectionsIndexed`,
`PrecomputedGeometricData`, and `FEMMatrices` currently live in **two pytree locations each**,
hand-synced via location lambdas (`*_loc` in `multi_group_operator_imp.py`). JAX pytrees are trees,
not DAGs — shared arrays physically live in exactly **one** node — so the fix is **single ownership +
operators derived as views**.

During design this generalised (with the user) into the pattern the transport refactor already
established:

> **`operator = setup (shared data) + spec (per-operator config) + strategy (implementation)`.**
> The composite owns one canonical `setup` and a small `spec` per operator, and exposes every
> sub-operator as a derived `@property`. No operator is privileged; a new implementation is a new
> strategy object, never a subclass.

## Key decisions (agreed)

1. **Composite-owns + per-operator specs**, not transport-owns. The composite (`SingleGroupOperator`
   / `MultiGroupOperator`) holds one canonical `setup`; transport/mass/scatter/moment/discrete are
   all symmetric derived views. (Rejected: transport owning the store — it made transport
   "privileged" and re-spread sweep/strategy knowledge nowhere clean.)
2. **`TransportSpec`** bundles transport's config (`topological_sorted_angle` + forward/inversion
   strategy + angle_lifting + energy_group) *minus* `setup`, so the composite stays ignorant of
   strategy internals (no layering regression vs the transport refactor).
3. **Harmonics (`wk_ylm`, `ylm`) precomputed once and stored in `setup`** (angular non-diff setup),
   not recomputed per `mv`. `setup` owns both spatial (domain+fem+geom) and angular
   (quadrature+harmonics) data; it is `l_max`-specific, like `fem_matrices` is element-specific.
4. **Diff/non-diff/static is a convention, not a container.** Handled by `eqx.field(static=True)` +
   the existing `ParametrisedModule.reconstruct_operator` boundary + `stop_gradient` — no new
   3-bucket type hierarchy.
   - *data* (differentiable): `cross_section_data`, `vertices`, `precomputed_geometric_data`.
   - *non-diff setup*: `fem_matrices` (reference-element only), `quadrature_set`, harmonics, toposort.
   - *static*: strategies, specs, classes, ints, shapes, `transpose_bool`.

## Status

| Stage | What | State |
|---|---|---|
| A | Strategy-ify moment & discrete; resurrect streaming; rehome `quadrature_set` | **DONE & committed** (FE + warp suites green) |
| B | `SingleGroupOperator` → setup + spec + derived views | **DONE** (FE + warp suites green) |
| B.1 | `SolutionSetup` introduced; `sweep_data` moved off the spec into setup; `TransportSpec` made data-free | **DONE** (FE + warp suites green) |
| C.1 | `MultiGroupOperator` → one `SolutionSetup` + specs; transport/mass/gather-scatter derived; `*_loc` collapsed to single locations | **DONE** (FE + warp suites green; §1 dedup verified) |
| C.2 | Fold angular data (quadrature + harmonics) into `SolutionSetup`; derive moment/discrete | **DONE** (FE + warp suites green) |
| C.2b | Remove `angle` from `sweep_data` (read it from the angular setup) | **DONE** (FE + warp suites green; angle now stored once) |

### Stage A — DONE (this session)

- **New `operators/moment_operator/moment_operator_strategies.py`** (mirrors
  `transport_operator_strategies.py`): `MomentStrategy` ABC (`forward`/`transpose`) →
  `BatchedMoment` (einsum) + `StreamingMoment` (the previously-orphaned scan kernels, now
  selectable); `DiscreteStrategy` → `BatchedDiscrete`. Kernels live here next to the strategies, so
  the `_imp` modules import only the array-free strategy classes (no circular import).
- **`MomentOperator` / `DiscreteOperator`**: each `Base`+concrete subclass pair collapsed into one
  **final** class with a `forward_strategy` field (`eqx.field(default_factory=...)`); `mv`
  dispatches `forward`/`transpose` through the strategy. The transpose flag (`transpose_bool`) is
  unchanged. The implementation-axis-as-subclass is gone (closes part of §3/§4 for these operators).
- **`quadrature_set` rehomed**: removed from moment/discrete (its only reader was
  `MGOp.quadrature_set`); now a **field on `MultiGroupOperator`** (interim — Stage C folds it into
  `setup`). Touched: MGOp field + `__check_init__` + `transpose` + `from_multigroup_operator` +
  `multi_group_operator_creation.py` + the positional constructions / `MGOP.moment_operator.quadrature_set`
  reads in `test_breeding_blanket.py` and `test_multigroup_derivative.py`.
- **Exports**: `moment_operator/__init__.py` now exports the final operators + the strategy classes
  (dropped `MomentOperatorBase`/`DiscreteOperatorBase`).

### Stage A.2 — mass operator strategy-ified (this session)

Same treatment for the **mass operator** (it had the same implementation-as-subclass smell:
`MassOperatorVmap` / `MassOperatorLax`):
- **New `operators/mass_operator/mass_operator_strategies.py`**: `MassStrategy` ABC → `VmapMass`
  (default, vmap-over-elements) + `LaxMass` (batched `lax.dot_general`). The mass operator is
  **symmetric**, so the strategy has a single `forward` (forward == transpose); no `transpose_bool`,
  unlike moment/discrete. The three kernels moved here.
- **`MassOperator`** collapsed `Base`+`Vmap`/`Lax` subclasses into one **final** class with a
  `strategy` field; `mv` dispatches through it. `from_solution_domain_data` /
  `from_transport_operator` gained a `strategy` param.
- **`create_multi_group_operator`** is now **mass-strategy-native**: `mass_operator_cls` /
  `mass_operator_cls_kwargs` → a single `mass_strategy: MassStrategy = VmapMass()` arg (matches the
  transport strategy args).
- **Call sites**: `MassOperatorVmap` → `MassOperator` (default strategy == `VmapMass`, so behavior is
  identical) in tests (`test_anisotropic_scattering`, `test_multigroup_derivative`,
  `test_transport_operator`, `test_moment_scatter_operator`, `test_breeding_blanket`) and examples.
  `mass_operator/__init__.py` exports `MassOperator` + the strategy classes (dropped
  `MassOperatorVmap`).
- **Validated here**: `VmapMass ≡ LaxMass` (exact, 0.0), default ≡ `VmapMass`, symmetry/adjoint
  (~1e-7 float32), jit, in/out structures, `lx.is_symmetric` True, whole package imports; affected
  test modules collect cleanly (only the pre-existing `sympy`-import errors remain). Full numerical
  run pending on stellsim.

### Stage A.3 — uniform strategy naming across the operator family (this session)

The three strategy-ified operator families had drifted to three different dispatch-method names
(`__call__` on transport forward/inversion, `forward`/`transpose` on moment/discrete, `apply` on
mass) and an overloaded `forward_strategy` field. Unified on a single vocabulary:

- **Method = the action.** Every strategy method is named for the action it performs: `forward`,
  `transpose`, `invert`. So transport's `ForwardStrategy.__call__` → `forward`,
  `InversionStrategy.__call__` → `invert` (and the pallas/warp inversion strategies likewise); mass's
  `apply` → `forward`. Moment/discrete (`forward`/`transpose`) and `AngleLifting`
  (`forward`/`invert`) were already on-convention. Call sites read self-documentingly now
  (`self.inversion_strategy.invert(self, b)` instead of `self.inversion_strategy(self, b)`).
- **Field = `<role>_strategy`, collapsing to `strategy` when there is a single implementation.**
  Transport keeps `forward_strategy` / `inversion_strategy` / `angle_lifting` — genuinely independent
  roles. Moment/discrete/mass renamed `forward_strategy` → `strategy` (one whole-operator
  implementation serving all directions; "forward" was a misnomer). `from_*` kwargs renamed to match.
- **ABC / concrete names left as-is** (role-named for transport's reusable roles, operator-named for
  the operator-bound ones; concrete names already follow `<Impl><Noun>`).
- **Validated here**: transport strategies expose `forward`/`invert`; mass/moment/discrete expose a
  `strategy` field (no `forward_strategy`); mass `VmapMass≡LaxMass`, moment `batched≡streaming`, and
  moment/discrete adjoints all hold; jit + whole-package import OK. Warp/pallas paths are type-only
  edits — exercised on stellsim.

**Validation done here** (domain-independent, since `jax_sbgeom`/`sympy` are absent in this env —
they live on stellsim): with a TN quadrature set, batched≡streaming for forward & transpose (~1e-16),
moment/discrete adjoint `⟨Mx,y⟩=⟨x,Mᵀy⟩` (~1e-14), jit + in/out structures OK, whole package imports.
**Still to run on stellsim**: `test_moment_scatter_operator.py`, `test_anisotropic_scattering.py`,
and the multigroup tests (MGOp constructor signature changed: `quadrature_set` added before
`n_energy_groups`).

### Stage B — `SingleGroupOperator` — DONE (this session)

**Decision taken (user): the full-symmetry composite form**, so Stage C's `MultiGroupOperator`
reuses the same shape. `SingleGroupOperator` now owns a canonical `setup` + a `TransportSpec`, and
the transport operator itself is a derived view (not just mass/scatter).

- **New `operators/transport_operator/transport_spec.py`** — `TransportSpec(eqx.Module)` bundling
  transport config minus the shared `SolutionDomainData`: `sweep_data` + `forward_strategy` /
  `inversion_strategy` / `angle_lifting` (strategy defaults) + `energy_group`. Methods:
  `build(solution_domain_data) -> TransportOperatorMultipleAngles` (keeps transport-construction
  knowledge in the transport layer — the composite never names a strategy),
  `from_transport_operator(top, energy_group=None)`, `to_energy_group(eg)`. Exported from
  `jax_sn.operators.transport_operator`.
- **`SingleGroupOperator` fields** → `setup: SolutionDomainData`, `transport_spec: TransportSpec`,
  `moment_operator`, `discrete_operator`, `mass_strategy: MassStrategy`, `transpose_bool`.
  Dropped `transport_operator` / `mass_operator` / `inscatter_operator` *fields*; all three are now
  `@property` derived views (`transport_spec.build(setup)`; `MassOperator.from_transport_operator`;
  `ScatterOperator(setup.domain.cross_sections_indexed, moment.out_structure(), eg, eg)`). So the
  shared spatial + cross-section data lives in exactly **one** pytree location (`setup`).
- **New classmethod `SingleGroupOperator.from_operators(moment, transport, mass, discrete,
  transpose_bool=False)`** — builds from already-constructed sub-operators (extracts `setup` from
  transport, spec via `TransportSpec.from_transport_operator`, `mass_strategy` from `mass.strategy`).
  Note: it takes **no** inscatter arg — that's the duplication being removed (derived from `setup`).
- `to_energy_group` now swaps only `transport_spec.energy_group` (inscatter follows automatically).
  `in_structure`/`out_structure` simplified to `moment_operator.out_structure()`.
- **Consumers updated to a single `tree_at` on the canonical location**:
  `parametrisation/singlegroup_parametrisation.py` (`op.setup.domain.cross_sections_indexed.
  cross_section_data`, was two-target) and `tests/.../test_single_group_ad.py:37` (same).
  `MultiGroupOperator.to_single_group_operator` now calls `SingleGroupOperator.from_operators`.
  Direct constructions in `test_anisotropic_scattering.py` + `examples/single_group_perf_mem_analysis.py`
  switched to `from_operators`.
- **Validated here (jaxsbgeom env, CPU + warp):** full FE suite `-k "not warp"` 33 passed / 24 skipped
  (identical to pre-Stage-B baseline) and `test_warp_linear_tetrahedron.py` 18 passed.

### Stage B.1 — `SolutionSetup` + data-free `TransportSpec` (this session)

Follow-up to B after a design review: `sweep_data` was sitting on `TransportSpec`, which mixed *data*
into what should be pure config. Reframed around the **operator-defining / operator-agnostic** axis
(preferred over diff/non-diff, which is user-dependent and arbitrary):

- **operator-agnostic** = data needed to *compute* the action whose *value never changes the result*.
  Essentially just the sweep execution plan (`sweep_data.levels` / `levels_argsort` — any valid
  topological order yields the identical inverse).
- **operator-defining** = everything else that's array data (geometry, FEM matrices, geometric data,
  cross sections; angular quadrature/harmonics once folded in at Stage C). Which *subset* of these a
  parametrisation differentiates stays a `theta_base` concern — *not* encoded in the type (decision 4
  unchanged: convention, not container, for the diff boundary).

Changes:
- **New `operators/transport_operator/solution_setup.py`** — `SolutionSetup(eqx.Module)` =
  `solution_domain_data` (operator-defining) + `sweep_data` (operator-agnostic). This is the canonical
  ``setup``; Stage C folds the angular data (quadrature + harmonics) in here too.
- **`TransportSpec` is now data-free**: dropped the `sweep_data` field; holds only
  `forward_strategy` / `inversion_strategy` / `angle_lifting` / `energy_group`. `build(setup)` reads
  `setup.solution_domain_data` + `setup.sweep_data`. (`TransportOperator`'s own two fields are
  unchanged, so warp/pallas/kernels are untouched.)
- **`SingleGroupOperator.setup`** is now a `SolutionSetup` (was a bare `SolutionDomainData`);
  `from_operators` builds `SolutionSetup(top.solution_domain_data, top.sweep_data)`.
- Consumers' canonical path is now `op.setup.solution_domain_data.domain...` (parametrisation + AD test).
- **Known wart (deferred to Stage C):** `sweep_data` still carries its `angle`, which is
  operator-*defining* (a cached copy of the quadrature angle). When quadrature/harmonics fold into
  `SolutionSetup` at C, the sweep should read its angle from there and carry only the ordering, making
  `sweep_data` purely operator-agnostic.
- **Validated here (jaxsbgeom env, CPU + warp):** full FE `-k "not warp"` 33 passed / 24 skipped and
  `test_warp_linear_tetrahedron.py` 18 passed (identical to baseline).

### Stage C.1 — `MultiGroupOperator` ownership (this session)

The actual §1 fix for the multigroup operator: it now owns one canonical `SolutionSetup` and derives
the operators that previously duplicated data.

- **`MultiGroupOperator` fields** → `setup: SolutionSetup`, `transport_spec: TransportSpec`,
  `moment_operator`, `discrete_operator`, `quadrature_set`, `n_energy_groups`,
  `mass_strategy: MassStrategy`, `gather_scatter_operator_cls` (static), `transpose_bool`.
  Dropped the `transport_operator` / `mass_operator` / `gather_scatter_operator` *fields*; all three
  are now `@property` derived views (`transport_spec.build(setup)`,
  `MassOperator.from_transport_operator`, `gather_scatter_operator_cls(setup...cross_sections_indexed)`).
  `moment_operator`/`discrete_operator` stay stored (harmonics; C.2 derives them).
- **`*_loc` helpers collapsed to single canonical locations** (all three now return
  `m.setup.solution_domain_data...`). `convert_to_lower_block_triangular` / `lower_block_triangular`
  simplified accordingly. `transpose` is now a `transpose_bool` flip via `tree_at`.
- **New `MultiGroupOperator.from_operators(...)`** (mirrors the single-group one) for building from
  already-constructed sub-operators; `from_multigroup_operator` updated to the new fields. The adjoint
  subclass and all `mv`/solve/RHS functions are unchanged — they read sub-operators by attribute, so the
  derived properties just work.
- **`multigroup_parametrisation.py` reconstruction massively simplified**: the fragile 5-surgery
  `reconstruct_multigroup_operator_including_toposort` (with its `transport_operator.transport_operator...`
  paths) is now ~3 single-location `tree_at`s on `op.setup...` (domain rebuild folds vertices +
  face_normals + cross sections; then geom; then sweep). No `*_loc` tuples, no per-operator special-casing.
- **`create_multi_group_operator`** builds `SolutionSetup` + `TransportSpec`; dropped the dead
  `gather_scatter_operator_cls_kwargs` param (no caller used it; the class takes only cross sections).
- **Call sites**: `test_breeding_blanket.py` / `test_multigroup_derivative.py` direct constructions →
  `from_operators`; `test_single_group_ad.py:setup_mgop` now edits `transport_spec` strategies
  (`BatchedForward/BatchedSweep`) instead of `tree_at`-ing the (now derived) `transport_operator`.
- **Verified §1 dedup**: exactly **one** `CrossSectionData` node in the MGOp pytree, and a single
  `tree_at` on the canonical cross-section location updates *both* the derived transport and
  gather-scatter operators (no stale second copy).
- **Validated here (jaxsbgeom env, CPU + warp):** full FE `-k "not warp"` 33 passed / 24 skipped and
  `test_warp_linear_tetrahedron.py` 18 passed (identical to baseline).

### Stage C.2 — fold angular data into `SolutionSetup` (this session)

Completes the symmetric design: **every** sub-operator now derives from one canonical `SolutionSetup`.

- **New `AngularSetup`** (in `solution_setup.py`): `quadrature_set` + precomputed harmonics `wk_ylm`
  (weighted, moment operator) + `ylm` (unweighted, discrete operator). `from_quadrature_set(quad, l_max)`.
  The single home for the angular discretisation.
- **`SolutionSetup`** gains `angular_setup` → `solution_domain_data` (operator-defining spatial) +
  `angular_setup` (operator-defining angular) + `sweep_data` (operator-agnostic). Added
  `elements_basis_shape` passthrough; `SolutionDomainData` gained `elements_basis_shape` / `n_elements`.
- **`moment_operator` / `discrete_operator` are now derived `@property` views** on both
  `MultiGroupOperator` and `SingleGroupOperator` (built from `setup.angular_setup.wk_ylm`/`ylm` +
  `moment_strategy` / `discrete_strategy` config fields). The stored fields + the interim
  `quadrature_set` field are gone; `MultiGroupOperator.quadrature_set` is a property onto
  `setup.angular_setup.quadrature_set`.
- **`to_single_group_operator`** shares the *same* `SolutionSetup` (only the energy group differs) —
  full dedup including angular data. `from_operators` (both operators) build the `AngularSetup` from the
  supplied `quadrature_set` + the operators' harmonics; `create_multi_group_operator` builds it via
  `AngularSetup.from_quadrature_set` and gained `moment_strategy` / `discrete_strategy` args.
- **Call sites**: `test_anisotropic_scattering` (+`quadrature_set` to single-group `from_operators`);
  examples updated likewise.
- **Verified**: one `CrossSectionData` *and* one `AngularSetup` node in the MGOp pytree; mv / transpose /
  `to_single_group_operator` consistent.
- **Validated here (jaxsbgeom env, CPU + warp):** full FE `-k "not warp"` 33 passed / 24 skipped and
  `test_warp_linear_tetrahedron.py` 18 passed (identical to baseline).

### Stage C.2b — `angle` out of `sweep_data` (this session)

The final dedup: the transport direction now lives in exactly **one** place. (Initially deferred as
"benign duplication / kernel surgery", then done at the user's request -- and it turned out clean because
`transport_operator` is a derived `@property`, so the angle it carries is *transient*, not a second
stored copy.)

- **`SweepData` is now purely operator-agnostic** (`levels` + `levels_argsort`; no `angle`).
  `reverse` reverses only the ordering; `reconstruct(domain, angle)` / `_rebuilt_sort(domain, angle)`
  take the angle as an argument; `__getitem__` slices only the ordering; `from_domain_*` compute from the
  angle but don't store it.
- **`TransportOperator` gains an `angle` field**, injected by `TransportSpec.build` from
  `setup.angular_setup.quadrature_set.angles`. `_forward_data` returns `self.angle`; `_inversion_args`
  returns `(*sdd, xs, angle, sweep_data)`; `transpose` negates `self.angle` *and* reverses the ordering;
  `__getitem__` / `VmapAngles` slice/map `(angle, sweep_data)` in lockstep.
- **Kernels take `angle` explicitly**: the sweep kernels (`transport_inverse_sweep_*`), their pallas and
  warp counterparts, `_assert_shape`, and `_flatten_base_args` gained an `angle` parameter (the
  `batch_size` `static_argnums` shifted 6->7 accordingly). Forward kernels already took `angle`, so they
  were unaffected. (`_flatten_transport_operator` consumes `_inversion_args()`, so it matched the new
  order automatically.)
- **Call sites**: the direct `TransportOperator*` factory constructors (`TransportOperatorVmap/Batched`,
  the pallas + warp single-angle factories) gained an `angle` parameter; the test fixtures + examples
  pass it (from the quadrature, or the indexed operator's `.angle`). Source-broadcast sites that read
  `sweep_data.angle.shape[0]` now read `quadrature_set.angles.shape[0]`.
- **Verified**: exactly **one** array of the angle's shape in the stored MGOp pytree; `sweep_data` has
  only `{levels, levels_argsort}`; adjoint holds to ~1e-16.
- **Validated here (jaxsbgeom env, CPU + warp):** full FE `-k "not warp"` 33 passed / 24 skipped and
  `test_warp_linear_tetrahedron.py` 18 passed (identical to baseline).

## Verification (per stage, on stellsim)

`pip install -e .[test]` (sympy) then `python -m pytest tests/finite_elements -k "not warp" -q`.
Land each stage as its own commit, green before the next.
- A: `test_moment_scatter_operator.py`, `test_anisotropic_scattering.py`, multigroup tests.
- B: `test_single_group_ad.py`.
- C: `test_multigroup_derivative.py`, `test_breeding_blanket.py`, `test_anisotropic_scattering.py`.
- B/C extra: construct an MGOp via `create_multi_group_operator`, assert one `eqx.tree_at` on the
  single `cross_sections_indexed` location round-trips with no stale second copy.

## Resume

**The refactor is complete.** Stages A, B, B.1, C.1, C.2, C.2b are all done & green (FE + warp,
`jaxsbgeom` conda env). §1 (duplicated shared data) is fully resolved: both `SingleGroupOperator` and
`MultiGroupOperator` own one canonical `SolutionSetup` (spatial + angular + sweep) and a data-free
`TransportSpec`, and derive *every* sub-operator (transport, mass, gather-scatter, moment, discrete,
inscatter) as `@property` views. Every shared array now lives in exactly one pytree location -- including
the transport direction (C.2b): `SolutionSetup.angular_setup` is the sole home of the angle, and
`sweep_data` is purely the operator-agnostic ordering. Nothing outstanding.

Note (env): tests run with `/home/tbogaarts/miniconda3/envs/jaxsbgeom/bin/python` — it has
`jax_sbgeom`, `sympy`, and `warp` (so warp tests run locally too, contrary to the older notes).
