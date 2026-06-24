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
| C | `MultiGroupOperator` + fold angular data (quadrature + harmonics) into `SolutionSetup` | not started |

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

### Stage C — `MultiGroupOperator` + `SolutionSetup`

Files: `operators/multi_group_operator/multi_group_operator_imp.py`,
`solution_domain/solution_domain.py`, `multi_group_operator_creation.py`.
- Introduce canonical `SolutionSetup` = spatial (`SolutionDomainData`: domain+fem+geom) + angular
  (`quadrature_set` + precomputed `wk_ylm`/`ylm` via `quadrature_set.wk_ylm_real(l_max)` /
  `ylm_real(l_max)`). Fold the interim `MGOp.quadrature_set` field into it.
- Composite stores one `setup` + `transport_spec` + `moment_spec` + `discrete_spec` + static config;
  all sub-operators become `@property`. Rewrite `transpose`, `from_multigroup_operator`,
  `to_single_group_operator`, `__check_init__`, `convert_to_lower_block_triangular` (single canonical
  location); collapse `*_loc` helpers to single-target lambdas (keep names for API).
- Consumers: `multigroup_parametrisation.py` reconstruction simplifies (flat `op.setup.domain...`
  instead of `op.transport_operator.transport_operator.solution_domain_data...`); read-only
  consumers (`mv` impls, `adjoint_multigroup_operator.py`, `right_hand_side.py`,
  `lower_triangular_solver.py`) need no change — properties return real operators.

## Verification (per stage, on stellsim)

`pip install -e .[test]` (sympy) then `python -m pytest tests/finite_elements -k "not warp" -q`.
Land each stage as its own commit, green before the next.
- A: `test_moment_scatter_operator.py`, `test_anisotropic_scattering.py`, multigroup tests.
- B: `test_single_group_ad.py`.
- C: `test_multigroup_derivative.py`, `test_breeding_blanket.py`, `test_anisotropic_scattering.py`.
- B/C extra: construct an MGOp via `create_multi_group_operator`, assert one `eqx.tree_at` on the
  single `cross_sections_indexed` location round-trips with no stale second copy.

## Resume

Stage A is **committed**. Stages B + B.1 are **done & green** (FE + warp, in the `jaxsbgeom` conda
env). Next session:
1. Commit B + B.1 (if not already committed).
2. Start Stage C (`MultiGroupOperator`). The composite `setup + spec + derived views` shape now exists
   on `SingleGroupOperator` — including the canonical `SolutionSetup` (`solution_domain_data` +
   `sweep_data`) and the data-free `TransportSpec`. Stage C: (a) give `MultiGroupOperator` one
   `SolutionSetup` + per-energy-group specs, sub-operators as `@property`, collapsing the `*_loc`
   helpers to single locations; (b) **fold the angular data (`quadrature_set` + precomputed
   `wk_ylm`/`ylm`) into `SolutionSetup`** and let `sweep_data` read its angle from there (closing the
   B.1 wart), so moment/discrete also derive from the canonical setup.

Note (env): tests run with `/home/tbogaarts/miniconda3/envs/jaxsbgeom/bin/python` — it has
`jax_sbgeom`, `sympy`, and `warp` (so warp tests run locally too, contrary to the older notes).
