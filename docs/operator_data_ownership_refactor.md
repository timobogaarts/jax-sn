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
| A | Strategy-ify moment & discrete; resurrect streaming; rehome `quadrature_set` | **DONE** (validated standalone; FE suite pending on stellsim) |
| B | `SingleGroupOperator` → setup + specs + derived views | not started |
| C | `MultiGroupOperator` + canonical `SolutionSetup` (harmonics in setup) | not started |

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

### Stage B — `SingleGroupOperator` (next)

File: `operators/single_group_operator/single_group_operator_imp.py`.
- Fields → canonical owner + `moment_operator`, `discrete_operator`, mass/scatter **config**,
  `transpose_bool`. Drop the `mass_operator` / `inscatter_operator` *fields*; make them `@property`:
  `mass_operator = mass_operator_cls.from_transport_operator(transport_operator)`;
  `inscatter_operator = ScatterOperator(sdd.domain.cross_sections_indexed,
  moment_operator.out_structure(), eg, eg)`.
- `to_energy_group` replaces **only** the transport part (inscatter follows via its property).
- Consumers: `parametrisation/singlegroup_parametrisation.py:13` and
  `tests/.../test_single_group_ad.py:37` two-target `tree_at` → single-target on the SDD location.
- (Decide at B: keep SDD inside `transport_operator` for the single-group case, or already hoist a
  composite `setup`+`TransportSpec` here for full symmetry. Stage C needs the composite form.)

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

## Resume tomorrow

Stage A is committed-ready but **uncommitted** and **unvalidated on stellsim**. Next session:
1. Run the FE suite on stellsim to confirm Stage A; commit it.
2. Start Stage B (`SingleGroupOperator`).
