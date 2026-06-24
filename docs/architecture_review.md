# Code review: jax-sn (architecture & coding patterns)

*Scope: `src/jax_sn/` (~10k lines, 60 files), package configs, and test layout. The `examples/`
directory and numerical logic are explicitly out of scope. Reviewed 2026-06-11 against `main`
(9b7ff76); where the `raf_bep` branch differs (pallas import guards) this is noted.*

## Overall impression

The high-level architecture is genuinely good: the `GeometricDomain → SolutionDomain →
Operators → solve` pipeline is clean, the operator-composition design (everything is a
`lineax.AbstractLinearOperator` with documented in/out structures) is coherent, and the
mathematical docstrings are excellent. The findings below are mostly about fragility and
hygiene, not about the design being wrong.

## 1. The biggest structural risk: shared data in multiple pytree locations

`MultiGroupOperator` stores the same `CrossSectionsIndexed`, `PrecomputedGeometricData`, and
`FEMMatrices` in two places each (transport operator and mass/gather-scatter operators), kept
in sync by hand via location lambdas (`cross_sections_indexed_loc` etc.,
`multi_group_operator_imp.py:312-333`). This is documented thoroughly, which helps, but it's an
invariant the type system can't enforce: any new operator class or any `eqx.tree_at` call that
touches one copy but not the other silently desynchronizes the operator.
`reconstruct_multigroup_operator_including_toposort` (`multigroup_parametrisation.py:122-155`)
is the most fragile consumer — five sequential tree surgeries with an `isinstance` special case
for the warp operator (already flagged with a TODO).

**Recommendation:** invert the ownership: one canonical store (e.g. `SolutionDomainData`) that
all sub-operators reference, so there is exactly one location to replace.

> **Update (2026-06-24): essentially resolved.** Addressed via the
> `operator = setup + spec + strategy` pattern (composite owns one canonical `setup`; sub-operators
> are derived views). Full design + staging in `operator_data_ownership_refactor.md`. Both
> `SingleGroupOperator` and `MultiGroupOperator` now own a single canonical `SolutionSetup`
> (`solution_domain_data` + `sweep_data`) plus a data-free `TransportSpec`, and derive
> transport/mass/inscatter/gather-scatter as `@property` views. The shared `CrossSectionsIndexed` /
> `PrecomputedGeometricData` / `FEMMatrices` now live in exactly **one** pytree location each — a
> single `eqx.tree_at` updates every consumer (verified: one `CrossSectionData` node in the MGOp
> pytree). The fragile `reconstruct_multigroup_operator_including_toposort` is now ~3 single-location
> surgeries with no `isinstance`/warp special-casing. The angular data (`quadrature_set` + harmonics)
> also folded into `SolutionSetup` (C.2), so moment/discrete derive from it too — **every** sub-operator
> is now a derived view onto one canonical setup. **Done.**


## 3. Transpose-via-subclass breaks substitutability

> **Update (2026-06-12): Fixed.** The moment, discrete, and single-group operators now carry a
> `transpose_bool` flag instead of a transposed subclass (the multi-group operator already did). As
> of Stage A (`operator_data_ownership_refactor.md`) the moment/discrete *implementation* axis is
> also flag/strategy-based, not subclass-based — both axes are now orthogonal flags/strategies.

`MomentOperatorTransposed(MomentOperator)`, `DiscreteOperatorTransposed(DiscreteOperator)`,
`SingleGroupOperatorTranspose(SingleGroupOperator)`: a transposed operator passes every
`isinstance(x, ForwardClass)` check in the codebase (and there are many such asserts, e.g.
`single_group_operator_imp.py:56-66`, `multi_group_operator_imp.py:193-201`). So the runtime
type checks that are clearly meant as guardrails cannot catch a transposed operator passed
where a forward one is expected — the operators have swapped in/out structures, so this is
exactly the mix-up you'd want caught. A sibling-class or `transpose_bool` flag design (which
`MultiGroupOperator` already uses) would be more consistent and safer.

Relatedly, `QuadratureSetReduced` used to subclass `QuadratureSet` and override `__getitem__`
to unconditionally `raise` — a Liskov violation. **Fixed:** an abstract `QuadratureSetBase`
was introduced holding the shared state (`angles`, `weights`, `n_angles`, the shape
`__check_init__`) plus the abstract `wk_ylm_real`/`ylm_real` interface; both `QuadratureSet` and
`QuadratureSetReduced` inherit from it as siblings. A reduced set therefore no longer passes
`isinstance(x, QuadratureSet)`, and `__getitem__` (single-angle access, meaningless once angles
are collapsed) lives only on `QuadratureSet`. Both harmonic methods are abstract — even
`ylm_real`, which happens to coincide today, may be computed differently by a future set — so the
base commits to state and interface but no algorithm. The `wk_ylm_real`/`ylm_real` asymmetry was
*not* a bug: `ylm_real` is the unweighted harmonics at the reduced angles, whereas `wk_ylm_real`
must weight with the original weights and scatter-accumulate over collapsed angles.

## 4. Two competing pytree idioms, and a risky subclassing of one

The codebase mixes `@jax.tree_util.register_dataclass + @dataclass(frozen=True)` (domain, FEM
data, `Source`) with `eqx.Module` (operators, `CrossSectionsIndexed`,
`LevelBasedParallelism`) — sometimes within the same conceptual family (`GeometricDomain` is a
dataclass, but `CrossSectionsIndexed` inside `Domain` is an eqx.Module). Pick one; eqx.Module
gives `__check_init__`, field defaults, and static fields uniformly. Two specific hazards:

- `Domain(GeometricDomain)` and `TopologicalSortedAnglePrecomputedLevels(TopologicalSortedAngle)`
  subclass already-registered dataclasses; the latter adds six fields all defaulting to `None`
  (`transport_operator_levels.py:45-56`), so a half-initialized instance is constructible and
  `jax.tree.map` over it will behave differently depending on which fields are `None`.
- `GeometricDomain` uses `eq=False` for id-based hashing to feed `lru_cache` for the BVH
  (`geometric_domain.py:39-46`, `390`). This works but couples cache correctness to object
  identity — any pytree surgery produces a "new" domain and a new cache slot. It's documented;
  just be aware it's a global cache keyed on identity (the guard at `geometric_domain.py:425`
  against tracers is good).

## 5. Layering and import structure

- **Circular-import workarounds are endemic**: a second import block in the middle of
  `geometric_domain.py` (line 144), function-local imports throughout
  (`geometric_domain.py:296, 380, 406, 429`, `setup_test_mgop.py`,
  `multigroup_parametrisation.py:123` which does `import jax_sn` *inside a jitted function* and
  then reaches into `jax_sn.solution_domain.basix_fem._precompute_jacobians_simplex`, a private
  function of another layer). These are symptoms of unclear dependency direction between
  `domain`, `tools`, and `solution_domain`.
- `domain/geometric_domain.py` imports `QuadratureSet` — the mesh layer now depends on the
  angular-discretization layer, only because `TopologicalSortedAngle` (a sweep concern) lives
  in `domain`. Moving toposort to the transport-operator layer would clean the dependency
  direction.
- `multigroup_parametrisation.py:145` references
  `jax_sn.operators.transport_operator.transport_operator_warp...` directly; since that module
  is imported under `try/except` in `transport_operator/__init__.py`, this line raises
  `AttributeError` whenever warp isn't installed — even for users not using warp.

## 6. Optional-dependency handling is inconsistent and partly broken

Multiple strategies coexist: warp gets a silent `try/except` module import, `jax_sbgeom` gets
function-local imports with a friendly error (`mesh_tools.py:360-363`) — but
`domain_generation.py:8` imports it unguarded. Pallas is imported unconditionally on `main`;
the `raf_bep` branch adds `try/except` guards with placeholder classes that raise on
instantiation (the nicest of the three patterns — worth adopting uniformly, though note that on
`raf_bep` `_pallas_available` is then computed twice independently, in
`transport_operator/__init__.py` and `transport_operator_base.py`). Additionally:

- `pyproject.toml` declares none of `warp-lang`, `jax_sbgeom`, or triton as optional extras,
  yet core paths (`create_cartesian_setup`, hence `setup_test_mgop`) hard-require `jax_sbgeom`,
  including private functions of it (`domain_generation.py:75,78`).
  **Update (2026-06-12): Done.** `[project.optional-dependencies]` now declares `warp`, `pallas`,
  `geometry` (jax_sbgeom), and `test` (pytest, sympy) extras.
- `transport_operator/__init__.py` lists `'transport_operator_warp'` in `__all__`
  unconditionally, so `from ... import *` raises `AttributeError` when warp is missing.
- `h5py` is a declared dependency but appears unused in `src/`. **Update (2026-06-12): Done** —
  removed from `dependencies`.

## 7. Validation strategy: `assert` everywhere

Nearly all runtime validation (shape checks, type checks, compatibility checks) is bare
`assert`, which disappears under `python -O` and blurs the line between "internal invariant"
and "user input validation" (e.g. `Domain.from_mesh_and_cross_sections` validating user data
with asserts). Two concrete instances are worse than style:

- `AbstractLinearOperator_sn.as_matrix` (`base_linear_operator.py`): `assert False` as a
  control-flow throw, followed by unreachable `return jnp.array([[]])` where `jnp` isn't even
  imported in that module. Should be `raise NotImplementedError(...)`. **Update (2026-06-12): Done**
  (now `raise NotImplementedError(...)`; the dead return and `jax-snj` typo removed).
- `MultigroupOperatorParametrisedObjective.__check_init`
  (`multigroup_parametrisation.py:81`) is missing the trailing underscores — equinox never
  calls it, so those checks are silently dead. **Update (2026-06-12): Done** (renamed to
  `__check_init__`).

## 8. Dead code, debug artifacts, and accidental imports

- `_mv_single_group_operator` (`single_group_operator_imp.py:152-153`): **two consecutive
  `return` statements with different operator orderings** (`discrete∘mass` vs `mass∘discrete`).
  The second is unreachable. This looks like an experiment that was never cleaned up — and it
  leaves the reader unsure which ordering is intended.
- `linear_tetrahedron_warp_kernel.py:4,6`: `from tracemalloc import start` and
  `from unittest import result` — classic IDE auto-import accidents. **Update (2026-06-12): Done**
  (removed).
- `compute_levels_iterative` (`graph_tools.py:155`): stray `print(n_elements_computed_new)` on
  a hot setup path. **Update (2026-06-12): Done** (removed).
- `parametrisation_tools.py`: research-script artifacts in library code — `print` statements
  throughout `evaluate_values`/`evaluate_jac`, `import time` at line 209 mid-file, hard-coded
  `save_dir = "iterations/"` writing `.npy` files, a custom `__init__` on a `@dataclass` that
  sets an undeclared `self._cache_result` (line 162; the real field is `_cache_val`, which only
  works because the dataclass field default papers over it), and the orphan `ks_agg` utility at
  the bottom.
- `_extract_nD_quadrature_set` (`quadrature_set_base.py:74-78`): `weights_red` computed twice
  identically, with a commented-out assertion between.
- Duplicate imports: `Tuple` twice (`geometric_domain.py:7,10`), `FEMMatrices` twice
  (`transport_operator_imp.py:4-5`, `transport_operator_pallas.py:4-5`), `Element` twice
  (`solution_domain.py:4,10`), repeated lines in `domain/__init__.py`.
- Apparent orphans worth auditing: `_call_streaming_moment_operator(_transposed)`
  (`moment_operator_imp.py:209-283`), `_setup_elements_from_args`
  (`transport_operator_imp.py:278`), `_multiple_angle_level_bounds_dynamic` vs
  `multiple_angle_level_bounds_safe` (near-duplicates, `graph_tools.py:294-313`),
  `MassOperatorLax` (defined, never exported in `__init__`), `SolverDiagnostics` doing
  dataclass-default sharing as `solver_diagnostics : SolverDiagnostics = SolverDiagnostics(...)`.
  **Update (2026-06-12):** the streaming moment kernels were **not dead by intent** — they were a
  real second implementation with no selection mechanism. Stage A resurrected them as the
  `StreamingMoment` strategy (`moment_operator_strategies.py`); the rest still stand.

## 9. Docstring drift

The docs are a strength, but many were copy-pasted and not updated, which actively misleads:
`TransportOperatorMultipleAngles.__getitem__` documents an "Energy group index" but indexes
angles (`transport_operator_base.py`); `transport_inverse_sweep_complete_vmap_pallas` documents
a `batch_size` parameter it doesn't have; `_call_streaming_moment_operator_transposed`'s
docstring is the forward operator's; `transport_inverse_sweep_multiple_angles_complete_vmap`'s
body text says "takes significantly less memory than the complete vmap version" — but it *is*
the complete vmap version; `_check_if_lower_triangular` documents a `mgop` parameter it doesn't
take. There are also recurring typos ("Mathetmatical", "hasing", "jax-snj",
"hardcodedfunction").

## 10. Smaller items

- Mutable default arguments (`= {}` six times) in `create_multi_group_operator`
  (`multi_group_operator_creation.py:12-22`). Harmless today (never mutated) but a known Python
  trap.
- `ParametrisedModule` (`parametrisation_tools.py:23-31`) declares
  `@staticmethod @abstractmethod def theta_base(self, module)` — a staticmethod with `self`,
  and implementations use a different arity (`theta_base(operator)`). The class is really a
  namespace of functions; either make it a `Protocol` or drop the fake `self`.
- The per-class `@lx.is_symmetric.register` / `@lx.linearise.register` boilerplate repeated for
  every operator subclass is redundant — singledispatch resolution follows the MRO, so the
  registration on `AbstractLinearOperator_sn` already covers them (only `MassOperator`'s `True`
  override is needed). Eight near-identical blocks could be deleted.
- Top-level `jax_sn/__init__.py` omits `source` and `analytical_solutions` from the package
  imports/`__all__` while including everything else — looks like an oversight rather than
  intent.
- In `basix_fem.py:63-65`, a leftover half-edited line (`face_vertices =
  basix.cell.topology(...)` computed, then immediately recomputed with the subscript) and
  `type` shadowing the builtin at line 159.
- `tests/` contains a checked-in notebook (`initial_tests.ipynb`) and
  `stell_blanket_functions.py` at the top level alongside the real suites — worth segregating
  into a `benchmarks/` or `notebooks/` area.

## Suggested priority

1. ~~Fix the two latent constructor-order bugs (§2)~~ — done. ~~dead double-`return` (§8)~~ — done.
2. ~~Declare optional-dependency extras in `pyproject.toml` (§6)~~ — done. One pytree idiom (§4)
   still open.
3. ~~`__check_init` typo (§7)~~ — done. ~~`as_matrix` assert→raise (§7)~~ — done. The warp
   `AttributeError` path (§5) still open.
4. ~~Longer term: single ownership for shared operator data (§1)~~ — **done** (Stages A, B, B.1, C.1,
   C.2), see `operator_data_ownership_refactor.md`. ~~sibling-class / flag transposes (§3)~~ — done.

### Progress log (2026-06-12)
- Quick cleanups: §7 (`as_matrix`, `__check_init__`), §8 (warp-kernel stray imports, `graph_tools`
  print), §6 (`pyproject` extras, drop `h5py`).
- §3: moment/discrete/single-group transposes are flag-based; moment/discrete impl axis is now
  strategy-based (no subclass).
- §1: Stage A of the data-ownership refactor (moment/discrete `setup+spec+strategy`; streaming
  resurrected; `quadrature_set` rehomed onto `MultiGroupOperator`). Stages B/C pending.
