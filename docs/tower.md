# The operator tower — design plan

Status: **design note / plan**, not yet implemented. Distilled from an architecture review of the
transport / multigroup operator structure (July 2026). Scope here is *differentiable* fused kernels
(JAX-traceable); the opaque-kernel story (fast primal / reference AD) is sketched at the end and
deliberately deferred.

## Motivation: two problems with the current structure

1. **Angle lifting is a dishonest abstraction.** `InversionAngleLifting.invert(single_op, b)` claims
   to be a combinator that lifts the single-angle inverse over the angle axis. `VmapInversionAngles`
   honors that; the warp lifts do not — `WarpSequentialAngles.invert` never calls the single-angle
   strategy, it invokes its own fused multi-angle kernel using `single_op._inversion_args()` as a
   data bag. Symptoms:
   - Hard pairing constraints across supposedly independent fields
     (`WarpSequentialAngles.check_compatible` demands `isinstance(inversion_strategy, WarpSequentialSweep)`)
     — the axes are not actually orthogonal.
   - `SingleAngleInversionStrategy.default_inversion_lifting()` is an upward dependency (single-angle level
     knows about the multi-angle lifting concept).
   - `vmappable()` / `vmaps_inner()` exist only so `perf_tests/_discovery.py` can flag the one bad
     combination (vmap over an opaque kernel silently serialises) that the type system permitted.

2. **The multigroup operator violates the codebase's own strategy pattern.** `MultiGroupOperatorScan`
   vs `MultiGroupOperatorScanCheckpoint` differ by one boolean threaded into the same scan function —
   a checkpoint flag as a subclass. The factory takes `multi_group_operator_cls` + kwargs dict
   instead of a strategy instance. And the core algebra `x_g − M T_g⁻¹ H D · (scatter)` is re-spelled
   inline five times across the family (single-group mv, multigroup forward scan, transpose scan,
   adjoint scan, rhs-preconditioning helpers), so there is no seam at which to fuse it.

## The unifying pattern

At every level, an axis strategy is **either a combinator parameterized by the level below, or a
native kernel that swallows everything below** (a sum type: combinator | native). Genuine lifts own
their inner strategy as a field (composite pattern), so pairing is by construction instead of by
cross-field `check_compatible`. Schematically, a configuration reads like a composition:
`GroupsApply(ScanGroups()) ∘ ComposedSweep ∘ VmapAngles(VmapPallasSweep())` ≈ "Scan(sweep(Vmap(Pallas)))".

The levels *rhyme* but deliberately do **not** share a generic `AxisLift` abstraction: the angle axis
is block-diagonal (`T_g = ⊕_k T_gk`, a lift is literally a map over independent blocks) while the
group axis is scatter-coupled (what gets lifted over groups is the per-group chain inside a
group-global gather algebra). Two small independent class families beat one lens-based generic.

## The shared node: the sweep chain

Everything in the operator family is the **sweep chain** `C_g = M T_g⁻¹ H D` (or its transpose)
wrapped in different scatter algebra:

| use site                    | expression                                        |
|-----------------------------|---------------------------------------------------|
| `SingleGroupOperator`       | `x − C_g · S_gg x`                                |
| multigroup forward scan     | `x_g − C_g · U_g[x]`                              |
| adjoint scan                | `x_g − C_g(Tᵀ) · V_g[x]`                          |
| multigroup transpose scan   | `ζ_g = C_gᵀ · x_g`, then gather-backward combine  |
| rhs preconditioning (×3)    | `M T_g⁻¹ H (·)` / `M T_g⁻¹ H D (·)`               |

The scatter part (`S_gg` vs `U_g` vs `V_g`) is what distinguishes the operators; the chain is
invariant. Naming it creates the highest-leverage fusion seam: **one fused differentiable chain
kernel (e.g. mass+discrete folded into the sweep's source assembly, moment folded into the sweep
output accumulation) accelerates single-group, multigroup, adjoint, and all rhs preconditioning
simultaneously.**

Two distinct roles, do not conflate them (L3 vs L4a/b): `C` is the shared *factor*, exposed as a
first-class linear operator (`ChainOperator`) that the whole-mv strategy builds on demand — alongside
`SourceSweepOperator` (`M T⁻¹ H`, for right-hand sides). It is **not** the operator's own swappable
strategy — the whole single-group mv is `I − C·S_gg` and a whole-mv fused kernel absorbs the scatter and
the `I−` too, so it does not decompose as "swap the chain, keep the scatter". The operator's field is
therefore a whole-mv strategy (`SingleGroupStrategy` / `MultiGroupStrategy`, L4); its *composed* variant
holds a `SweepStrategy` (`ComposedSweep`) that produces both sweep operators, and one fused sweep is
picked up by both composed variants at once — the reuse argument above is preserved as a shared strategy,
not as the top-level field.

## The tower

```
data (pytree leaves, single canonical home — UNCHANGED)
  OperatorData = DiscretisationData(spatial: SpatialData + angular: AngularData) + NonDifferentiableData(sweep_data)

configs/strategies (data-free eqx.Modules)
  L1  SingleAngleForwardStrategy, SingleAngleInversionStrategy          single-angle kernels
  L2  MultiAngleInversionStrategy = VmapAngles(inner) | ScanAngles(inner) | WarpSequentialAngles | WarpLevelAngles…
  ──  MultiAngleTransportSpec  = forward_strategy + angle_inversion   (energy_group is a build param, not a field)
  L3  SweepStrategy  = ComposedSweep(moment, discrete, mass strategies + transport_spec) | FusedSweep…
        produces two linear operators, sharing one strategy:
          ChainOperator        C = M T⁻¹ D H   (moment→moment, the shared scatter-coupling factor)
          SourceSweepOperator  M T⁻¹ H         (angular source→moment, builds right-hand sides)
  L4a SingleGroupStrategy = ComposedSingleGroupStrategy(sweep: SweepStrategy) | FusedSingleGroupStrategy…
        whole single-group mv  I − C_g ∘ S_gg ; the composed variant builds ChainOperator from its sweep
  L4b MultiGroupStrategy  = GroupsApply(lifting: GroupLifting) | FusedMultiGroupStrategy…
        GroupLifting  = ScanGroups(checkpoint: bool) | VmapGroups
        GroupsApply's per-group body reuses the same SweepStrategy for its C factor
  L5  lx solvers (lineax boundary — opaque-kernel territory, later)
```

### L2 — `MultiAngleInversionStrategy` (replaces `InversionAngleLifting` + pairing machinery)

```python
class MultiAngleInversionStrategy(eqx.Module):
    @abstractmethod
    def invert(self, op, b): ...
    @abstractmethod
    def single_angle_strategy(self) -> SingleAngleInversionStrategy: ...   # for __getitem__ / indexing
    def required_capabilities(self) -> Tuple[type, ...]: ...    # INSTANCE method (see below)
    def default_sweep_data_type(self) -> Type[SweepData]: ...   # INSTANCE method

# combinators: genuine lifts, owning their inner strategy
class VmapAngles(MultiAngleInversionStrategy):
    inner: SingleAngleInversionStrategy
    def __check_init__(self):
        assert self.inner.vmappable(), "vmap over an opaque kernel silently serialises; use ScanAngles or a fused lift"
    def required_capabilities(self):   return self.inner.required_capabilities()
    def default_sweep_data_type(self): return self.inner.default_sweep_data_type()
    def single_angle_strategy(self):   return self.inner

class ScanAngles(MultiAngleInversionStrategy):
    inner: SingleAngleInversionStrategy   # legal over ANY inner, including opaque warp kernels

# native implementations: no inner, they ARE the kernel
class WarpSequentialAngles(MultiAngleInversionStrategy):
    def invert(self, op, b): return multi_angle_warp_sweep(*op._inversion_args(), b)
    def single_angle_strategy(self): return WarpSequentialSweep()
class WarpLevelAngles(MultiAngleInversionStrategy):
    def required_capabilities(self):   return (ProvidesLevels,)
    def default_sweep_data_type(self): return LeveledSweepData
```

Consequences:
- `vmaps_inner()` disappears; `vmappable()` survives but is consumed at *construction*
  (`VmapAngles.__check_init__`) instead of by external perf-test discovery. The `serializes` flag
  becomes redundant.
- `default_inversion_lifting()` inverts direction: the creation helper does
  `VmapAngles(inversion_strategy)` when vmappable, otherwise requires/looks up a native class
  (optionally via `SingleAngleInversionStrategy.native_multi_angle: Optional[Type[MultiAngleInversionStrategy]]`).
- Sweep-data declarations become **instance methods** that fold over the composition tree
  (combinators delegate to `inner`; natives declare their own). The current classmethod rationale
  ("answerable without an instance") dies with combinators, and nothing needs it: the creation
  factory and perf discovery both hold instances. `required_capabilities` stays the ground truth
  for `check_compatible`; `default_sweep_data_type` is a convenience resolver — if a combinator ever
  adds capabilities on top of its inner, resolve the union or fail at construction asking the caller
  to pass sweep data explicitly.
- `ScanAngles(WarpSequentialSweep())` — scan over an opaque single-angle kernel — becomes a
  legitimate, expressible configuration (today it's only expressible as the flagged footgun).
- `ForwardAngleLifting` is collapsed (multi-angle forward = vmap of `SingleAngleForwardStrategy`; only one
  implementation exists and warp has no fused forward). A fused forward, when it arrives, slots in
  as a mirrored `AngleForward` sum type.
- `MultiAngleTransportSpec` shrinks to `forward_strategy + angle_inversion` (the energy group became a
  `build(setup, energy_group)` parameter — see L4a).
- `TransportOperatorMultiAngle` **holds its data directly** (`spatial_data`, batched `sweep_data`,
  batched `angle`, `energy_group`, `forward_strategy`, `angle_inversion`) instead of wrapping a nested
  `TransportOperatorSingleAngle`. The single-angle operator is a **derived view** (`__getitem__` /
  `to_single_angle(k)`) built on demand from that data and `angle_inversion.single_angle_strategy()` —
  the same "data has one home, operators are derived" move as `to_single_group_operator(g)` one level up.
  This kills two warts of the old nesting: the stored "single-angle" op was secretly *batched* (a type
  that lied about itself), and the single-angle inversion strategy lived in two places (the nested op's
  `inversion_strategy` and `angle_inversion.single_angle_strategy()`), which a `VmapAngles` lift could
  silently desync. Now `angle_inversion` is the single home of that strategy. **[implemented]**

### L3 — sweep operators (`ChainOperator` / `SourceSweepOperator`), built from a `SweepStrategy`

The chain and the fixed-source sweep are **first-class linear operators**, not bare apply-methods, so
callers use the usual `.mv` / `.transpose()` / `.to_energy_group()` idiom (which collapses the
forward/transpose direction into the operator, instead of a verb per direction). Both share one
data-free `SweepStrategy` (the "how" — composed today, fusable later). Lives in
`single_group_operator/sweep_operators.py`.

```python
class SweepStrategy(eqx.Module):          # data-free; backs both operators
    @abstractmethod
    def apply(self, operator_data, energy_group, x): ...             # C x = M T⁻¹ D H x
    @abstractmethod
    def apply_transpose(self, operator_data, energy_group, x): ...   # Cᵀ x
    @abstractmethod
    def apply_source(self, operator_data, energy_group, q): ...      # M T⁻¹ H q
    @abstractmethod
    def apply_source_transpose(self, operator_data, energy_group, x): ...
    def check_structures(self, operator_data, energy_group): ...     # optional, no-op default

class ComposedSweep(SweepStrategy):
    transport_spec:    MultiAngleTransportSpec = default_transport_spec()   # owns the transport kernel too
    moment_strategy:   MomentStrategy   = BatchedMoment()
    discrete_strategy: DiscreteStrategy = BatchedDiscrete()
    mass_strategy:     MassStrategy     = VmapMass()
    # apply builds T = transport_spec.build(operator_data, energy_group) + the contraction ops, then
    #   moment(T.inv(discrete(mass(x)))). apply_source drops D. Transposes come free from sub-op transposes.

class ChainOperator(AbstractLinearOperator_sn):        # C, moment→moment
    operator_data: OperatorData; energy_group: int = 0
    strategy: SweepStrategy = ComposedSweep(); transpose_bool: bool = False
    # mv → strategy.apply / apply_transpose ; transpose()/to_energy_group() are free tree_at flips

class SourceSweepOperator(AbstractLinearOperator_sn):  # M T⁻¹ H, angular→moment
    # identical shape; mv → strategy.apply_source / apply_source_transpose
```

- **The strategy owns all four contraction/transport strategies and consumes data + a scalar group
  (`operator_data` + `energy_group`), never a built sub-operator.** Load-bearing, not stylistic: a
  `FusedSweep` must fold `H D` into the sweep's per-element source assembly, which needs the raw sweep
  ordering / cross sections / geometric factors from `operator_data` — impossible if handed a black-box
  transport operator to call `.inv()` on. `ComposedSweep` builds `transport_spec.build(operator_data,
  energy_group)` itself; `FusedSweep` reaches into `operator_data`'s arrays. (Mirrors the transport
  operator, whose strategies receive a raw `InversionData` bag, never the operator.)
- The two operators share **one** `SweepStrategy` instance, so a single `FusedSweep` accelerates the
  chain and the rhs sweep at once — and, via L4 reuse, single-group, multigroup, adjoint and rhs
  preconditioning simultaneously.
- `energy_group` is a **scalar threaded from the operator**, not a field of `MultiAngleTransportSpec`
  (now purely `forward_strategy + angle_inversion`). `transpose_bool` picks the direction. That is what
  lets the adjoint/transpose scans and the per-group multigroup loop reuse these operators verbatim.
- The sweep operators are **not** the group operator's swappable field: they are built by the *composed*
  whole-mv strategies of L4 (`ComposedSingleGroupStrategy.sweep`, later `GroupsApply.sweep`). A fused
  whole-mv strategy bypasses them entirely, which is why there is nothing to "require".
- A `FusedSweep` supplies both directions itself; in differentiable-fused land `apply_transpose` may
  default to `jax.linear_transpose(apply)` (lineax `linear_solve` is transposable), though an explicit
  transpose kernel is preferable.

### L4 — group operators are concrete final classes holding a whole-mv strategy

The operator's swappable field is a **whole-mv strategy** — the whole `I − C·S` for single-group,
the whole gather-scatter algebra for multigroup — not the `C` factor. The *composed* variant of each
builds the `C` factor (a `ChainOperator`) from its `SweepStrategy`; a *fused* variant swallows the
scatter and the `I−` too (which a bare sweep could not express). This makes single-group and multigroup
symmetric (`SingleGroupStrategy` ↔ `MultiGroupStrategy`) and lets the abstract group-operator bases
disappear — all variation lives in the strategy field.

```python
class SingleGroupOperator(AbstractLinearOperator_sn):   # concrete, final — no abstract base
    operator_data:  OperatorData       # unchanged
    energy_group:   int = 0            # intrinsic: this operator IS A_g. Single home (transport + scatter)
    strategy:       SingleGroupStrategy = ComposedSingleGroupStrategy()   # the ONE "how" field
    transpose_bool: bool = False
    # mv = strategy.mv_forward(operator_data, energy_group, x)  /  mvᵀ = strategy.mv_transpose(…)
    # chain_operator()/source_operator() forward to strategy; inscatter_operator is a pure operator_data fn
    # in_structure / out_structure derive from `operator_data` (n_moments + elements_basis_shape), NOT strategy

class MultiGroupOperator(AbstractLinearOperator_sn):    # concrete, final
    operator_data:      OperatorData
    strategy:           MultiGroupStrategy = GroupsApply(ScanGroups())   # nests the transport_spec in its sweep
    gather_scatter_cls: Type[GatherScatterOperator] = GatherScatterOperatorBroadcasted
    n_energy_groups:    int                                           # a range, not a scalar group
    transpose_bool:     bool = False
```

Single-group carries an intrinsic scalar `energy_group`; multigroup carries a *range* (`n_energy_groups`)
and loops. So `energy_group`-on-the-operator is a single-group move; the transport config lives entirely
in the strategy's sweep either way — the operator has one "how" field (`strategy`), not a `strategy`
sitting next to a separate `transport_spec`.

Strategies consume **data + a scalar group**, never the owning operator (same rule as the sweep strategy
and the transport operator's strategies — keeps them pure and fusion-friendly). They also build the
solver-facing sweep operators (`chain_operator`, `source_operator`).

```python
class SingleGroupStrategy(eqx.Module):      # whole single-group mv + sweep-operator builders
    @abstractmethod
    def mv_forward(self, operator_data, energy_group, x): ...
    @abstractmethod
    def mv_transpose(self, operator_data, energy_group, x): ...
    def chain_operator(self, operator_data, energy_group): ...   # raises if a fused mv exposes none
    def source_operator(self, operator_data, energy_group): ...
    def check_structures(self, operator_data, energy_group): ...

class ComposedSingleGroupStrategy(SingleGroupStrategy):
    sweep: SweepStrategy = ComposedSweep()      # produces ChainOperator + SourceSweepOperator
    def chain_operator(self, operator_data, energy_group):
        return ChainOperator(operator_data, energy_group, self.sweep)
    def source_operator(self, operator_data, energy_group):
        return SourceSweepOperator(operator_data, energy_group, self.sweep)
    def mv_forward(self, operator_data, energy_group, x):
        inscatter = build_inscatter_operator(operator_data, energy_group)
        return x - self.chain_operator(operator_data, energy_group).mv(inscatter.mv(x))
    def mv_transpose(self, operator_data, energy_group, x):
        inscatter = build_inscatter_operator(operator_data, energy_group)
        return x - inscatter.transpose().mv(self.chain_operator(operator_data, energy_group).transpose().mv(x))

class GroupsApply(MultiGroupStrategy):      # step 3; multigroup also needs its gather-scatter config
    lifting: GroupLifting = ScanGroups()
    sweep:   SweepStrategy = ComposedSweep()     # same sweep, reused across groups
    def mv_forward(self, operator_data, gather_scatter, n_energy_groups, x):
        def one_group(g):
            return x[g] - ChainOperator(operator_data, g, self.sweep).mv(gather_scatter.gather_forward(x, g))
        return self.lifting.map_groups(one_group, n_energy_groups)
    def mv_transpose(self, operator_data, gather_scatter, n_energy_groups, x):
        # pass 1: ζ_g = ChainOperator(…, g, sweep).transpose().mv per group (one lifting pass)
        # pass 2: x_g − gather_backward(ζ, g)  — current two-scan structure, same lifting
```

- `SingleGroupOperatorBase` collapses: the impl axis is now the `strategy` field, so there is one
  concrete `SingleGroupOperator`, no abstract base. `inscatter_operator` stays a derived view on the
  operator (a pure `operator_data`+`energy_group` function; a fused strategy would fold it in).
- `MultiGroupOperatorScan` / `MultiGroupOperatorScanCheckpoint` / the abstract base collapse into
  `GroupsApply(ScanGroups())` and `GroupsApply(ScanGroups(checkpoint=True))`.
- `VmapGroups` is legitimate (memory-hungry, parallel): energy enters only as an index into cross
  sections, so vmapping over `g` works.
- `AdjointMultiGroupOperator` stays a distinct operator type (its `transpose()` legitimately
  raises/differs), but its scan body reduces to
  `x[g] − ChainOperator(operator_data, g, sweep).transpose().mv(gather_backward(x, g))` — the
  duplication with the forward scan disappears.
- The three rhs-preconditioning loops become one `precondition_rhs(op, rhs, transposed: bool)` built on
  `source_operator()` (the `M T⁻¹ H` sweep).
- Factory: `multi_group_operator_cls` + kwargs dict → `strategy: MultiGroupStrategy` instance;
  `inversion_lifting=None` auto-resolution → `VmapAngles(...)`-or-native resolution.

## The load-bearing rule (setup/spec split, restated for kernels)

> **Strategies at every level are data-free. All array data enters through the operator (ultimately
> `operator_data`) at call time. A fused kernel never stores arrays; it reads them from the operator it
> is handed.**

Everything currently relied on falls out of this rule:
- `tree_at` surgery via `cross_sections_indexed_loc` etc. keeps working — a `FusedSweep` sees new
  cross sections automatically because it never copied them.
- `to_energy_group` stays a free integer swap on the operator's `energy_group` field, identical for
  composed and fused sweeps (the group index is threaded to the kernel, which indexes cross sections
  like `_forward_data` does today).
- **Differentiability is automatic** for traceable fused kernels: parameters live only in
  `operator_data`, so `jax.grad` w.r.t. cross sections flows through whichever primal path is active —
  no reference machinery needed.
- `in_structure` / `out_structure` derive from `operator_data` (harmonics + mesh shapes), never from
  strategies, so fully fused applies don't break lineax structure checks.
- `from_operators` / `from_multigroup_operator` / `to_single_group_operator` survive nearly
  verbatim; `to_single_group_operator(g)` shares `operator_data`, sets `energy_group = g`, hands the
  single-group strategy the same sweep strategy, and drops gather-scatter — the single-group operator
  is visibly "the multigroup operator minus the group axis."

Callers that used to read `op.moment_operator` / `op.transport_operator` / `op.mass_operator` to
hand-build `M T⁻¹ (D) H · x` (rhs helpers, the lower-triangular solver, tests) now go through the sweep
operators: `op.chain_operator().mv` (scatter chain, forward or `.transpose()`) and `op.source_operator().mv`
(fixed-source rhs). The five sub-operator `@property`s and the `_require_composed_chain` gate are **gone**
— a fused strategy simply raises from `chain_operator`, which is what excludes it from the block solvers;
no isinstance probing on the operator.

## Where a differentiable fused kernel plugs in, by ambition level

1. **Fuse the angle loop** — native `MultiAngleInversionStrategy` (warp lives here; a pallas multi-angle would too).
2. **Fuse the contraction chain** — native `SweepStrategy` (a `FusedSweep`): fold `H D` into the sweep's
   per-element source assembly and/or `M` into the sweep output accumulation. One class, benefits the
   whole operator family at once. **Point the first pallas fusion effort here** — highest leverage per seam.
3. **Fuse the sweeps across groups** — a `GroupedTransportInverse` doing all G sweeps in one kernel,
   called by `GroupsApply` instead of the per-group `T.to_energy_group(g).inv`. Add the hook only
   when such a kernel exists (YAGNI until then).
4. **Fuse the whole mv** — native `SingleGroupStrategy` / `MultiGroupStrategy`; must supply
   `mv_forward` **and** `mv_transpose`; the differentiable case needs nothing else.
5. **Fuse the whole solve** — L5, an `lx.AbstractLinearSolver`; out of scope now, and conveniently
   the level where the non-differentiable story is free (see below).

Monotonicity: each level up fuses more and reuses less — only climb when profiling says the seam
overhead matters. `perf_tests` discovery extends naturally: `MultiAngleInversionStrategy`, `SweepStrategy`,
`GroupLifting`, `SingleGroupStrategy`, `MultiGroupStrategy` are new strategy bases whose
`__subclasses__()` walk gives the benchmark matrix (combinators need one `overrides` entry to supply an
`inner`/`sweep`, machinery that already exists).

## Deferred: opaque (non-differentiable) fused kernels

Two mechanisms, chosen by where the kernel sits; nothing about the tower layout changes when these
arrive — they slot in at the same seams.

- **Opaque at a solve boundary (L5): no AD machinery at all.** lineax implicit diff treats
  `solver.compute` as a primal black box; it needs only a differentiable operator `mv` (for the
  `dA·sol` term) and a primal transposed solve (`solver.transpose`). This is exactly the existing
  `DirectInverseSolver` / `MultiGroupLowerTriangularBlockSolver` pattern, generalized.
- **Opaque at an apply boundary (L2–L4): `fast_primal` wrapper.** Must be `custom_jvp`, **not**
  `custom_vjp` — lineax applies `jax.jvp` to the operator's `mv`, and custom_vjp functions reject
  forward-mode. Primal = fused kernel; the *entire tangent rule* = `jax.jvp` of a reference
  (composed, traceable) implementation. The tangent graph is pure traceable JAX, linear in tangents,
  hence transposable → reverse mode (parameter cotangents) works too. Cost: one reference mv per AD
  linearization — O(1) vs O(10²–10³) primal applications under implicit diff.

  ```python
  def fast_primal(fast_fn, reference_fn):
      @eqx.filter_custom_jvp
      def apply(params, x): return fast_fn(params, x)
      @apply.def_jvp
      def apply_jvp(primals, tangents):
          y = fast_fn(*primals)
          _, dy = jax.jvp(reference_fn, primals, tangents)   # reference tangent, fully traceable
          return y, dy
      return apply
  ```

  Strategy interface addition when needed: `reference(self)` defaulting to the composed strategy at
  that level; fused strategies route `mv` through `fast_primal` in the base class, so kernel authors
  supply kernels, never AD rules. Contract to enforce in the strategy test harness: fast ≈ reference
  (primal agreement, and `jax.grad` through a small solve matches the pure-reference gradient) —
  if they disagree, AD silently computes the reference's derivative at the fused primal's output.

## Implementation order (each step lands green independently)

1. **`MultiAngleInversionStrategy` refactor** (L2) — self-contained; the discovery-driven strategy test harness
   covers it with essentially no new test code. Collapse `ForwardAngleLifting`; shrink
   `MultiAngleTransportSpec`; move sweep-data declarations to instance methods; delete
   `vmaps_inner`/`serializes` machinery; update `__getitem__` and the creation factory's
   lifting auto-resolution.
2. **Single-group whole-mv strategy** (L4a, do this first) — introduce `SingleGroupStrategy` with the
   default `ComposedSingleGroupStrategy`, which holds a `SweepStrategy`/`ComposedSweep` (L3). The sweep
   owns the moment/discrete/mass strategies **and** the `MultiAngleTransportSpec`, and produces two
   first-class operators (`ChainOperator` = `M T⁻¹ D H`, `SourceSweepOperator` = `M T⁻¹ H`). Drop
   `energy_group` from `MultiAngleTransportSpec` (→ `build(operator_data, energy_group)`) and put
   `energy_group` on the operator instead. Rename the operator's data field `setup → operator_data`.
   Collapse the abstract `SingleGroupOperatorBase`: one concrete `SingleGroupOperator` = `operator_data
   + energy_group + strategy + transpose_bool` (no `transport_spec` field). Derive
   `in_structure`/`out_structure` from `operator_data`. Move the single-group mv/mvᵀ through
   `chain_operator().mv`; expose `chain_operator()`/`source_operator()` as strategy-built operators and
   drop the five `*_operator` `@property`s + `_require_composed_chain`. Move `sweep_operators.py` into
   `single_group_operator/`. **[done]**
3. **Multigroup whole-mv strategy** (L4b) — `MultiGroupStrategy` / `GroupLifting`; collapse the
   multigroup subclass hierarchy into `GroupsApply(ScanGroups(checkpoint=…))` reusing the **same**
   `SweepStrategy` (moving the `transport_spec` off the multigroup operator and into that sweep, as done
   for single-group); make `MultiGroupOperator` a concrete final class; rework the adjoint scan onto the
   sweep operators; fold the three rhs-preconditioning loops into one `precondition_rhs`; change the
   factory signature (`multi_group_operator_cls` → `strategy`). (Interim: the multigroup operator still
   holds `transport_spec` + moment/discrete/mass strategies and `to_single_group_operator` wraps them
   into the single-group sweep.)
4. Later, driven by profiling: fused kernels at levels 1–4; opaque-kernel `fast_primal` machinery
   only when a non-differentiable fused kernel actually exists.

## Misc findings from the same review (independent of the tower)

- `MultiGroupOperator.from_multigroup_operator` passes fields positionally — silent misassignment
  risk if field order changes; prefer keyword construction after the refactor.
- The derived `transport_operator` property rebuilds via `transport_spec.build(setup)` per access —
  free under `filter_jit`, but looks expensive; worth a comment.
- `AdjointMultiGroupOperator._mv_transpose` raising `NotImplementedError` will surface deep inside a
  lineax solver that requests a transpose; the message should say what to use instead.
