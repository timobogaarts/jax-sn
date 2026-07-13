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
`GroupsApply(ScanGroups()) ∘ ComposedChain ∘ VmapAngles(VmapPallasSweep())` ≈ "Scan(chain(Vmap(Pallas)))".

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

## The tower

```
data (pytree leaves, single canonical home — UNCHANGED)
  SolutionSetup = DiscretisationData(spatial: SpatialData + angular: AngularData) + NonDifferentiableData(sweep_data)

configs/strategies (data-free eqx.Modules)
  L1  SingleAngleForwardStrategy, SingleAngleInversionStrategy          single-angle kernels
  L2  MultiAngleInversionStrategy = VmapAngles(inner) | ScanAngles(inner) | WarpSequentialAngles | WarpLevelAngles…
  ──  MultiAngleTransportSpec  = forward_strategy + angle_inversion + energy_group
  L3  SweepChain     = ComposedChain(moment, discrete, mass strategies)
                     | FusedChain…                 (native, swallows L3's algebra)
  L4a SingleGroupOperator algebra: I − C_g ∘ S_gg          (fixed, tiny, final class)
  L4b MultiGroupApply = GroupsApply(lifting: GroupLifting) | FusedMultigroupApply…
        GroupLifting  = ScanGroups(checkpoint: bool) | VmapGroups
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
- `MultiAngleTransportSpec` shrinks to `forward_strategy + angle_inversion + energy_group`.
- `TransportOperatorMultiAngle` **holds its data directly** (`spatial_data`, batched `sweep_data`,
  batched `angle`, `energy_group`, `forward_strategy`, `angle_inversion`) instead of wrapping a nested
  `TransportOperatorSingleAngle`. The single-angle operator is a **derived view** (`__getitem__` /
  `to_single_angle(k)`) built on demand from that data and `angle_inversion.single_angle_strategy()` —
  the same "data has one home, operators are derived" move as `to_single_group_operator(g)` one level up.
  This kills two warts of the old nesting: the stored "single-angle" op was secretly *batched* (a type
  that lied about itself), and the single-angle inversion strategy lived in two places (the nested op's
  `inversion_strategy` and `angle_inversion.single_angle_strategy()`), which a `VmapAngles` lift could
  silently desync. Now `angle_inversion` is the single home of that strategy. **[implemented]**

### L3 — `SweepChain`

```python
class SweepChain(eqx.Module):
    @abstractmethod
    def apply(self, t_op, mass_op, moment_op, discrete_op, source):
        '''M T⁻¹ H D · source. Caller supplies t_op already at the right group / direction.'''
    @abstractmethod
    def apply_transpose(self, t_op, mass_op, moment_op, discrete_op, x):
        '''Dᵀ Hᵀ T⁻ᵀ Mᵀ · x.'''

class ComposedChain(SweepChain):
    moment_strategy:   MomentStrategy   = BatchedMoment()
    discrete_strategy: DiscreteStrategy = BatchedDiscrete()
    mass_strategy:     MassStrategy     = VmapMass()
    # apply = moment(t_op.inv(discrete(mass(source)))) — today's inline chain, written once.
    # apply_transpose comes free from sub-operator transposes, exactly as now.
```

- The chain is *agnostic about which transport operator drives it* — the caller passes
  `T.to_energy_group(g)` or its transpose. That single choice is what lets the adjoint and the
  transpose scans reuse it verbatim.
- The three contraction strategies move off the group operators into `ComposedChain`.
- A `FusedChain` supplies both directions itself; in differentiable-fused land
  `apply_transpose` may default to `jax.linear_transpose(apply)` (lineax `linear_solve` is
  transposable), though an explicit transpose kernel is preferable.

### L4 — group operators become concrete, final classes

```python
class SingleGroupOperator(AbstractLinearOperator_sn):   # concrete, final
    setup:          SolutionSetup      # unchanged
    transport_spec: MultiAngleTransportSpec
    chain:          SweepChain = ComposedChain()
    transpose_bool: bool = False
    # mv:  x − chain.apply(T_g, …, inscatter.mv(x))
    # mvᵀ: x − inscatterᵀ.mv(chain.apply_transpose(T_g, …, x))

class MultiGroupOperator(AbstractLinearOperator_sn):    # concrete, final
    setup:              SolutionSetup
    transport_spec:     MultiAngleTransportSpec
    chain:              SweepChain      = ComposedChain()
    apply:              MultiGroupApply = GroupsApply(ScanGroups())
    gather_scatter_cls: Type[GatherScatterOperator] = GatherScatterOperatorBroadcasted
    n_energy_groups:    int
    transpose_bool:     bool = False
```

```python
class GroupsApply(MultiGroupApply):
    lifting: GroupLifting = ScanGroups()
    def mv_forward(self, op, x):
        T = op.transport_operator                      # derived property, as today
        def one_group(g):
            return x[g] - op.chain.apply(T.to_energy_group(g), ...,
                                         op.gather_scatter_operator.gather_forward(x, g))
        return self.lifting.map_groups(one_group, op.n_energy_groups)
    def mv_transpose(self, op, x):
        # pass 1: ζ_g = chain.apply_transpose per group (one lifting pass)
        # pass 2: x_g − gather_backward(ζ, g)  — current two-scan structure, same lifting
```

- `MultiGroupOperatorScan` / `MultiGroupOperatorScanCheckpoint` / the abstract base collapse into
  `GroupsApply(ScanGroups())` and `GroupsApply(ScanGroups(checkpoint=True))`.
- `VmapGroups` is legitimate (memory-hungry, parallel): energy enters only as an index into cross
  sections, so vmapping over `g` works.
- `AdjointMultiGroupOperator` stays a distinct operator type (its `transpose()` legitimately
  raises/differs), but its scan body reduces to
  `x[g] − chain.apply(T.transpose().to_energy_group(g), …, gather_backward(x, g))` — the duplication
  with the forward scan disappears.
- The three rhs-preconditioning loops become one `precondition_rhs(op, rhs, transposed: bool)` on
  the chain.
- Factory: `multi_group_operator_cls` + kwargs dict → `apply: MultiGroupApply` instance;
  `inversion_lifting=None` auto-resolution → `VmapAngles(...)`-or-native resolution.

## The load-bearing rule (setup/spec split, restated for kernels)

> **Strategies at every level are data-free. All array data enters through the operator (ultimately
> `setup`) at call time. A fused kernel never stores arrays; it reads them from the operator it is
> handed.**

Everything currently relied on falls out of this rule:
- `tree_at` surgery via `cross_sections_indexed_loc` etc. keeps working — a `FusedChain` sees new
  cross sections automatically because it never copied them.
- `to_energy_group` stays a free integer swap on `transport_spec`, identical for composed and fused
  chains (the group index is threaded to the kernel, which indexes cross sections like
  `_forward_data` does today).
- **Differentiability is automatic** for traceable fused kernels: parameters live only in `setup`,
  so `jax.grad` w.r.t. cross sections flows through whichever primal path is active — no reference
  machinery needed.
- `in_structure` / `out_structure` derive from `setup` (harmonics + mesh shapes), never from
  strategies, so fully fused applies don't break lineax structure checks.
- `from_operators` / `from_multigroup_operator` / `to_single_group_operator` survive nearly
  verbatim; `to_single_group_operator(g)` shares `setup`, swaps the spec's group, passes the same
  chain instance, and drops `apply` + gather-scatter — the single-group operator is visibly "the
  multigroup operator minus the group axis."

Compatibility shim: code reading `op.moment_operator` / `op.discrete_operator` / `op.mass_operator`
(rhs helpers, tests) keeps working via derived `@property`s that delegate to the chain when it is a
`ComposedChain`, and raise informatively for a fused chain (a fused chain has no standalone moment
operator to expose — the seam was fused away; the raise is honest).

## Where a differentiable fused kernel plugs in, by ambition level

1. **Fuse the angle loop** — native `MultiAngleInversionStrategy` (warp lives here; a pallas multi-angle would too).
2. **Fuse the contraction chain** — native `SweepChain`: fold `H D` into the sweep's per-element
   source assembly and/or `M` into the sweep output accumulation. One class, benefits the whole
   operator family at once. **Point the first pallas fusion effort here** — highest leverage per seam.
3. **Fuse the sweeps across groups** — a `GroupedTransportInverse` doing all G sweeps in one kernel,
   called by `GroupsApply` instead of the per-group `T.to_energy_group(g).inv`. Add the hook only
   when such a kernel exists (YAGNI until then).
4. **Fuse the whole mv** — native `MultiGroupApply`; must supply `mv_forward` **and** `mv_transpose`;
   the differentiable case needs nothing else.
5. **Fuse the whole solve** — L5, an `lx.AbstractLinearSolver`; out of scope now, and conveniently
   the level where the non-differentiable story is free (see below).

Monotonicity: each level up fuses more and reuses less — only climb when profiling says the seam
overhead matters. `perf_tests` discovery extends naturally: `MultiAngleInversionStrategy`, `SweepChain`,
`GroupLifting`, `MultiGroupApply` are new strategy bases whose `__subclasses__()` walk gives the
benchmark matrix (combinators need one `overrides` entry to supply an `inner`, machinery that
already exists).

## Deferred: opaque (non-differentiable) fused kernels

Two mechanisms, chosen by where the kernel sits; nothing about the tower layout changes when these
arrive — they slot in at the same seams.

- **Opaque at a solve boundary (L5): no AD machinery at all.** lineax implicit diff treats
  `solver.compute` as a primal black box; it needs only a differentiable operator `mv` (for the
  `dA·sol` term) and a primal transposed solve (`solver.transpose`). This is exactly the existing
  `TransportOperatorSolver` / `MultiGroupLowerTriangularBlockSolver` pattern, generalized.
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
2. **Extract `SweepChain`** (L3) with only `ComposedChain` — pure deduplication, zero behavior
   change; kills the five inline chains (single-group mv/mvᵀ, both multigroup scans, adjoint scan,
   rhs helpers). Add the compatibility `@property` shims.
3. **`MultiGroupApply` / `GroupLifting`** (L4) — collapse the multigroup subclass hierarchy into
   `GroupsApply(ScanGroups(checkpoint=…))`; make `SingleGroupOperator` and `MultiGroupOperator`
   concrete final classes; rework the adjoint scan onto the chain; change the factory signature.
4. Later, driven by profiling: fused kernels at levels 1–4; opaque-kernel `fast_primal` machinery
   only when a non-differentiable fused kernel actually exists.

## Misc findings from the same review (independent of the tower)

- `MultiGroupOperator.from_multigroup_operator` passes fields positionally — silent misassignment
  risk if field order changes; prefer keyword construction after the refactor.
- The derived `transport_operator` property rebuilds via `transport_spec.build(setup)` per access —
  free under `filter_jit`, but looks expensive; worth a comment.
- `AdjointMultiGroupOperator._mv_transpose` raising `NotImplementedError` will surface deep inside a
  lineax solver that requests a transpose; the message should say what to use instead.
