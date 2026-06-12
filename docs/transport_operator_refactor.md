# Transport-operator refactor: strategy composition

## Goal

Replace the parallel subclass hierarchies of `TransportOperator` (Vmap / Batched / Pallas / Warp)
and `TransportOperatorMultipleAngles` (Vmap / SequentialWarp / LevelsWarp) with **two final
concrete operator classes parameterized by strategy objects**. This:

1. Makes a new inversion method a single small class (`InversionStrategy.__call__`), not a new
   operator subclass.
2. Lets every forward strategy compose with every inversion strategy for free (independent fields,
   `F + I` classes instead of `F × I` operators).
3. Makes the multi-angle lift pluggable (vmap / scan / decomposition) and removes vmap knowledge
   from the operator base — it lives in `AngleLifting` strategies.

It preserves the categorical split: `TransportOperator` is a lineax operator on `(n_elements,
n_basis)`; `TransportOperatorMultipleAngles` is a *distinct* lineax operator on `(n_angles,
n_elements, n_basis)`. The lift uses the single-angle op strictly as a black-box F-morphism — no
multi-angle methods are bolted onto the single-angle class.

## Design

### Operators (final, concrete)

```python
class TransportOperator(AbstractLinearOperator_sn):
    solution_domain_data:     SolutionDomainData
    topological_sorted_angle: TopologicalSortedAngleBase
    energy_group:             int
    forward_strategy:         ForwardStrategy
    inversion_strategy:       InversionStrategy

    def mv(self, x):              return self.forward_strategy(self, x)
    def _invert_internal(self, b): return self.inversion_strategy(self, b)
    # transpose / inv / in_structure / out_structure unchanged
    def _forward_data(self): return (*sdd, xs_full(energy_group), toposort.angle)
    def _sweep_data(self):   return (*sdd, xs_full(energy_group), toposort)

class TransportOperatorMultipleAngles(AbstractLinearOperator_sn):
    transport_operator: TransportOperator
    angle_lifting:      AngleLifting

    def mv(self, x):              return self.angle_lifting.forward(self.transport_operator, x)
    def _invert_internal(self, b): return self.angle_lifting.invert(self.transport_operator, b)
    def _sweep_data(self):        return self.transport_operator._sweep_data()  # for warp flatten
    # transpose / properties / __getitem__ unchanged
```

`_forward_data()` / `_sweep_data()` replace `_base_forward_args` / `_base_inv_args` (minus the
trailing `x`/`b`, which the strategy appends).

### Strategies (`transport_operator_strategies.py`, new)

```python
class ForwardStrategy(eqx.Module):
    @abstractmethod def __call__(self, op, x): ...
class VmapForward(ForwardStrategy):
    def __call__(self, op, x): return forward_transport_operator_vmap(*op._forward_data(), x)
class BatchedForward(ForwardStrategy):
    batch_size: int = None
    def __call__(self, op, x): return forward_transport_operator_batch(*op._forward_data(), x, self.batch_size)

class InversionStrategy(eqx.Module):
    @abstractmethod def __call__(self, op, b): ...
class VmapSweep(InversionStrategy):
    def __call__(self, op, b): return transport_inverse_sweep_complete_vmap(*op._sweep_data(), b)
class BatchedSweep(InversionStrategy):
    batch_size: int = None
    def __call__(self, op, b): return transport_inverse_sweep_batched(*op._sweep_data(), b, self.batch_size)

class AngleLifting(eqx.Module):
    @abstractmethod def forward(self, single_op, x): ...
    @abstractmethod def invert(self, single_op, b): ...
class VmapAngles(AngleLifting):
    # The angle axis lives in the toposort: map a single-angle reduction over its leading axis.
    @staticmethod
    def _map(single_op, leading, fn):
        def one(toposort_i, v_i):
            op_i = eqx.tree_at(lambda o: o.topological_sorted_angle, single_op, toposort_i)
            return fn(op_i, v_i)
        return eqx.filter_vmap(one, in_axes=(0, 0))(single_op.topological_sorted_angle, leading)
    def forward(self, single_op, x): return self._map(single_op, x, lambda o, v: o.mv(v))
    def invert(self, single_op, b):  return self._map(single_op, b, lambda o, v: o._invert_internal(v))
```

### Pallas strategies (`transport_operator_pallas.py`)
`VmapPallasSweep`, `BatchedPallasSweep` (+ factory funcs `TransportOperatorVmapPallas`,
`TransportOperatorBatchedPallas`). Stay in the pallas file (it already imports base).

### Warp strategies (`transport_operator_warp.py`)
`WarpSequentialSweep`, `WarpLevelSweep` (inversion); `WarpSequentialAngles`, `WarpLevelAngles`
(lift; `forward` delegates to `VmapAngles`, `invert` calls the fused kernel). The warp-shape
asserts move into a `_assert_warp_compatible(sdd)` used by the warp factories. The
`TransportOperatorLevelsWarpSingleAngle` factory builds the `TopologicalSortedAngleLevelParallel`.

### Backward-compatible factory functions
The old class names become module-level factory functions with the *same call signatures*, so all
call sites and `isinstance(x, TransportOperator[MultipleAngles])` checks keep working (they return
instances of the two concrete classes). Examples:

```python
def TransportOperatorVmap(sdd, toposort, eg):
    return TransportOperator(sdd, toposort, eg, VmapForward(), VmapSweep())
def TransportOperatorBatched(sdd, toposort, eg, batch_size=None):
    return TransportOperator(sdd, toposort, eg, BatchedForward(batch_size), BatchedSweep(batch_size))
def TransportOperatorMultipleAnglesVmap(top):           return TransportOperatorMultipleAngles(top, VmapAngles())
def TransportOperatorLevelsWarpSingleAngle(sdd, toposort, eg, batch_size, max_n_levels=None): ...
def TransportOperatorLevelsWarpMultipleAngles(top):     return TransportOperatorMultipleAngles(top, WarpLevelAngles())
```

This dissolves two earlier smells for free: `batch_size` now lives on the strategy (not the
operator base), and warp ops are no longer `TransportOperatorBatched` subclasses.

## Touch list
1. `transport_operator_strategies.py` (new): ABCs + Vmap/Batched forward, inversion, VmapAngles.
2. `transport_operator_base.py`: rewrite `TransportOperator` (strategy fields, `_forward_data`/
   `_sweep_data`, drop `_*_op_args`/`_*_args`/`_vmap_for_*`); make `TransportOperatorMultipleAngles`
   final concrete (drop abstract `mv`/`_invert_internal`, add `angle_lifting`, add `_sweep_data`);
   delete the Vmap/Batched/Pallas-subclass + `TransportOperatorMultipleAnglesVmap` + the
   `vmap_multiple_angles_*` free functions; add Vmap/Batched factory funcs.
3. `transport_operator_pallas.py`: add pallas strategies + pallas factory funcs.
4. `transport_operator_warp.py`: warp strategies + warp factory funcs; drop the old classes and the
   `__check_init__` pairing asserts.
5. `linear_tetrahedron_warp_kernel.py:290`: `_base_inv_args(None)[:-1]` → `_sweep_data()`.
6. `transport_operator/__init__.py`: export the two operator classes, the strategy classes, and the
   factory funcs (old names).
7. `multigroup_parametrisation.py`: replace the `isinstance(..., TransportOperatorLevelsWarpMultipleAngles)`
   warp check with `isinstance(toposort, TopologicalSortedAngleLevelParallel)` (drops the
   `_warp_available`/warp-module dependency entirely).
8. `multi_group_operator_creation.py`: defaults already name `TransportOperatorVmap` /
   `TransportOperatorMultipleAnglesVmap` (now factory funcs) — should need no change.

## Risks / validation order
1. **Risky core first**: `VmapAngles` — closing over `single_op`, swapping toposort inside
   `eqx.filter_vmap(in_axes=(0,0))`. Validate vmap + jit + transpose + `__getitem__` parity against
   the current `TransportOperatorMultipleAnglesVmap` on CPU (stellsim) before touching warp/pallas.
   Watch for the closed-over batched toposort being baked in as a constant (rely on DCE; if it
   bites, strip the toposort from the closure).
2. **Static fields**: `batch_size` and strategy objects must partition as static under
   `filter_jit`/`filter_vmap` (plain `int` / no-array Modules → static by default; verify).
3. Verify stellsim (no warp): AD/reconstruction + transport tests. Verify blanketopt (warp):
   `test_warp_linear_tetrahedron.py` (level sweep, transpose, index, grad parity).

## Strategy preconditions: demand, don't prepare (final shape)

Most strategies need no setup and consume any toposort; the level-based warp sweep is the exception —
it requires a `TopologicalSortedAngleLevelParallel`. Building that (possibly expensive) level setup is
made an *explicit, visible* step rather than hidden work inside operator construction:

- **The toposort owns its (re)construction.** `TopologicalSortedAngleBase.reconstruct(new_domain)`
  recomputes the sort for a deformed domain, reusing the toposort's own stored `angle` and preserving
  its concrete type (a level-parallel toposort rebuilds its level setup with the same padding, read
  back from `level_fwd.n_levels`). Abstract on the base + a final `_rebuilt_sort` helper, matching the
  `reverse`/`__getitem__` pattern. `multigroup_parametrisation` just calls
  `existing_toposort.reconstruct(domain_replaced)` — no isinstance, no manual level-rebuild, no
  `n_levels` lookup, no quadrature_set threading.
- **The operator demands the right toposort; creation is "free".** `InversionStrategy.check_compatible(operator)`
  (default no-op) declares the precondition; `WarpLevelSweep` asserts the toposort is level-parallel +
  warp-compatible sdd, `WarpSequentialSweep` asserts warp-compatible sdd, and `AngleLifting.check_compatible(single_op)`
  asserts the warp lifts' single-angle partner carries the matching warp inversion strategy. These run
  in `TransportOperator.__check_init__` / `TransportOperatorMultipleAngles.__check_init__`, so they fire
  at construction and on every `eqx.tree_at` (transpose, indexing, reconstruction). Equinox runs
  `__check_init__` on `tree_at`; the checks are type/shape only, so they are trace-safe.

The caller builds the required toposort explicitly (e.g. `TopologicalSortedAngleLevelParallel.from_topological_sorted_angle(plain, max_n_levels)`)
before constructing a warp-levels operator; the factory just validates it. There is no
`prepare_toposort` and no `TransportOperator.create` — construction does no hidden setup.

## Strategy-native creation + `required_toposort_type` (the helper-buildability fix)

"Demand, don't prepare" left a gap: a generic helper can't know which toposort to build, because the
operator doesn't exist yet to be asked. The answer is a property of the *strategy class*, not instance
data — so it's a classmethod:

- `InversionStrategy.required_toposort_type() -> Type[TopologicalSortedAngleBase]` — the concrete
  toposort type a creation helper should build (default `TopologicalSortedAngle`; `WarpLevelSweep` →
  `TopologicalSortedAngleLevelParallel`). This is the *build hint* and is distinct from
  `check_compatible` (the permissive, instance/data-dependent *acceptance* check — the plain sweeps
  accept any toposort, so it can't be an isinstance of the build type).
- `TopologicalSortedAngleLevelParallel.from_domain_quadrature_set(domain, quad, maximum_number_of_levels=None)`
  added so both toposort types share the builder interface the helper calls polymorphically (levels
  computed once from the domain).

`create_multi_group_operator` is now **strategy-native**: it takes `forward_strategy` /
`inversion_strategy` / `angle_lifting` (defaults `VmapForward`/`VmapSweep`/`VmapAngles`, reproducing the
old default) plus `topological_sorted_angle_kwargs`, and builds the toposort as
`inversion_strategy.required_toposort_type().from_domain_quadrature_set(domain, quad, **kwargs)`. So a
warp-levels mgop is now buildable generically:

```python
create_multi_group_operator(sd, quad,
    forward_strategy=BatchedForward(100), inversion_strategy=WarpLevelSweep(), angle_lifting=WarpLevelAngles(),
    topological_sorted_angle_kwargs={"maximum_number_of_levels": 400})
```

`max_n_levels` stays a *toposort-build* concern (the `topological_sorted_angle_kwargs`), not a strategy
field — keeping the strategy a pure consumer. No caller passed the old `_cls`/`_kwargs` transport
params, so the signature change needed no call-site updates. Construction is still demand-only; the
build is the explicit `required_toposort_type().from_domain_quadrature_set(...)` step in the helper.

Footgun (explicit, loud): `topological_sorted_angle_kwargs` must match the chosen type (passing
`maximum_number_of_levels` with a plain-toposort strategy → `TypeError`).

## Deferred / out of scope
- Strategy-native `create_multi_group_operator` API (keep the `_cls`/`_kwargs` API via factories).
- A `scan`-based `AngleLifting` (the design supports it; don't build speculatively).
- Structural enforcement of warp single/multi strategy pairing (compatibility is by construction;
  the old `__check_init__` assert is simply dropped).
