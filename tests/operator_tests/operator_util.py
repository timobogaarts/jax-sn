'''
Operator utilities.

Each operator has a set of properties it should *always* satisfy:

1. Adjoint identity  <Ax, y> = <x, A^Ty>            
2. Transpose idempotence  A^T^T = A                 

Furthermore, it should be consistent for all implementations, including jvp & vjp.

These are all tested using the framework in this file: each operator subclasses the BaseOperatorProperties
and provides a way to construct the operator (for a number of different implementations/strategies) and a way 
to construct a reference operator.
'''
import os

import jax
import numpy as onp
import jax.numpy as jnp
import pytest

import equinox as eqx

from math import comb
from dataclasses import dataclass, field
from typing import Callable, Dict, Type
from conftest import DIM_DEGREES


MAX_WARP_LENGTH = 4
'''Warp's largest fixed-size vector/matrix type; bounds both n_basis and n_faces.'''


def simplex_n_basis(dim: int, degree: int) -> int:
    '''Number of Lagrange nodes on a ``dim``-simplex of the given degree: C(dim + degree, degree).'''
    return comb(dim + degree, degree)

def warp_supported(dim: int, degree: int) -> bool:
    '''Whether the Warp kernels cover this element. They are parametrised on ``(n_basis, n_faces)``
    but Warp's fixed-size vector/matrix types stop at 4, so both must fit: that admits the linear,
    quadratic and cubic interval, the linear triangle and the linear tetrahedron -- and rules out
    every element of degree >= 2 in 2-D/3-D.'''
    return simplex_n_basis(dim, degree) <= MAX_WARP_LENGTH and (dim + 1) <= MAX_WARP_LENGTH

@dataclass
class Spec:
    '''How to build + when to test one strategy variant.    
    '''
    kwargs: Dict = field(default_factory=dict)
    '''Constructor kwargs for the strategy (e.g. a batch size).'''

    supports: Callable[[int, int], bool] = lambda dim, degree: True
    '''Predicate for whether this strategy supports the given (dim, degree).'''

    marks: tuple = ()
    '''pytest marks to apply to the strategy's tests (e.g. skip/xfail/pallas)'''

def valid_combos(strategies: Dict[Type, Spec], dim_degrees=DIM_DEGREES):
    '''((dim, degree), strategy_cls, kwargs) over the supported combos only.'''
    return [
        pytest.param((dim, degree), cls, spec.kwargs, id=f"dim{dim}-deg{degree}-{cls.__name__}", marks=spec.marks)
        for (dim, degree) in dim_degrees
        for cls, spec in strategies.items()
        if spec.supports(dim, degree)
    ]


def parametrize_strategies(strategies: Dict[Type, Spec], dim_degrees=DIM_DEGREES):
    '''Class/function decorator applying the standard strategy x (dim, degree) parametrisation.'''

    return pytest.mark.parametrize(
        "bb_mgop_dim_degree, strategy_cls, kwargs",
        valid_combos(strategies, dim_degrees),
        indirect=["bb_mgop_dim_degree"],
    )


def instance_label(instance) -> str:
    '''Readable, collision-free test id for a strategy *instance*: ``VmapAnglesInversion(VmapSweep())`` ->
    ``VmapAnglesInversion-VmapSweep``, a native ``WarpSequentialAngles()`` -> ``WarpSequentialAngles``. Composed
    strategies carry the wrapped kernel in ``.inner``; native ones have no inner so the class name alone
    is unambiguous.'''
    name = type(instance).__name__
    inner = getattr(instance, "inner", None)
    return f"{name}-{type(inner).__name__}" if inner is not None else name


def valid_instance_combos(strategies: Dict[object, Spec], dim_degrees=DIM_DEGREES):
    '''((dim, degree), instance) over supported combos. Like :func:`valid_combos` but keyed on pre-built
    strategy *instances* -- for a strategy axis whose variants are configured objects (e.g. the multi-angle
    inversion, where a variant is a lift wrapping an inner kernel) rather than a class + ctor kwargs.'''
    return [
        pytest.param((dim, degree), inst, id=f"dim{dim}-deg{degree}-{instance_label(inst)}", marks=spec.marks)
        for (dim, degree) in dim_degrees
        for inst, spec in strategies.items()
        if spec.supports(dim, degree)
    ]


def parametrize_instances(strategies: Dict[object, Spec], dim_degrees=DIM_DEGREES):
    '''Class/function decorator parametrising over strategy *instances* x (dim, degree) (see
    :func:`valid_instance_combos`). The test receives the instance as the ``strategy`` fixture arg.'''

    return pytest.mark.parametrize(
        "bb_mgop_dim_degree, strategy",
        valid_instance_combos(strategies, dim_degrees),
        indirect=["bb_mgop_dim_degree"],
    )


def _rand(shape, seed):
    '''Random array of the given shape, seeded for reproducibility.'''
    return jnp.array(onp.random.default_rng(seed).random(shape))

def _inexact_leaves(tree):
    '''The inexact-array leaves of a pytree, discarding its structure.
    Exact leaves don't contribute differentiability but prohibit direct pytree comparison so they are filtered out here.
    '''
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))


def _rand_tangent(tree, base_seed):
    '''Random tangent for a pytree. Exact leaves (containing solution setup which cannot be differentiated anyway) are dropped,
    such that operators with different solution setup still can be compared directly.'''
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    tangents, j = [], 0
    for a in leaves:
        if eqx.is_inexact_array(a):
            tangents.append(_rand(a.shape, base_seed + j))
            j += 1
        else:
            tangents.append(None)
    return jax.tree_util.tree_unflatten(treedef, tangents)


def _operator_jvp(operator, apply):    
    x = _rand(operator.in_structure().shape, 6)
    tangent = _rand_tangent(operator, base_seed=8)
    _, jv = eqx.filter_jvp(lambda op: apply(op, x), (operator,), (tangent,))
    return jv


def _operator_vjp(operator, apply):
    '''VJP of the operator. Only inexact leaves are returned to compare different operators with different solution setup (exact leafs)'''
    x = _rand(operator.in_structure().shape, 5)
    out, vjp_fn = eqx.filter_vjp(lambda op: apply(op, x), operator)
    (cot,) = vjp_fn(_rand(out.shape, 9))
    return _inexact_leaves(cot)

# ---------------------------------------------------------------------------
# Reference values: the pivot (default-strategy) operator's forward / transpose / vjp / jvp, which every
# variant of an axis is compared against. Sourced per JAXSN_REFERENCE (or --reference-mode):
#   frozen -- frozen bundle only; error if missing (DEFAULT).
#   live   -- compute from the live pivot (same-run cross-strategy consistency; no stored data needed)
#   freeze -- compute live and (re)write the bundle 
# ---------------------------------------------------------------------------
_REF_CACHE = pytest.StashKey[dict]()


def reference_mode() -> str:
    return os.environ.get("JAXSN_REFERENCE", "frozen")


def _reference_values(reference_id, build_reference, apply) -> Dict:
    '''Load (frozen) or compute (live/freeze) the {forward, transpose, vjp, jvp} for one reference_id.
    '''
    from reference_data import save, load
    (cls, (dim, degree)) = reference_id
    name, mode = f"opref__{cls}__d{dim}_deg{degree}", reference_mode()

    if mode == "frozen":
        try:
            b = load(name)
        except FileNotFoundError:
            raise RuntimeError(f"reference mode 'frozen' but no bundle '{name}'; freeze it first (--reference-mode freeze)")
        vjp = [b[k] for k in sorted((k for k in b if k.startswith("vjp_") and k[4:].isdigit()), key=lambda k: int(k[4:]))]
        return {"forward": b["forward"], "transpose": b["transpose"], "jvp": b["jvp"], "vjp": vjp}

    ref = build_reference()
    si, so = ref.in_structure().shape, ref.out_structure().shape
    vals = {"forward":   onp.asarray(apply(ref, _rand(si, 3))),
            "transpose": onp.asarray(apply(ref.transpose(), _rand(so, 4))),
            "jvp":       onp.asarray(_operator_jvp(ref, apply)),
            "vjp":       [onp.asarray(g) for g in _operator_vjp(ref, apply)]}
    if mode == "freeze":
        save(name, forward=vals["forward"], transpose=vals["transpose"], jvp=vals["jvp"],
             **{f"vjp_{i}": g for i, g in enumerate(vals["vjp"])})
    return vals


# ---------------------------------------------------------------------------
# Operator properties compared agains the default strategy (reference values)
# ---------------------------------------------------------------------------

class BaseOperatorProperties:
    '''
    Property + consistency battery for one operator axis:

    1. Adjoint identity  <Ax, y> = <x, A^Ty>            -- self-contained (operator only)
    2. Transpose idempotence  A^T^T = A                 -- self-contained (operator only)
    3. Forward / transpose consistency with the reference
    4. VJP / JVP (w.r.t. operator) consistency with the reference

    Subclass provides two fixtures:
      * ``operator``         -- the variant under test.
      * ``build_reference``  -- returning the reference operator.
    Override :meth:`apply` to test ``op.inv`` instead of ``op.mv`` 
    '''
    rtol_exact, atol_exact = 1e-12, 1e-12
    rtol, atol = 1e-12, 1e-12
    rtol_grad, atol_grad = 1e-12, 1e-12
    cpu_tol_floor = 1e-9
    '''Lower bound on every tolerance when running on CPU. XLA's multithreaded CPU reductions sum in an
    order that depends on thread scheduling, so results move by ~1e-11 between runs (and between the
    pivot and a variant) -- noise that 1e-12 cannot absorb. GPU runs keep the tighter class tolerances.'''

    def _tol(self, rtol, atol):
        if jax.default_backend() == "cpu":
            rtol, atol = max(rtol, self.cpu_tol_floor), max(atol, self.cpu_tol_floor)
        return dict(rtol=rtol, atol=atol)

    def apply(self, operator, x):
        '''The operator action tested by the property battery. Override to test ``op.inv`` instead of ``op.mv``.
        [This requires the application to have the same in/out structure as the mv, which is true for inverses.]'''
        return operator.mv(x)

    @pytest.fixture
    def operator(self):
        '''The variant operator under test.'''
        raise NotImplementedError

    @pytest.fixture
    def build_reference(self):
        '''Zero-arg thunk returning the pivot (default-strategy) operator. Called only when a live
        reference is needed (``live``/``freeze`` mode, or ``auto`` with no frozen bundle).'''
        raise NotImplementedError

    @pytest.fixture
    def reference_id(self, bb_mgop_dim_degree):
        '''Cache/bundle key for the reference values: ``(test class, dim, degree)``. Strategy-independent,
        so every variant of this axis shares one reference (and one frozen bundle).'''
        return (type(self).__name__, tuple(bb_mgop_dim_degree))

    @pytest.fixture
    def reference_values(self, request, reference_id, build_reference):
        '''Reference values for this axis+size, memoised on the pytest session so the pivot is built / the
        bundle loaded once -- not per variant or per property method. (reference_id fully determines apply
        and build_reference, so keying on it alone is safe.)'''
        cache = request.session.stash.setdefault(_REF_CACHE, {})
        if reference_id not in cache:
            cache[reference_id] = _reference_values(reference_id, build_reference, self.apply)
        return cache[reference_id]

    def test_adjoint_identity(self, operator):
        x = _rand(operator.in_structure().shape,  0)
        y = _rand(operator.out_structure().shape, 1)
        
        onp.testing.assert_allclose(onp.sum(self.apply(operator, x) * y), onp.sum(x * self.apply(operator.transpose(), y)), **self._tol(self.rtol_exact, self.atol_exact))

    def test_transpose_idempotence(self, operator):
        x = _rand(operator.in_structure().shape, 2)
        
        onp.testing.assert_allclose(self.apply(operator.transpose().transpose(), x), self.apply(operator, x), **self._tol(self.rtol_exact, self.atol_exact))

    def test_forward_consistency(self, operator, reference_values):
        x = _rand(operator.in_structure().shape, 3)
        
        onp.testing.assert_allclose(self.apply(operator, x), reference_values["forward"], **self._tol(self.rtol, self.atol))

    def test_transpose_consistency(self, operator, reference_values):
        y = _rand(operator.out_structure().shape, 4)
        
        onp.testing.assert_allclose(self.apply(operator.transpose(), y), reference_values["transpose"], **self._tol(self.rtol, self.atol))

    def test_vjp_wrt_operator_consistency(self, operator, reference_values):        
        g_op  = _operator_vjp(operator, self.apply)
        g_ref = reference_values["vjp"]
        assert len(g_op) == len(g_ref)
        for a, b in zip(g_op, g_ref):
            onp.testing.assert_allclose(a, b, **self._tol(self.rtol_grad, self.atol_grad))

    def test_jvp_wrt_operator_consistency(self, operator, reference_values):        
        jv = _operator_jvp(operator, self.apply)
        onp.testing.assert_allclose(jv, reference_values["jvp"], **self._tol(self.rtol_grad, self.atol_grad))

