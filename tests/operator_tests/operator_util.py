
import os

import jax
import numpy as onp
import jax.numpy as jnp
import pytest

import equinox as eqx

from math import comb
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Callable, Dict, Type


# ---------------------------------------------------------------------------
# Strategy registry + (dim, degree) matrix for the operator-property test classes.
#
# Each operator test class declares a ``{StrategyClass: Spec}`` registry. ``valid_combos`` crosses it with
# the (dim, degree) matrix but *drops* combinations a strategy does not support -- so an unsupported combo
# is never generated (no skip, no green: the test simply does not exist). The (dim, degree) axis is fed
# through the cached ``bb_mgop_dim_degree`` fixture by indirect parametrisation, so builds stay shared.
# ---------------------------------------------------------------------------
DIM_DEGREES = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2)]
'''The (dim, FEM degree) matrix the BB problem is exercised at (single source; conftest imports this).'''


def simplex_n_basis(dim: int, degree: int) -> int:
    '''Number of Lagrange nodes on a ``dim``-simplex of the given degree: C(dim + degree, degree).'''
    return comb(dim + degree, degree)

def linear_tet_only(dim: int, degree: int) -> bool:
    '''The warp kernels are hardcoded to the 3-D linear tetrahedron (n_basis == 4 *and* tet topology).'''
    return dim == 3 and degree == 1

@dataclass
class Spec:
    '''How to build + when to test one strategy variant.

    ``kwargs``   -- constructor kwargs for the strategy (e.g. a batch size).
    ``supports`` -- ``supports(dim, degree) -> bool``; a combo where this is False is not generated at all.
    ``marks``    -- pytest marks applied to every generated combo for this strategy (e.g.
                    ``pytest.mark.pallas`` so the Pallas variants skip unless ``--runpallas`` is given).
    '''
    kwargs: Dict = field(default_factory=dict)
    supports: Callable[[int, int], bool] = lambda dim, degree: True
    marks: tuple = ()


def valid_combos(strategies: Dict[Type, Spec], dim_degrees=DIM_DEGREES):
    '''``pytest.param`` list of ``((dim, degree), strategy_cls, kwargs)`` over the supported combos only.

    Consumed by :func:`parametrize_strategies`; the (dim, degree) drives the cached ``bb_mgop`` fixture.'''
    return [
        pytest.param((dim, degree), cls, spec.kwargs, id=f"dim{dim}-deg{degree}-{cls.__name__}", marks=spec.marks)
        for (dim, degree) in dim_degrees
        for cls, spec in strategies.items()
        if spec.supports(dim, degree)
    ]


def parametrize_strategies(strategies: Dict[Type, Spec], dim_degrees=DIM_DEGREES):
    '''Class/function decorator applying the standard strategy x (dim, degree) parametrisation.
    '''

    return pytest.mark.parametrize(
        "bb_mgop_dim_degree, strategy_cls, kwargs",
        valid_combos(strategies, dim_degrees),
        indirect=["bb_mgop_dim_degree"],
    )


def _rand(shape, seed):
    return jnp.array(onp.random.default_rng(seed).random(shape))


@lru_cache
def _grad_of_sum(apply):
    '''``grad_wrt_operator( sum(apply(op, x)) )``, jitted once per distinct ``apply`` (forward vs inverse)
    so the compile is shared across every parametrised variant. ``apply`` is a plain function (hashable),
    so the cache key is stable.'''
    return eqx.filter_jit(eqx.filter_grad(lambda op, x: jnp.sum(apply(op, x))))


def _inexact_leaves(tree):
    '''The inexact-array leaves of a pytree, discarding its structure.

    Two operators that differ only in their (non-differentiated) strategy field produce gradients whose
    *tree structure* differs -- the strategy node has a different type per variant (e.g. LaxMass vs
    VmapMass) and trips a structural pytree compare -- even though it carries no gradient. Comparing the
    inexact-array leaves directly sidesteps that: the empty strategy modules contribute no leaves, so the
    gradient arrays line up regardless of which strategy produced them.'''
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))


# ---------------------------------------------------------------------------
# Reference values: the pivot (default-strategy) operator's forward / transpose / grad / jvp, which every
# variant of an axis is compared against. Sourced per JAXSN_REFERENCE (or --reference-mode):
#   live   -- compute from the live pivot (classic same-run cross-strategy consistency)
#   auto   -- frozen bundle if present, else live (default)
#   frozen -- frozen only; error if missing (reference strategy need not exist; pins the pivot itself)
#   freeze -- compute live and (re)write the bundle
# Keyed by reference_id = (test-class name, dim, degree): strategy-independent, so the default strategy is
# checked against the same stored truth as every variant -- turning a pivot regression into a real failure.
# Memoised on the pytest session (see the ``reference_values`` fixture), not a module global.
# ---------------------------------------------------------------------------
_REF_CACHE = pytest.StashKey[dict]()


def reference_mode() -> str:
    return os.environ.get("JAXSN_REFERENCE", "auto")


def _reference_values(reference_id, build_reference, apply, grad_leaves) -> Dict:
    '''Load (auto/frozen) or compute (live/freeze) the {forward, transpose, grad, jvp} for one reference_id.
    ``build_reference`` (zero-arg thunk -> pivot operator) is invoked only when a live value is needed. The
    grad is a variable-shape list of inexact leaves, flattened into ``grad_i`` in the flat npz namespace.'''
    from reference_data import save, load
    (cls, (dim, degree)) = reference_id
    name, mode = f"opref__{cls}__d{dim}_deg{degree}", reference_mode()

    if mode in ("auto", "frozen"):
        try:
            b = load(name)
        except FileNotFoundError:
            if mode == "frozen":
                raise RuntimeError(f"reference mode 'frozen' but no bundle '{name}'; freeze it first")
        else:
            grad = [b[k] for k in sorted((k for k in b if k.startswith("grad_") and k[5:].isdigit()), key=lambda k: int(k[5:]))]
            return {"forward": b["forward"], "transpose": b["transpose"], "jvp": b["jvp"], "grad": grad}

    ref = build_reference()
    si, so = ref.in_structure().shape, ref.out_structure().shape
    _, jvp = jax.jvp(lambda v: apply(ref, v), (_rand(si, 6),), (_rand(si, 7),))
    vals = {"forward":   onp.asarray(apply(ref, _rand(si, 3))),
            "transpose": onp.asarray(apply(ref.transpose(), _rand(so, 4))),
            "jvp":       onp.asarray(jvp),
            "grad":      [onp.asarray(g) for g in grad_leaves(_grad_of_sum(apply)(ref, _rand(si, 5)))]}
    if mode == "freeze":
        save(name, forward=vals["forward"], transpose=vals["transpose"], jvp=vals["jvp"],
             **{f"grad_{i}": g for i, g in enumerate(vals["grad"])})
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
    4. Gradient / jvp consistency with the reference

    Subclass provides two fixtures:
      * ``operator``         -- the variant under test.
      * ``build_reference``  -- returning the reference operator.
    Override :meth:`apply` to test ``op.inv`` instead of ``op.mv`` (the transport-inversion classes do).
    '''
    rtol_exact, atol_exact = 1e-12, 1e-12
    rtol, atol = 1e-12, 1e-12
    rtol_grad, atol_grad = 1e-12, 1e-12

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
            cache[reference_id] = _reference_values(reference_id, build_reference, self.apply, self._grad_compare_leaves)
        return cache[reference_id]

    def _grad_compare_leaves(self, grad_tree):
        '''Which gradient leaves are compared/frozen. Default: every inexact-array leaf (the empty strategy
        modules contribute none, so op and reference line up regardless of variant). Override where op and
        reference carry structurally different auxiliary data -- e.g. a level-parallel vs a basic
        ``sweep_data`` -- so only the shared (solution-domain) gradient, which all variants agree on, is
        compared.'''
        return _inexact_leaves(grad_tree)

    def test_adjoint_identity(self, operator):
        x = _rand(operator.in_structure().shape,  0)
        y = _rand(operator.out_structure().shape, 1)
        
        onp.testing.assert_allclose(onp.sum(self.apply(operator, x) * y), onp.sum(x * self.apply(operator.transpose(), y)), rtol=self.rtol_exact, atol=self.atol_exact)

    def test_transpose_idempotence(self, operator):
        x = _rand(operator.in_structure().shape, 2)
        
        onp.testing.assert_allclose(self.apply(operator.transpose().transpose(), x), self.apply(operator, x), rtol=self.rtol_exact, atol=self.atol_exact)

    def test_forward_consistency(self, operator, reference_values):
        x = _rand(operator.in_structure().shape, 3)
        
        onp.testing.assert_allclose(self.apply(operator, x), reference_values["forward"], rtol=self.rtol, atol=self.atol)

    def test_transpose_consistency(self, operator, reference_values):
        y = _rand(operator.out_structure().shape, 4)
        
        onp.testing.assert_allclose(self.apply(operator.transpose(), y), reference_values["transpose"], rtol=self.rtol, atol=self.atol)

    def test_grad_wrt_input_consistency(self, operator, reference_values):
        x = _rand(operator.in_structure().shape, 5)
        
        g_op  = self._grad_compare_leaves(_grad_of_sum(self.apply)(operator, x))
        g_ref = reference_values["grad"]
        assert len(g_op) == len(g_ref)
        for a, b in zip(g_op, g_ref):
            onp.testing.assert_allclose(a, b, rtol=self.rtol_grad, atol=self.atol_grad)

    def test_jvp_wrt_input_consistency(self, operator, reference_values):
        
        x = _rand(operator.in_structure().shape, 6)
        t = _rand(operator.in_structure().shape, 7)
        _, jv = jax.jvp(lambda v: self.apply(operator, v), (x,), (t,))
        onp.testing.assert_allclose(jv, reference_values["jvp"], rtol=self.rtol, atol=self.atol)

