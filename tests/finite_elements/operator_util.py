
import jax
import numpy as onp
import jax.numpy as jnp
import pytest

import equinox as eqx

from math import comb
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
    '''
    kwargs: Dict = field(default_factory=dict)
    supports: Callable[[int, int], bool] = lambda dim, degree: True


def valid_combos(strategies: Dict[Type, Spec], dim_degrees=DIM_DEGREES):
    '''``pytest.param`` list of ``((dim, degree), strategy_cls, kwargs)`` over the supported combos only.

    Use with ``@pytest.mark.parametrize("bb_mgop_dim_degree, strategy_cls, kwargs", ...,
    indirect=["bb_mgop_dim_degree"])`` so the (dim, degree) drives the cached ``bb_mgop`` fixture.'''
    return [
        pytest.param((dim, degree), cls, spec.kwargs, id=f"dim{dim}-deg{degree}-{cls.__name__}")
        for (dim, degree) in dim_degrees
        for cls, spec in strategies.items()
        if spec.supports(dim, degree)
    ]


def _rand(shape, seed):
    return jnp.array(onp.random.default_rng(seed).random(shape))


def _sum_of_operator(operator, x):
    return jnp.sum(operator.mv(x))

_grad_of_sum = eqx.filter_jit(eqx.filter_grad(_sum_of_operator))


def _inexact_leaves(tree):
    '''The inexact-array leaves of a pytree, discarding its structure.

    Two operators that differ only in their (non-differentiated) strategy field produce gradients whose
    *tree structure* differs -- the strategy node has a different type per variant (e.g. LaxMass vs
    VmapMass) and trips a structural pytree compare -- even though it carries no gradient. Comparing the
    inexact-array leaves directly sidesteps that: the empty strategy modules contribute no leaves, so the
    gradient arrays line up regardless of which strategy produced them.'''
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))


class BaseOperatorProperties:
    '''
    Defines the base operator properties:

    1. Adjoint identity: <Ax, y> = <x, A^Ty>
    2. Idempotence of transpose: A^T^T = A
    3. Consistency of forward and transpose with the reference operator
    4. Consistency of the gradient and jvp with the reference operator

    Implement fixture for the base operator and the reference operator to specialize to a particular class.
    '''
    @pytest.fixture
    def operator_and_reference(self):
            raise NotImplementedError

    def test_adjoint_identity(self, operator_and_reference):
        operator, _ = operator_and_reference

        x = _rand(operator.in_structure().shape,  0)
        y = _rand(operator.out_structure().shape, 1)
        onp.testing.assert_allclose(onp.sum(operator.mv(x) * y), onp.sum(x * operator.transpose().mv(y)), rtol=1e-12, atol=1e-12)

    def test_transpose_idempotence(self, operator_and_reference):
        operator, _ = operator_and_reference
        x = _rand(operator.in_structure().shape, 2)
        onp.testing.assert_allclose( operator.transpose().transpose().mv(x), operator.mv(x), rtol=1e-12, atol=1e-12)

    def test_forward_consistency(self, operator_and_reference):
        operator, reference = operator_and_reference
        x = _rand(operator.in_structure().shape, 3)
        onp.testing.assert_allclose(operator.mv(x), reference.mv(x), rtol=1e-11, atol=1e-11)

    def test_transpose_consistency(self, operator_and_reference):
        operator, reference = operator_and_reference
        y = _rand(operator.out_structure().shape, 4)
        onp.testing.assert_allclose(operator.transpose().mv(y), reference.transpose().mv(y), rtol=1e-11, atol=1e-11)

    def test_grad_wrt_input_consistency(self, operator_and_reference):
        operator, reference = operator_and_reference
        x = _rand(operator.in_structure().shape, 5)
        g_op  = _inexact_leaves(_grad_of_sum(operator, x))
        g_ref = _inexact_leaves(_grad_of_sum(reference, x))
        assert len(g_op) == len(g_ref)
        for a, b in zip(g_op, g_ref):
            onp.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)

    def test_jvp_wrt_input_consistency(self, operator_and_reference):
        operator, reference = operator_and_reference
        x = _rand(operator.in_structure().shape, 6)
        t = _rand(operator.in_structure().shape, 7)
        _, jv = jax.jvp(lambda v: operator.mv(v), (x,), (t,))
        _, jv_ref = jax.jvp(lambda v: reference.mv(v), (x,), (t,))
        onp.testing.assert_allclose(jv, jv_ref, rtol=1e-11, atol=1e-11)
