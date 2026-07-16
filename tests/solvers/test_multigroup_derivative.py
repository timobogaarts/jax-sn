'''Finite-difference check of end-to-end differentiation through the block solve.

The multigroup fixed-source solve -- RHS conditioning + MultiGroupLowerTriangularBlockSolver -- is
differentiated w.r.t. the cross sections with reverse-mode AD (lineax differentiates the solve
implicitly through the operator's mv, so the block substitution itself need not be differentiable), and
the resulting gradient is checked entry-by-entry against a one-sided finite difference that re-solves
from the perturbed cross sections. Runs at a single (dim, degree) to bound cost.
'''
import jax
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import lineax as lx
import pytest

from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver
from jax_sn.solvers.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs
from jax_sn.parametrisation.multigroup_parametrisation import MultigroupOperatorParametrisedXS

from bb_setup import downscattering_only as _downscattering_only

_DIM_DEGREE = (1, 2)   # cheapest combo that still has a non-trivial (degree>1) FEM basis


@pytest.mark.slow
def test_mg_block_solve_ad(bb_mgop):
    '''grad(sum(solve)) w.r.t. the cross sections matches a one-sided finite difference, for a few
    total-xs and downscatter entries, when solving the fixed source with the block solver.'''
    mgop, source_total, _, bb_args = bb_mgop
    if (bb_args[0], bb_args[1]) != _DIM_DEGREE:
        pytest.skip(f"derivative check runs only at dim,degree={_DIM_DEGREE} to bound cost")

    mgop   = _downscattering_only(mgop)
    solver = MultiGroupLowerTriangularBlockSolver.from_composed(mgop.strategy, lx.BiCGStab(1e-13, 1e-13, max_steps=200))
    rhs    = IsotropicSourceSingleGroup.from_operator(mgop, source_total[0] * onp.sqrt(4.0 * onp.pi), 0)

    def solve_sum(theta):
        op = MultigroupOperatorParametrisedXS.reconstruct_operator(mgop, theta)
        val = solve_multigroup_operator_rhs(op, rhs, solver, op.strategy.single_group_chain_strategy).value
        return jnp.sum(val)

    theta0 = MultigroupOperatorParametrisedXS.theta_base(mgop)
    grad   = eqx.filter_jit(jax.grad(solve_sum))(theta0)
    base   = solve_sum(theta0)

    dx = 1e-4
    # (leaf, index) pairs: total-xs entries and one lower-triangular (downscatter) scattering entry.
    cases = [
        (lambda t: t.total,      (0, 0)),
        (lambda t: t.total,      (3, 3)),
        (lambda t: t.total,      (5, 1)),
        (lambda t: t.scattering, (3, 0, 1, 0)),   # material 3, l=0, group 0 -> group 1 (downscatter)
    ]
    for leaf, idx in cases:
        theta_p = eqx.tree_at(leaf, theta0, replace_fn=lambda a: a.at[idx].add(dx))
        fd = (solve_sum(theta_p) - base) / dx
        ad = leaf(grad)[idx]
        onp.testing.assert_allclose(onp.array(fd), onp.array(ad), rtol=1e-3, atol=1e-3)
