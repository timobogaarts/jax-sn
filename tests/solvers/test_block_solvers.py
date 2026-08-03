# ==================================================================================================
# Block-solver equivalence: MultiGroupLowerTriangularBlockSolver vs the generic global BiCGStab solve.
# ==================================================================================================
from bb_setup import downscattering_only as _downscattering_only
import equinox as eqx
import pytest
import jax.numpy as jnp
import numpy as onp
import lineax as lx

from jax_sn.operators.multi_group_operator import MultiGroupChainGatherScanStrategy, MultiGroupChainGatherVmapStrategy, ComposedMultiGroupChainStrategy
from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver

from conftest import DIM_DEGREES


def _block_solver(mgop, single_group_solver):
    return MultiGroupLowerTriangularBlockSolver.from_composed(mgop.strategy, single_group_solver)


@pytest.mark.slow
def test_multigroup_lower_triangular_vs_global_solve(bb_mgop):
    '''On a downscattering-only operator the block forward/back-substitution reproduces the global
    BiCGStab solve -- in both directions -- and leaves a near-zero residual.'''
    mgop = _downscattering_only(bb_mgop[0])
    bicgstab = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)
    block    = _block_solver(mgop, bicgstab)

    rhs = jnp.array(onp.random.RandomState(14012026).rand(*mgop.in_structure().shape))

    res_global = lx.linear_solve(mgop, rhs, bicgstab).value
    res_block  = lx.linear_solve(mgop, rhs, block).value
    onp.testing.assert_allclose(onp.array(res_block), onp.array(res_global), rtol=1e-12, atol=1e-12)

    res_global_T = lx.linear_solve(mgop.transpose(), rhs, bicgstab).value
    res_block_T  = lx.linear_solve(mgop.transpose(), rhs, block).value
    onp.testing.assert_allclose(onp.array(res_block_T), onp.array(res_global_T), rtol=1e-12, atol=1e-12)

    onp.testing.assert_allclose(onp.array(mgop.mv(res_global) - rhs), 0.0, atol=1e-10)
    onp.testing.assert_allclose(onp.array(mgop.transpose().mv(res_global_T) - rhs), 0.0, atol=1e-10)


def test_multigroup_rhs_construction(bb_mgop):
    
    from jax_sn.solvers.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs
    from bb_setup import single_group_operator_from_mgop, source_operator_from_sgop

    mgop, source_total = _downscattering_only(bb_mgop[0]), bb_mgop[1]
    solver = _block_solver(mgop, lx.BiCGStab(rtol=1e-20, atol=1e-20, max_steps=300))

    # hand-built: first-collision moment source (M T_0^-1 H on the angular source) in group 0, zero elsewhere.
    rhs_eg0   = source_operator_from_sgop(single_group_operator_from_mgop(mgop, 0)).mv(source_total)
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0)] * (mgop.n_energy_groups - 1), axis=0)
    values_manual = lx.linear_solve(mgop, rhs_total, solver).value

    rhs_impl = IsotropicSourceSingleGroup.from_operator(mgop, source_total[0] * onp.sqrt(4.0 * onp.pi), 0)
    chain_strategy = ComposedMultiGroupChainStrategy.from_composed(mgop.strategy)
    values_impl = solve_multigroup_operator_rhs(mgop, rhs_impl, solver, chain_strategy).value

    onp.testing.assert_allclose(onp.array(values_impl), onp.array(values_manual), rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_multigroup_reduced_quadrature(bb_mgop):
    from jax_sn.quadrature_set import QuadratureSetReduced
    from jax_sn.operator_creation.builders import MultiGroupBuilder
    from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
    from jax_sn.solvers.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs

    mgop, _, solution_domain, bb_args = bb_mgop
    dim = bb_args[0]
    if dim == 3:
        pytest.skip("dim=3: the angular quadrature cannot be reduced further")

    full_qset = mgop.data.angular_data.quadrature_set

    def _solve(qset):
        strategy = DefaultMultiGroupStrategy.default_multi_group_strategy(checkpoint=False)
        built = MultiGroupBuilder.from_solution_domain(
            solution_domain, quadrature_set=qset, strategy=strategy).multi_group_operator
        op  = _downscattering_only(built)
        src = IsotropicSourceSingleGroup.from_operator(op, jnp.ones(op.out_structure().shape[2:]), 0)
        solver = _block_solver(op, lx.BiCGStab(1e-13, 1e-15, max_steps=160))
        chain_strategy = ComposedMultiGroupChainStrategy.from_composed(op.strategy)
        return solve_multigroup_operator_rhs(op, src, solver, chain_strategy).value

    result_full = _solve(full_qset)
    result_2d   = _solve(QuadratureSetReduced.from_quadrature_set(full_qset, 2))
    onp.testing.assert_allclose(onp.array(result_2d), onp.array(result_full), rtol=1e-8, atol=1e-9)
    if dim == 1:
        result_1d = _solve(QuadratureSetReduced.from_quadrature_set(full_qset, 1))
        onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_full), rtol=1e-8, atol=1e-9)
