'''Multigroup operator tests: the property/consistency battery over the multigroup *lift* variants, plus
the block-solver equivalence tests.

Every variant shares one chain + gather-scatter (the single-group body); they differ only in how that body
is lifted over the energy-group axis: a serialising scan, a checkpointed scan, or a full vmap. All are
compared against the pivot (plain scan) via the shared BaseOperatorProperties battery.

The block-solver equivalence tests (bottom) exercise MultiGroupLowerTriangularBlockSolver against the
generic global BiCGStab solve on the same operator: they assert the direct forward/back-substitution
reproduces the global solve, that the packaged RHS construction matches a hand-built preconditioned RHS,
and that a reduced angular quadrature leaves the solution unchanged. The continuous-adjoint identity and
the frozen-flux accuracy oracle remain pending the adjoint (variance_reduction) port.
'''

import equinox as eqx
import pytest
import jax.numpy as jnp
import numpy as onp
import lineax as lx

from jax_sn.operators.multi_group_operator import MultiGroupScanStrategy, MultiGroupVmapStrategy
from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver
from .operator_util import BaseOperatorProperties
from conftest import DIM_DEGREES


def _relift(mgop, strategy):
    '''Swap the multigroup lift, keeping the shared chain + gather-scatter sub-strategies and data.'''
    return eqx.tree_at(lambda op: op.strategy, mgop, strategy)


def _scan(mgop, checkpoint):
    s = mgop.strategy
    return _relift(mgop, MultiGroupScanStrategy(s.single_group_chain_strategy,
                                                s.single_group_gather_scatter_strategy, checkpoint=checkpoint))


def _vmap(mgop):
    s = mgop.strategy
    return _relift(mgop, MultiGroupVmapStrategy(s.single_group_chain_strategy,
                                                s.single_group_gather_scatter_strategy))


# name -> builder(mgop) for each lift variant. `scan` (no checkpoint) is the pivot the reference is built
# from, so it is trivially self-consistent; the other two must reproduce it.
MG_VARIANTS = {
    "scan":      lambda mgop: _scan(mgop, checkpoint=False),
    "scan_ckpt": lambda mgop: _scan(mgop, checkpoint=True),
    "vmap":      _vmap,
}


def _combos():
    return [pytest.param((dim, degree), builder, id=f"dim{dim}-deg{degree}-{name}")
            for (dim, degree) in DIM_DEGREES
            for name, builder in MG_VARIANTS.items()]


@pytest.mark.parametrize("bb_mgop_dim_degree, build_variant", _combos(), indirect=["bb_mgop_dim_degree"])
class TestMultiGroupOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, build_variant):
        return build_variant(bb_mgop[0])

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: bb_mgop[0]


# ==================================================================================================
# Block-solver equivalence: MultiGroupLowerTriangularBlockSolver vs the generic global BiCGStab solve.
# ==================================================================================================
from bb_setup import downscattering_only as _downscattering_only


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
    '''The packaged fixed-source RHS (IsotropicSourceSingleGroup + solve_multigroup_operator_rhs) yields
    the same solution as a hand-built preconditioned RHS solved with the block solver. The isotropic
    l=0 source is scaled by sqrt(4*pi) so the discrete operator maps it back to the flat angular source.'''
    from jax_sn.solvers.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs
    from bb_setup import single_group_operator_from_mgop, source_operator_from_sgop

    mgop, source_total = _downscattering_only(bb_mgop[0]), bb_mgop[1]
    solver = _block_solver(mgop, lx.BiCGStab(rtol=1e-20, atol=1e-20, max_steps=300))

    # hand-built: first-collision moment source (M T_0^-1 H on the angular source) in group 0, zero elsewhere.
    rhs_eg0   = source_operator_from_sgop(single_group_operator_from_mgop(mgop, 0)).mv(source_total)
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0)] * (mgop.n_energy_groups - 1), axis=0)
    values_manual = lx.linear_solve(mgop, rhs_total, solver).value

    rhs_impl = IsotropicSourceSingleGroup.from_operator(mgop, source_total[0] * onp.sqrt(4.0 * onp.pi), 0)
    values_impl = solve_multigroup_operator_rhs(mgop, rhs_impl, solver,
                                                mgop.strategy.single_group_chain_strategy).value

    onp.testing.assert_allclose(onp.array(values_impl), onp.array(values_manual), rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_multigroup_reduced_quadrature(bb_mgop):
    '''A reduced angular quadrature (fewer angular dimensions) gives the identical fixed-source solution
    to the full 3D set when the geometry lives in lower dimensions: dim=1 -> 1D/2D/3D all agree,
    dim=2 -> 2D/3D agree, dim=3 -> nothing to reduce (skipped). The source must be physically
    representable by every set -- a random moment-space RHS is not (a reduced set cannot carry the
    higher-m moments) -- so we drive with an isotropic l=0 source, which every quadrature integrates
    identically; the m!=0 solution moments then stay zero by symmetry and the rest agree.'''
    from jax_sn.quadrature_set import QuadratureSetReduced
    from jax_sn.operator_creation.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
    from jax_sn.solvers.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs

    mgop, _, solution_domain, bb_args = bb_mgop
    dim = bb_args[0]
    if dim == 3:
        pytest.skip("dim=3: the angular quadrature cannot be reduced further")

    full_qset = mgop.data.angular_data.quadrature_set

    def _solve(qset):
        strategy = DefaultMultiGroupStrategy.default_multi_group_strategy(checkpoint=False)
        op  = _downscattering_only(create_multi_group_operator(solution_domain, qset, strategy))
        src = IsotropicSourceSingleGroup.from_operator(op, jnp.ones(op.out_structure().shape[2:]), 0)
        solver = _block_solver(op, lx.BiCGStab(1e-13, 1e-15, max_steps=160))
        return solve_multigroup_operator_rhs(op, src, solver, op.strategy.single_group_chain_strategy).value

    result_full = _solve(full_qset)
    result_2d   = _solve(QuadratureSetReduced.from_quadrature_set(full_qset, 2))
    onp.testing.assert_allclose(onp.array(result_2d), onp.array(result_full), rtol=1e-8, atol=1e-9)
    if dim == 1:
        result_1d = _solve(QuadratureSetReduced.from_quadrature_set(full_qset, 1))
        onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_full), rtol=1e-8, atol=1e-9)
