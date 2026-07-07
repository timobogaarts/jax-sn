'''Algebraic / self-consistency tests for the multigroup operator.

Built via the shared ``bb_mgop`` fixture, these check properties that need no external ground
truth: implementation equivalence (Scan vs alternatives), Euclidean adjointness, RHS-construction
equivalence, lower-triangular vs global solver equivalence, angular-quadrature reduction, and the
continuous-adjoint identity. The accuracy-vs-oracle test lives in test_multigroup_operator_solve.py.
'''
import numpy as onp
import jax.numpy as jnp
import jax
import pytest

jax.config.update("jax_enable_x64", True)

from finite_elements.operator_util import _check_adjoint_of_operator


# --- implementation equivalence -------------------------------------------------------------
# Alternative MultiGroupOperator implementations, each checked against the Scan reference that
# bb_mgop provides. Add new backends (e.g. a future vmap variant) here -- one line, like the
# transport-operator factory lists.
def _mgop_checkpoint(base):
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import MultiGroupOperatorScanCheckpoint
    return MultiGroupOperatorScanCheckpoint.from_multigroup_operator(base)


MG_IMPL_FACTORIES = [pytest.param(_mgop_checkpoint, id="ScanCheckpoint")]


@pytest.mark.parametrize("mg_factory", MG_IMPL_FACTORIES)
def test_multigroup_implementations_equivalent(bb_mgop, mg_factory):
    '''Each alternative implementation reproduces the Scan reference matvec (forward + transpose).'''
    base = bb_mgop[0]
    op = mg_factory(base)

    onp.random.seed(15012026)
    x = jnp.array(onp.random.rand(*base.in_structure().shape))

    onp.testing.assert_allclose(onp.array(op.mv(x)), onp.array(base.mv(x)), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(onp.array(op.transpose().mv(x)), onp.array(base.transpose().mv(x)), rtol=1e-12, atol=1e-12)


def test_multigroup_adjoint(bb_mgop):
    '''Euclidean adjointness <Ax, y> == <x, A^T y> (and transpose idempotence).'''
    _check_adjoint_of_operator(bb_mgop[0])


def test_multigroup_rhs_construction(bb_mgop):
    '''IsotropicSourceSingleGroup + solve_multigroup_operator_rhs produces the same solution as
    manually building the preconditioned RHS and calling solve_lower_triangular.'''
    import lineax as lx
    MGOP, source_eg0, solution_domain, bb_args = bb_mgop
    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups - 1)], axis=0)
    bicgstab_solver = lx.BiCGStab(1e-20, 1e-20, max_steps=300)

    from jax_sn.operators.multi_group_operator import solve_lower_triangular, MultiGroupLowerTriangularBlockSolver
    from jax_sn.operators.multi_group_operator.lower_triangular_solver import SolverDiagnostics
    from jax_sn.operators.multi_group_operator.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs

    values_lower_triangular = solve_lower_triangular(MGOP, bicgstab_solver, rhs_total, SolverDiagnostics(False, False))[0]

    rhs_impl = IsotropicSourceSingleGroup.from_operator(MGOP, source_eg0[0] * onp.sqrt(onp.pi * 4), 0)
    solver_lt = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)
    solution2 = solve_multigroup_operator_rhs(MGOP, rhs_impl, solver_lt)

    onp.testing.assert_allclose(onp.array(values_lower_triangular), onp.array(solution2.value), rtol=1e-12, atol=1e-12)


def test_multigroup_lower_triangular_vs_global_solve(bb_mgop):
    '''The lower-triangular block solver and the global BiCGStab solver agree (forward + transpose)
    and leave a near-zero residual. Uses convert_to_lower_block_triangular() since the full BB
    problem has thermal upscattering that breaks strict lower-triangularity.'''
    import lineax as lx
    MGOP, _, _, _ = bb_mgop
    MGOP = MGOP.convert_to_lower_block_triangular()
    from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver

    bicgstab_solver               = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)
    lower_triangular_block_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)

    onp.random.seed(14012026)
    rhs_rng = jnp.array(onp.random.rand(*MGOP.in_structure().shape))

    result_total_solve = lx.linear_solve(MGOP, rhs_rng, bicgstab_solver).value
    result_lower_triangular_solve = lx.linear_solve(MGOP, rhs_rng, lower_triangular_block_solver).value
    onp.testing.assert_allclose(onp.array(result_total_solve), onp.array(result_lower_triangular_solve), rtol=1e-12, atol=1e-12)

    result_total_solve_T = lx.linear_solve(MGOP.transpose(), rhs_rng, bicgstab_solver).value
    result_lower_triangular_solve_T = lx.linear_solve(MGOP.transpose(), rhs_rng, lower_triangular_block_solver).value
    onp.testing.assert_allclose(onp.array(result_total_solve_T), onp.array(result_lower_triangular_solve_T), rtol=1e-12, atol=1e-12)

    residual = MGOP.mv(result_total_solve) - rhs_rng
    onp.testing.assert_allclose(onp.array(residual), onp.zeros_like(onp.array(residual)), rtol=1e-12, atol=1e-12)

    residual_T = MGOP.transpose().mv(result_total_solve_T) - rhs_rng
    onp.testing.assert_allclose(onp.array(residual_T), onp.zeros_like(onp.array(residual_T)), rtol=1e-12, atol=1e-12)


def test_multigroup_reduced_quadrature(bb_mgop):
    '''A reduced quadrature set (fewer angular dimensions) gives identical solutions to the full 3D
    set when solving in lower spatial dimensions: dim=1 -> 1D/2D/3D sets all agree; dim=2 -> 2D/3D
    agree; dim=3 -> skipped (no reduction possible).'''
    MGOP, _, solution_domain, bb_args = bb_mgop

    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
    from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
    from jax_sn.quadrature_set import QuadratureSetReduced
    import lineax as lx
    onp.random.seed(14012026)

    qset_1d = QuadratureSetReduced.from_quadrature_set(MGOP.quadrature_set, 1)
    qset_2d = QuadratureSetReduced.from_quadrature_set(MGOP.quadrature_set, 2)

    dim = bb_args[0]

    def _solve(qset):
        mgop = create_multi_group_operator(solution_domain, qset)
        rhs = jnp.array(onp.random.RandomState(14012026).rand(MGOP.n_energy_groups, solution_domain.n_elements, solution_domain.n_basis))
        rhs_stack = jnp.broadcast_to(rhs[:, None, :, :], (mgop.n_energy_groups, qset.n_angles, solution_domain.n_elements, solution_domain.n_basis))
        solver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160))
        return _solve_multigroup_operator_rhs_scan(mgop, rhs_stack, solver).value

    if dim == 1:
        result_1d, result_2d, result_3d = _solve(qset_1d), _solve(qset_2d), _solve(MGOP.quadrature_set)
        onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_2d), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_3d), rtol=1e-12, atol=1e-12)
    elif dim == 2:
        result_2d, result_3d = _solve(qset_2d), _solve(MGOP.quadrature_set)
        onp.testing.assert_allclose(onp.array(result_2d), onp.array(result_3d), rtol=1e-12, atol=1e-12)


def test_multigroup_continuous_adjoint(bb_mgop):
    '''Continuous adjoint identity <psi^dagger, q> == <psi, q^dagger> in 1D, evaluated by
    interpolating solution and source along a line (instead of a full phase-space integral).
    Only loosely accurate by construction (the discrete adjoint is not the adjoint of the discrete
    operator, the integral is on quadrature points, and a random source may be spatially
    unconverged) -- so this checks the implementation is correct and non-trivial, not accuracy.'''
    import lineax as lx
    MGOP, _, solution_domain, _ = bb_mgop

    if solution_domain.dimension != 1:
        pytest.skip("Continuous adjoint test only implemented for 1D: it interpolates along a line instead of a full 6D space integral")

    from jax_sn.operators.multi_group_operator.adjoint_multigroup_operator import AdjointMultiGroupOperatorScan, solve_multigroup_operator_rhs_adjoint
    from jax_sn.operators.multi_group_operator.right_hand_side import TotalSource, solve_multigroup_operator_rhs

    onp.random.seed(25032026)
    source_shape_isotropic = (MGOP.n_energy_groups, MGOP.n_elements, MGOP.n_basis)
    source_total = onp.random.random(source_shape_isotropic)
    source_dagger_total = onp.random.random(source_shape_isotropic)

    rhs = TotalSource.from_operator_and_isotropic_source(MGOP, source_total * onp.sqrt(onp.pi * 4))
    rhs_dagger = TotalSource.from_operator_and_isotropic_source(MGOP, source_dagger_total * onp.sqrt(onp.pi * 4))

    solver = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)
    adjoint_op = AdjointMultiGroupOperatorScan.from_multigroup_operator(MGOP)

    solution_adjoint = solve_multigroup_operator_rhs_adjoint(adjoint_op, rhs_dagger, solver)
    solution = solve_multigroup_operator_rhs(MGOP, rhs, solver)

    x = jnp.linspace(0, jnp.max(MGOP.domain.vertices), 800)
    result = solution_domain.interpolate(x[:, None], solution.value[:, 0])
    result_adjoint = solution_domain.interpolate(x[:, None], solution_adjoint.value[:, 0])
    source_interp = solution_domain.interpolate(x[:, None], source_total)
    source_interp_dagger = solution_domain.interpolate(x[:, None], source_dagger_total)

    psidagger_q = jnp.sum(result_adjoint * source_interp)
    psi_qdagger = jnp.sum(result * source_interp_dagger)

    onp.testing.assert_allclose(onp.array(psidagger_q), onp.array(psi_qdagger), rtol=1e-2, atol=1e-2)
