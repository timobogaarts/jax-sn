import numpy as onp
import jax.numpy as jnp
import jax
import pytest

jax.config.update("jax_enable_x64", True)

from jax_sn.operators.multi_group_operator.adjoint_multigroup_operator import solve_multigroup_operator_rhs_adjoint
from jax_sn.operators.multi_group_operator.right_hand_side import TotalSource, solve_multigroup_operator_rhs

from finite_elements.bb_setup import _interp_x


def _get_expected_values_reference(dim, degree):
    # Frozen reference: per-group interpolated scalar flux (see tests/reference_data/_generate.py)
    from reference_data import load, key_bb
    return load(key_bb(dim))['phi']


def test_bb_right_hand_side(bb_mgop):
    '''
    Verify that IsotropicSourceSingleGroup + solve_multigroup_operator_rhs produces the same
    solution as manually constructing the preconditioned RHS and calling solve_lower_triangular.
    '''
    import lineax as lx
    MGOP, source_eg0, solution_domain, bb_args = bb_mgop
    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups -1)], axis =0)
    bicgstab_solver = lx.BiCGStab(1e-20, 1e-20, max_steps = 300)

    from jax_sn.operators.multi_group_operator import solve_lower_triangular
    from jax_sn.operators.multi_group_operator.lower_triangular_solver import SolverDiagnostics

    diags = SolverDiagnostics(False, False)

    values_lower_triangular = solve_lower_triangular(MGOP, bicgstab_solver, rhs_total, diags)[0]

    from jax_sn.operators.multi_group_operator.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs
    rhs_impl = IsotropicSourceSingleGroup.from_operator(MGOP, source_eg0[0] * onp.sqrt(onp.pi * 4), 0)

    from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
    solver_lt = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)
    solution2 = solve_multigroup_operator_rhs(MGOP, rhs_impl, solver_lt)

    onp.testing.assert_allclose(onp.array(values_lower_triangular), onp.array(solution2.value), rtol=1e-12, atol = 1e-12)



    

def test_bb(bb_mgop, expected_value_function = _get_expected_values_reference):
    '''
    Accuracy test against a frozen reference solution for a 4-group isotropic-source breeding blanket problem.

    Checks both solve_lower_triangular and lx.linear_solve (global BiCGStab) against interpolated
    scalar flux moments from the frozen reference along 1000 points spanning the slab.

    Tolerances are uniform across dimensions (rtol=1e-10 for the lower-triangular solve, 6e-4 for
    the global BiCGStab which only converges to its own tolerance). dim=1,2 are validated against
    an external solver; dim=3 is self-pinned to the current implementation (the external solver
    meshed the slab with a different tetrahedralisation) -- see tests/reference_data/__init__.py.
    '''
    import lineax as lx
    MGOP, source_eg0, solution_domain, bb_args = bb_mgop

    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups -1)], axis =0)
    bicgstab_solver = lx.BiCGStab(1e-20, 1e-20, max_steps = 300)

    from jax_sn.operators.multi_group_operator import solve_lower_triangular
    from jax_sn.operators.multi_group_operator.lower_triangular_solver import SolverDiagnostics

    diag =  SolverDiagnostics(False, False)

    values_lower_triangular = solve_lower_triangular(MGOP, bicgstab_solver, rhs_total, diag)[0]
    values_total            = lx.linear_solve(MGOP,  rhs_total, bicgstab_solver).value

    x = _interp_x()    
    
    dim = bb_args[0]
    d_other = bb_args[-3]
    xyz = jnp.stack([x] + [jnp.full_like(x, d_other / 2)] * (dim-1), axis = -1)

    phi_jsn = [solution_domain.interpolate(xyz, values_lower_triangular[i,0]) for i in range(values_lower_triangular.shape[0])]
    phi_total = [solution_domain.interpolate(xyz, values_total[i,0]) for i in range(values_total.shape[0])]

    dim = bb_args[0]
    degree = bb_args[1]
    rtol = 1e-10
    rtol_total = 6e-4
    

    expected_values = expected_value_function(dim, degree)
    
    onp.testing.assert_allclose(onp.array(expected_values), onp.array(phi_jsn), rtol=rtol)
        
    
    onp.testing.assert_allclose(onp.array(expected_values), onp.array(phi_total), rtol=rtol_total)


def test_bb_moment_operator_transpose(bb_mgop):
    '''
    Verify the Euclidean adjointness of the MultiGroupOperator: checks <y, Ax> == <A^T y, x>
    to rtol=1e-12 with random vectors, using the breeding blanket operator.
    '''
    MGOP, source_eg0, solution_domain, bb_args = bb_mgop
    onp.random.seed(13012026)
    x = onp.random.rand(*MGOP.in_structure().shape)
    y = onp.random.rand(*MGOP.out_structure().shape)

    Ax = MGOP.mv(x)
    ATy = MGOP.transpose().mv(y)
    onp.testing.assert_allclose(onp.sum(y * Ax), onp.sum(ATy * x), rtol=1e-12)


def test_solving_mg_total_single_and_transpose(bb_mgop):
    '''
    Check that the lower-triangular block solver and the global BiCGStab solver agree to rtol=1e-12
    for both the forward and transposed operator. Also verifies the residual is near zero.

    Uses convert_to_lower_block_triangular() since the full BB problem has thermal upscattering
    which breaks strict lower-triangularity.
    '''
    import lineax as lx 
    MGOP,_, _, _ = bb_mgop
    MGOP = MGOP.convert_to_lower_block_triangular() # the example isn't actually lower block triangular due to thermal scattering. we just remove it for this test
    from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver

    bicgstab_solver               = lx.BiCGStab(rtol = 1e-12, atol = 1e-15, max_steps = 160)
    lower_triangular_block_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)
    
    onp.random.seed(14012026)
    rhs_rng = jnp.array(onp.random.rand(*MGOP.in_structure().shape))

    result_total_solve = lx.linear_solve(MGOP, rhs_rng, bicgstab_solver).value
    result_lower_triangular_solve = lx.linear_solve(MGOP, rhs_rng, lower_triangular_block_solver).value

    onp.testing.assert_allclose(onp.array(result_total_solve), onp.array(result_lower_triangular_solve), rtol=1e-12, atol = 1e-12)

    result_total_solve_T = lx.linear_solve(MGOP.transpose(), rhs_rng, bicgstab_solver).value
    result_lower_triangular_solve_T = lx.linear_solve(MGOP.transpose(), rhs_rng, lower_triangular_block_solver).value

    onp.testing.assert_allclose(onp.array(result_total_solve_T), onp.array(result_lower_triangular_solve_T), rtol=1e-12, atol = 1e-12)

    residual = MGOP.mv(result_total_solve) - rhs_rng
    onp.testing.assert_allclose(onp.array(residual), onp.zeros_like(onp.array(residual)), rtol=1e-12, atol = 1e-12)

    residual_T = MGOP.transpose().mv(result_total_solve_T) - rhs_rng
    onp.testing.assert_allclose(onp.array(residual_T), onp.zeros_like(onp.array(residual_T)), rtol=1e-12, atol = 1e-12)


    
def test_mg_with_and_without_checkpoint(bb_mgop):
    '''
    Check that MultiGroupOperatorScanCheckpoint (gradient-checkpointed) gives identical matvec
    results to MultiGroupOperatorScan for both forward and transpose, to rtol=1e-12.
    '''

    MGOP, _, _, _ = bb_mgop

    

    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import MultiGroupOperatorScanCheckpoint

    MGOP_cp = MultiGroupOperatorScanCheckpoint.from_multigroup_operator(MGOP)

    onp.random.seed(15012026)

    rhs_rng = jnp.array(onp.random.rand(*MGOP.in_structure().shape))

    x = MGOP.mv(rhs_rng)
    x_cp = MGOP_cp.mv(rhs_rng)

    onp.testing.assert_allclose(onp.array(x), onp.array(x_cp), rtol=1e-12, atol = 1e-12)

    x2 = MGOP.transpose().mv(rhs_rng)
    x2_cp = MGOP_cp.transpose().mv(rhs_rng) 
    onp.testing.assert_allclose(onp.array(x2), onp.array(x2_cp), rtol=1e-12, atol = 1e-12)

def test_mg_reduced_quadrature(bb_mgop):
    '''
    Verify that a reduced quadrature set (fewer angular dimensions than the full 3D set) gives
    identical solutions to the full set when solving in lower spatial dimensions.

    - dim=1: 1D, 2D, and 3D quadrature sets should all agree (all reduce to the same 1D problem)
    - dim=2: 2D and 3D quadrature sets should agree
    - dim=3: skipped (no reduction possible)
    '''

    MGOP, _, solution_domain, ani_args = bb_mgop
    

    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
    from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
    from jax_sn.quadrature_set import QuadratureSetReduced
    import lineax as lx
    onp.random.seed(14012026)

    qset_1d    = QuadratureSetReduced.from_quadrature_set(MGOP.quadrature_set, 1)
    qset_2d      = QuadratureSetReduced.from_quadrature_set(MGOP.quadrature_set, 2)

    dim = ani_args[0]

    if dim == 1:
        mgop_1d = create_multi_group_operator(solution_domain, qset_1d)
        mgop_2d = create_multi_group_operator(solution_domain, qset_2d)
        mgop_3d = create_multi_group_operator(solution_domain, MGOP.quadrature_set)

        rhs_rng = jnp.array(onp.random.rand(MGOP.n_energy_groups, solution_domain.n_elements, solution_domain.n_basis))
        rhs_1d_stack = jnp.broadcast_to(rhs_rng[:, None, :,:], (mgop_1d.n_energy_groups, qset_1d.n_angles, solution_domain.n_elements, solution_domain.n_basis))
        rhs_2d_stack = jnp.broadcast_to(rhs_rng[:, None, :,:], (mgop_2d.n_energy_groups, qset_2d.n_angles, solution_domain.n_elements, solution_domain.n_basis))
        rhs_3d_stack = jnp.broadcast_to(rhs_rng[:, None, :,:], (mgop_3d.n_energy_groups, MGOP.quadrature_set.n_angles, solution_domain.n_elements, solution_domain.n_basis))

        solver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160))
        result_1d = _solve_multigroup_operator_rhs_scan(mgop_1d, rhs_1d_stack, solver).value
        result_2d = _solve_multigroup_operator_rhs_scan(mgop_2d, rhs_2d_stack, solver).value
        result_3d = _solve_multigroup_operator_rhs_scan(mgop_3d, rhs_3d_stack, solver).value

        onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_2d), rtol=1e-12, atol=1e-12)
        onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_3d), rtol=1e-12, atol=1e-12)
    elif dim == 2:
         
        mgop_2d = create_multi_group_operator(solution_domain, qset_2d)
        mgop_3d = create_multi_group_operator(solution_domain, MGOP.quadrature_set)

        rhs_rng = jnp.array(onp.random.rand(MGOP.n_energy_groups, solution_domain.n_elements, solution_domain.n_basis))
        
        rhs_2d_stack = jnp.broadcast_to(rhs_rng[:, None, :,:], (mgop_2d.n_energy_groups, qset_2d.n_angles, solution_domain.n_elements, solution_domain.n_basis))
        rhs_3d_stack = jnp.broadcast_to(rhs_rng[:, None, :,:], (mgop_3d.n_energy_groups, MGOP.quadrature_set.n_angles, solution_domain.n_elements, solution_domain.n_basis))

        solver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160))        
        result_2d = _solve_multigroup_operator_rhs_scan(mgop_2d, rhs_2d_stack, solver).value
        result_3d = _solve_multigroup_operator_rhs_scan(mgop_3d, rhs_3d_stack, solver).value
        
        onp.testing.assert_allclose(onp.array(result_2d), onp.array(result_3d), rtol=1e-12, atol=1e-12)
    else:
        pass

        
def test_continuous_adjoint_implementation(bb_mgop):
    '''
    Verify that the adjoint operator works in 1D. Instead of a full phase-space integral, we just perform a 1D integral of solution in both cases by interpolating both solution and source.
    Note that because the source is isotropic, we can just test only the scalar flux; the other moments are zeroed out by the application of the source.

    This is not expected to hold to numerical accuracy: the discrete adjoint is not the adjoint of the discrete operator. Furthermore, the integral is on a set of quadrature points.
    Finally, the random source can be very noisy; it's not guaranteed that the solution will be converged spatially.
    
    '''
    import lineax as lx
    MGOP, _, solution_domain, _ = bb_mgop

    if solution_domain.dimension != 1:
        pytest.skip("Continuous adjoint test only implemented for 1D: it interpolates along a line instead of a full 6D space integral")

    onp.random.seed(16012026)
    x = onp.random.rand(*MGOP.in_structure().shape)

    seed = onp.random.seed(25032026)

    source_shape_isotropic = (MGOP.n_energy_groups, MGOP.n_elements, MGOP.n_basis)

    source_total = onp.random.random(source_shape_isotropic)
    source_dagger_total = onp.random.random(source_shape_isotropic)

    from jax_sn.operators.multi_group_operator.adjoint_multigroup_operator import AdjointMultiGroupOperatorScan
    from jax_sn.operators.multi_group_operator.right_hand_side import TotalSource

    rhs = TotalSource.from_operator_and_isotropic_source(MGOP, source_total* onp.sqrt(onp.pi * 4))
    rhs_dagger = TotalSource.from_operator_and_isotropic_source(MGOP, source_dagger_total* onp.sqrt(onp.pi * 4))

    solver = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)

    adjoint_op = AdjointMultiGroupOperatorScan.from_multigroup_operator(MGOP)
    
    solution_adjoint = solve_multigroup_operator_rhs_adjoint(adjoint_op, rhs_dagger, solver)
    solution         =  solve_multigroup_operator_rhs(MGOP, rhs, solver)

    
    x = jnp.linspace(0, jnp.max(MGOP.domain.vertices), 800)
    result = solution_domain.interpolate(x[:,None], solution.value[:,0])
    result_adjoint = solution_domain.interpolate(x[:,None], solution_adjoint.value[:,0])
    source_interp = solution_domain.interpolate(x[:,None], source_total)
    source_interp_dagger = solution_domain.interpolate(x[:,None], source_dagger_total)

    psidagger_q = jnp.sum(result_adjoint * source_interp)
    psi_qdagger = jnp.sum(result * source_interp_dagger)

    onp.testing.assert_allclose(onp.array(psidagger_q), onp.array(psi_qdagger), rtol=1e-2, atol = 1e-2) # this is not expected to be very accurate for reasons described above; just checking the implementation is correct and gives a non-trivial result    