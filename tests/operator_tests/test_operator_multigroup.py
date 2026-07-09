'''Multigroup operator tests: algebraic / self-consistency properties (no external ground truth) plus the
one frozen-reference oracle it owns.

Property tests are built via the shared ``bb_mgop`` fixture: implementation equivalence (Scan vs
alternatives), Euclidean adjointness, RHS-construction equivalence, lower-triangular vs global solver
equivalence, angular-quadrature reduction, and the continuous-adjoint identity. The oracle
(``test_multigroup_solve_accuracy``, against ``key_bb``) lives at the bottom, co-located with the operator
it pins, and runs at the reference degree only (``bb_mgop_reference``).
'''

import pytest
from .operator_util import BaseOperatorProperties
from jax_sn.operators.multi_group_operator.multi_group_operator_imp import MultiGroupOperatorScanCheckpoint, MultiGroupOperatorScan
from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

_DEFAULT_MG  = MultiGroupOperatorScan   # pure-JAX reference (pivot) every variant is compared against

MG_STRATEGIES = {MultiGroupOperatorScan: Spec(), 
                   MultiGroupOperatorScanCheckpoint: Spec()}  


@parametrize_strategies(MG_STRATEGIES)
class TestMultiGroupOperator(BaseOperatorProperties):    

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):        
        return strategy_cls.from_multigroup_operator(bb_mgop[0], **kwargs)

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return _DEFAULT_MG.from_multigroup_operator(bb_mgop[0])


# # --- solver-equivalence / self-consistency properties (no external reference) ---------------
# # Moved here from test_multigroup_operator_solve.py so that file holds only the frozen-oracle accuracy
# # test: these all assert internal agreement (RHS construction, lower-triangular vs global solve, reduced
# # quadrature, continuous adjoint), not agreement with stored data.
# def test_multigroup_rhs_construction(bb_mgop):
#     '''IsotropicSourceSingleGroup + solve_multigroup_operator_rhs produces the same solution as
#     manually building the preconditioned RHS and calling solve_lower_triangular.'''
#     import lineax as lx
#     MGOP, source_eg0, _, _ = bb_mgop
#     rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
#     rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups - 1)], axis=0)
#     bicgstab_solver = lx.BiCGStab(1e-20, 1e-20, max_steps=300)

#     from jax_sn.operators.multi_group_operator import solve_lower_triangular, MultiGroupLowerTriangularBlockSolver
#     from jax_sn.operators.multi_group_operator.lower_triangular_solver import SolverDiagnostics
#     from jax_sn.operators.multi_group_operator.right_hand_side import IsotropicSourceSingleGroup, solve_multigroup_operator_rhs

#     values_lower_triangular = solve_lower_triangular(MGOP, bicgstab_solver, rhs_total, SolverDiagnostics(False, False))[0]

#     rhs_impl = IsotropicSourceSingleGroup.from_operator(MGOP, source_eg0[0] * onp.sqrt(onp.pi * 4), 0)
#     solver_lt = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)
#     solution2 = solve_multigroup_operator_rhs(MGOP, rhs_impl, solver_lt)

#     onp.testing.assert_allclose(onp.array(values_lower_triangular), onp.array(solution2.value), rtol=1e-12, atol=1e-12)


# def test_multigroup_lower_triangular_vs_global_solve(bb_mgop):
#     '''The lower-triangular block solver and the global BiCGStab solver agree (forward + transpose)
#     and leave a near-zero residual. Uses convert_to_lower_block_triangular() since the full BB
#     problem has thermal upscattering that breaks strict lower-triangularity.'''
#     import lineax as lx
#     MGOP, _, _, _ = bb_mgop
#     MGOP = MGOP.convert_to_lower_block_triangular()
#     from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver

#     bicgstab_solver               = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)
#     lower_triangular_block_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)

#     onp.random.seed(14012026)
#     rhs_rng = jnp.array(onp.random.rand(*MGOP.in_structure().shape))

#     result_total_solve = lx.linear_solve(MGOP, rhs_rng, bicgstab_solver).value
#     result_lower_triangular_solve = lx.linear_solve(MGOP, rhs_rng, lower_triangular_block_solver).value
#     onp.testing.assert_allclose(onp.array(result_total_solve), onp.array(result_lower_triangular_solve), rtol=1e-12, atol=1e-12)

#     result_total_solve_T = lx.linear_solve(MGOP.transpose(), rhs_rng, bicgstab_solver).value
#     result_lower_triangular_solve_T = lx.linear_solve(MGOP.transpose(), rhs_rng, lower_triangular_block_solver).value
#     onp.testing.assert_allclose(onp.array(result_total_solve_T), onp.array(result_lower_triangular_solve_T), rtol=1e-12, atol=1e-12)

#     residual = MGOP.mv(result_total_solve) - rhs_rng
#     onp.testing.assert_allclose(onp.array(residual), onp.zeros_like(onp.array(residual)), rtol=1e-12, atol=1e-12)

#     residual_T = MGOP.transpose().mv(result_total_solve_T) - rhs_rng
#     onp.testing.assert_allclose(onp.array(residual_T), onp.zeros_like(onp.array(residual_T)), rtol=1e-12, atol=1e-12)


# def test_multigroup_reduced_quadrature(bb_mgop):
#     '''A reduced quadrature set (fewer angular dimensions) gives identical solutions to the full 3D
#     set when solving in lower spatial dimensions: dim=1 -> 1D/2D/3D sets all agree; dim=2 -> 2D/3D
#     agree; dim=3 -> skipped (no reduction possible).'''
#     MGOP, _, solution_domain, bb_args = bb_mgop

#     from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
#     from jax_sn.operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
#     from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
#     from jax_sn.quadrature_set import QuadratureSetReduced
#     import lineax as lx
#     onp.random.seed(14012026)

#     qset_1d = QuadratureSetReduced.from_quadrature_set(MGOP.quadrature_set, 1)
#     qset_2d = QuadratureSetReduced.from_quadrature_set(MGOP.quadrature_set, 2)

#     dim = bb_args[0]

#     def _solve(qset):
#         mgop = create_multi_group_operator(solution_domain, qset)
#         rhs = jnp.array(onp.random.RandomState(14012026).rand(MGOP.n_energy_groups, solution_domain.n_elements, solution_domain.n_basis))
#         rhs_stack = jnp.broadcast_to(rhs[:, None, :, :], (mgop.n_energy_groups, qset.n_angles, solution_domain.n_elements, solution_domain.n_basis))
#         solver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160))
#         return _solve_multigroup_operator_rhs_scan(mgop, rhs_stack, solver).value

#     if dim == 1:
#         result_1d, result_2d, result_3d = _solve(qset_1d), _solve(qset_2d), _solve(MGOP.quadrature_set)
#         onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_2d), rtol=1e-12, atol=1e-12)
#         onp.testing.assert_allclose(onp.array(result_1d), onp.array(result_3d), rtol=1e-12, atol=1e-12)
#     elif dim == 2:
#         result_2d, result_3d = _solve(qset_2d), _solve(MGOP.quadrature_set)
#         onp.testing.assert_allclose(onp.array(result_2d), onp.array(result_3d), rtol=1e-12, atol=1e-12)


# def test_multigroup_continuous_adjoint(bb_mgop):
#     '''Continuous adjoint identity <psi^dagger, q> == <psi, q^dagger> in 1D, evaluated by
#     interpolating solution and source along a line (instead of a full phase-space integral).
#     Only loosely accurate by construction (the discrete adjoint is not the adjoint of the discrete
#     operator, the integral is on quadrature points, and a random source may be spatially
#     unconverged) -- so this checks the implementation is correct and non-trivial, not accuracy.'''
#     import lineax as lx
#     MGOP, _, solution_domain, _ = bb_mgop

#     if solution_domain.dimension != 1:
#         pytest.skip("Continuous adjoint test only implemented for 1D: it interpolates along a line instead of a full 6D space integral")

#     from jax_sn.operators.multi_group_operator.adjoint_multigroup_operator import AdjointMultiGroupOperatorScan, solve_multigroup_operator_rhs_adjoint
#     from jax_sn.operators.multi_group_operator.right_hand_side import TotalSource, solve_multigroup_operator_rhs

#     onp.random.seed(25032026)
#     source_shape_isotropic = (MGOP.n_energy_groups, MGOP.n_elements, MGOP.n_basis)
#     source_total = onp.random.random(source_shape_isotropic)
#     source_dagger_total = onp.random.random(source_shape_isotropic)

#     rhs = TotalSource.from_operator_and_isotropic_source(MGOP, source_total * onp.sqrt(onp.pi * 4))
#     rhs_dagger = TotalSource.from_operator_and_isotropic_source(MGOP, source_dagger_total * onp.sqrt(onp.pi * 4))

#     solver = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)
#     adjoint_op = AdjointMultiGroupOperatorScan.from_multigroup_operator(MGOP)

#     solution_adjoint = solve_multigroup_operator_rhs_adjoint(adjoint_op, rhs_dagger, solver)
#     solution = solve_multigroup_operator_rhs(MGOP, rhs, solver)

#     x = jnp.linspace(0, jnp.max(MGOP.domain.vertices), 800)
#     result = solution_domain.interpolate(x[:, None], solution.value[:, 0])
#     result_adjoint = solution_domain.interpolate(x[:, None], solution_adjoint.value[:, 0])
#     source_interp = solution_domain.interpolate(x[:, None], source_total)
#     source_interp_dagger = solution_domain.interpolate(x[:, None], source_dagger_total)

#     psidagger_q = jnp.sum(result_adjoint * source_interp)
#     psi_qdagger = jnp.sum(result * source_interp_dagger)

#     onp.testing.assert_allclose(onp.array(psidagger_q), onp.array(psi_qdagger), rtol=1e-2, atol=1e-2)


# # --- frozen reference (oracle) --------------------------------------------------------------
# # The one multigroup test tied to a stored bundle (key_bb): dim 1,2 are external-solver ground truth,
# # dim 3 is self-pinned to the current implementation (see tests/reference_data/__init__.py). Co-located
# # with the operator; runs at the reference degree only (bb_mgop_reference). Both the lower-triangular and
# # the global (BiCGStab) solve are checked against the same stored scalar flux -- the global one only to
# # its own convergence tolerance (thermal upscattering), hence the looser 6e-4.
# # class TestMultigroupSolveReference(BaseOperatorReference):
# #     def bundle_key(self, dim):
# #         from reference_data import key_bb
# #         return key_bb(dim)

# #     def comparisons(self, bb_mgop_reference, bundle):
# #         import lineax as lx
# #         from operator_tests.bb_setup import _interp_x
# #         from jax_sn.operators.multi_group_operator import solve_lower_triangular
# #         from jax_sn.operators.multi_group_operator.lower_triangular_solver import SolverDiagnostics
# #         MGOP, source_eg0, solution_domain, bb_args = bb_mgop_reference

# #         rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
# #         rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups - 1)], axis=0)
# #         bicgstab_solver = lx.BiCGStab(1e-20, 1e-20, max_steps=300)

# #         values_lower_triangular = solve_lower_triangular(MGOP, bicgstab_solver, rhs_total, SolverDiagnostics(False, False))[0]
# #         values_total            = lx.linear_solve(MGOP, rhs_total, bicgstab_solver).value

# #         dim, d_other = bb_args[0], bb_args[-3]
# #         x = _interp_x()
# #         xyz = jnp.stack([x] + [jnp.full_like(x, d_other / 2)] * (dim - 1), axis=-1)

# #         phi_lower_triangular = [solution_domain.interpolate(xyz, values_lower_triangular[i, 0]) for i in range(values_lower_triangular.shape[0])]
# #         phi_total            = [solution_domain.interpolate(xyz, values_total[i, 0]) for i in range(values_total.shape[0])]

# #         expected = bundle['phi']  # per-group interpolated scalar flux
#         yield "lower_triangular_solve", phi_lower_triangular, expected, 1e-10, 0.0
#         yield "global_bicgstab_solve",  phi_total,            expected, 6e-4, 0.0

