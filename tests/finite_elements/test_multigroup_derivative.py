import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
from functools import partial

import equinox as eqx
import lineax as lx

from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
from jax_sn.parametrisation.multigroup_parametrisation import MultigroupOperatorParametrisedVerticesXS
from finite_elements.bb_setup import base_region_sizes, base_resolutions, bb_mesh, setup_bb_from_mesh, build_multigroup_operator

onp.set_printoptions(linewidth = 200)


@eqx.filter_jit
def get_rhs(MGOP, source_eg0):
    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.inv(MGOP.mass_operator.mv(source_eg0)))

    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups -1)], axis=0)
    return rhs_total


def modify_setup_mgop(dim, degree, modify_verts_xs):
    # Perturbing vertex positions changes the geometry, so (unlike the cross-section-only
    # test_single_group_ad) we rebuild the operator from the perturbed mesh via the shared bb_setup
    # seam. Solved lower-block-triangular to match the AD path in solve_result_everything.
    verts, conn, cross_sections, sources = bb_mesh(dim, base_region_sizes(), base_resolutions(), 5.0, 1)
    verts, cross_sections = modify_verts_xs(verts, cross_sections)

    solution_domain, sources, quadrature_set = setup_bb_from_mesh(verts, conn, cross_sections, sources, degree, 3)
    Aop, source_total, solution_domain = build_multigroup_operator(solution_domain, sources, quadrature_set)
    return Aop.convert_to_lower_block_triangular(), source_total, solution_domain



@eqx.filter_jit
def solve_and_sum(MGOP, rhs, solver):
    solution = lx.linear_solve(MGOP, rhs, solver)
    return jnp.sum(solution.value)



@eqx.filter_jit
def solve_result_everything(theta, MGOP, rhs_base, ltri_solver):    
    MGOP_new = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(MGOP, theta)
    solution_value =  lx.linear_solve(MGOP_new, rhs_base, solver=ltri_solver).value
    return jnp.sum(solution_value)

grad_theta_everything = eqx.filter_jit(jax.grad(solve_result_everything, argnums=0))

def _modify_vertices(verts, xs, dx, vert_mod):
    verts = verts.at[vert_mod].set(verts[vert_mod] + jnp.array([dx, 0.0]))
    return verts, xs

def _modify_xs(verts, xs, dx, xs_mod, total = True):    
    if total:
        xs = eqx.tree_at(lambda x : x.cross_section_data.total, xs, replace_fn=lambda t : t.at[xs_mod].set(t[xs_mod] + dx)) 
    else:
        xs = eqx.tree_at(lambda x : x.cross_section_data.scattering, xs, replace_fn=lambda t : t.at[xs_mod].set(t[xs_mod] + dx)) 
    return verts, xs

def _dim_deg_test():
    return 2,2

def f_cases(dx):
    
    verts_cases = [0,5, 25]

    xs_total_cases = [(0,0),(3,3), (5,1)]
    xs_scatter_cases = [(3,1,3,2)]

    verts_rep_f =  [partial(_modify_vertices, dx=dx, vert_mod=vert_mod) for vert_mod in verts_cases]
    xs_tot_rep_f   = [partial(_modify_xs, dx=dx, xs_mod=xs_mod, total = True) for xs_mod in xs_total_cases]
    xs_scat_rep_f   = [partial(_modify_xs, dx=dx, xs_mod=xs_mod, total = False) for xs_mod in xs_scatter_cases]

    verts_check_f = [lambda theta, vert_mod=vert_mod: theta.vertices[vert_mod,0] for vert_mod in verts_cases]
    xs_check_f   = [lambda theta, xs_mod=xs_mod: theta.cross_section_data.total[xs_mod] for xs_mod in xs_total_cases]
    xs_scat_check_f   = [lambda theta, xs_mod=xs_mod: theta.cross_section_data.scattering[xs_mod] for xs_mod in xs_scatter_cases]

    

    return verts_rep_f +  xs_tot_rep_f + xs_scat_rep_f, verts_check_f + xs_check_f + xs_scat_check_f

def test_mg_ad():
    print_f = False
    dim, degree = _dim_deg_test()
    bicgstab_solver = lx.BiCGStab(1e-12, 1e-12, max_steps = 160)
    lower_triangular_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)

    MGOP, source_eg0, _ = modify_setup_mgop(dim, degree, lambda v,c : (v,c))

    theta_everything_base = MultigroupOperatorParametrisedVerticesXS.theta_base(MGOP)

    rhs = get_rhs(MGOP, source_eg0)

    df_dxi = grad_theta_everything(theta_everything_base, MGOP, rhs, lower_triangular_solver)
    
    dx = 1e-4


    for modify_f, check_f in zip(*f_cases(dx)):
        if print_f:
            print("Testing case:", modify_f, check_f)
        #print(" with ", modify_f(MGOP.transport_operator.transport_operator.solution_domain_data.domain.vertices, MGOP.transport_operator.transport_operator.solution_domain_data.domain.cross_sections_indexed))
        
        MGOP_mod, _, _ = modify_setup_mgop(dim, degree, modify_f)        

        solve_base     = solve_and_sum(MGOP, rhs, lower_triangular_solver)
        solve_mod      = solve_and_sum(MGOP_mod, rhs, lower_triangular_solver)

        finite_diff_derivative = (solve_mod - solve_base) / 1e-4

        derivative_ad = check_f(df_dxi)

        # get derivative from ad
        if print_f:
            print("Finite diff:", finite_diff_derivative)
            print("AD         :", derivative_ad)
        onp.testing.assert_allclose(finite_diff_derivative, derivative_ad, rtol=1e-3, atol=1e-3)




    