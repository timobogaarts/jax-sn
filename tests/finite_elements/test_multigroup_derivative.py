import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
from typing import Tuple
from functools import partial

from jax_sn.operators.transport_operator import TransportOperatorBatched, TransportOperatorMultipleAnglesVmap
from jax_sn.domain.geometric_domain import Domain, TopologicalSortedAngle
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
from jax_sn.operators.mass_operator import MassOperatorVmap
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
from jax_sn.operators.scatter_operator import GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan, MultiGroupLowerTriangularBlockSolver
from jax_sn.parametrisation.multigroup_parametrisation import MultigroupOperatorParametrisedVerticesXS

import equinox as eqx
from jax_sn.domain.domain_generation import create_cartesian_setup

import lineax as lx
import jax_sn

onp.set_printoptions(linewidth = 200)

def _get_bb_verts_conn_xs_sources(dim, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim):
    from finite_elements.test_breeding_blanket import bb_data_dict
    data_dictionary = bb_data_dict() # dictionary with matrices
    xs_data_dict = {name : CrossSectionData(total = jnp.array(data_dictionary[name][0]), scattering = jnp.array(data_dictionary[name][1])) for name in data_dictionary.keys()}    
    other_dims   = [[(d_other_dim, n_other_dim)]]*(dim - 1)
    region_names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone','BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']
    xs_regions = [xs_data_dict[i] for i in region_names]
    region_size_res = [(size, n_elem) for size, n_elem in zip(d_elem_regions, n_elem_regions)]    
    source_eg0 = [1 if i ==0 else 0 for i in range(len(region_names))]

    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [region_size_res] + other_dims, xs_regions, source_eg0, 0)
    return verts, conn, cross_sections, sources

def setup_from_verts(verts, conn, cross_sections, sources,  degree, tn_order):        
    element_type                         = BasixLagrangianSimplex(degree, verts.shape[-1])
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
    # isotropic flat source
    sources = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    return solution_domain, sources, jax_sn.quadrature_set.create_tn_quadrature_set(tn_order)

def _setup_operators(solution_domain: SolutionDomain, sources: jnp.ndarray, quadrature_set: jax_sn.quadrature_set.QuadratureSet, batch_size = 50) -> Tuple[MultiGroupOperatorScan, jnp.ndarray, SolutionDomain]:        
    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    topological_sorted_angles = TopologicalSortedAngle.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAnglesVmap(TransportOperatorBatched(solution_domain.solution_domain_data, topological_sorted_angles, 0, batch_size))
    mass_operator             = MassOperatorVmap.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)    
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = GatherScatterOperatorBroadcasted(cross_sections_indexed=solution_domain.domain.cross_sections_indexed)
    source_total              = jnp.broadcast_to(sources[None, :, :], (topological_sorted_angles.angle.shape[0],) + sources.shape) # source for all angles
    Aop                       = MultiGroupOperatorScan(moment_operator, discrete_operator, transport_operator, mass_operator, gather_scatter_operator, quadrature_set, n_energy_groups)
    return Aop.convert_to_lower_block_triangular(), source_total, solution_domain

@eqx.filter_jit
def get_rhs(MGOP, source_eg0):
    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.inv(MGOP.mass_operator.mv(source_eg0)))

    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups -1)], axis=0)
    return rhs_total
from finite_elements.test_breeding_blanket import bb_data_dict, base_region_sizes, base_resolutions,  _get_bb_args, _setup_bb_args, _get_bb_args


def modify_setup_mgop(dim, degree, modify_verts_xs):
    verts, conn, cross_sections, sources = _get_bb_verts_conn_xs_sources(dim, base_region_sizes(), base_resolutions(), 5.0, 1)
    verts, cross_sections = modify_verts_xs(verts, cross_sections)

    solution_domain, sources, quadrature_set = setup_from_verts(verts, conn, cross_sections, sources, degree, 3)
    
    return _setup_operators(solution_domain, sources, quadrature_set)    



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




    