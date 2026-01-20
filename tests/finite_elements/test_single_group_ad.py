import numpy as onp
import jax.numpy as jnp
import jax 
import pytest
import equinox as eqx
import lineax as lx

jax.config.update("jax_enable_x64", True)
from finite_elements.test_breeding_blanket import bb_data_dict, base_region_sizes, base_resolutions,  _get_bb_args, _setup_bb_args, _get_bb_args


def setup_mgop(dim, degree):
    from jax_sn.operators.transport_operator import TransportOperatorBatched, TransportOperatorMultipleAnglesVmap
    MGOP, source_eg0, solution_domain = _setup_bb_args(dim, degree, base_region_sizes(), base_resolutions(), 5.0, 10, 3)
    transport_operator                = TransportOperatorMultipleAnglesVmap(TransportOperatorBatched(MGOP.transport_operator.solution_domain_data, MGOP.transport_operator.topological_sorted_angle, 0, 100))
    MGOP                              = eqx.tree_at(lambda t : t.transport_operator, MGOP, transport_operator)
    return MGOP, source_eg0, solution_domain

def setup_mgop_mod_theta(dim, degree, xs_data_indices, dx):
    from jax_sn.domain.cross_sections import CrossSectionData

    MGOP, source_eg0, solution_domain = setup_mgop(dim, degree)    

    total_xs = MGOP.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.total
    scattering_xs = MGOP.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.scattering    
    if xs_data_indices[0] == 'total':
        total_xs = total_xs.at[xs_data_indices[1]].add(dx)
        
    elif xs_data_indices[0] == 'scattering':
        scattering_xs = scattering_xs.at[xs_data_indices[1]].add(dx)
    else:
        raise ValueError("Invalid xs_data_indices")    
    
    sgop = MGOP.to_single_group_operator(0)

    xs_data_new = CrossSectionData(total_xs, scattering_xs)
    sgop_new = eqx.tree_at(lambda sgop : [sgop.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data, sgop.inscatter_operator.cross_sections_indexed.cross_section_data], sgop,[xs_data_new,xs_data_new])
    return sgop_new, source_eg0, solution_domain

def total_solve(sgop, source_eg0, solver):
    rhs = sgop.moment_operator.mv(sgop.transport_operator.inv(sgop.mass_operator.mv(source_eg0)))
    return lx.linear_solve(sgop, rhs, solver)

def solve_mgop_mod_theta(dim,degree, xs_data_indices, dx, solver):
    sgop_new, source_eg0, solution_domain = setup_mgop_mod_theta(dim, degree, xs_data_indices, dx)
    return total_solve(sgop_new, source_eg0, solver).value.sum()



def dx_mgop_mod_theta(dim, degree, xs_data_indices, dx, solver):
    return (solve_mgop_mod_theta(dim,degree, xs_data_indices, dx, solver) - solve_mgop_mod_theta(dim,degree, xs_data_indices, 0.0, solver)) / dx



def all_testing_derivatives():
    return ['total', (1,0)], ['scattering', (1,0,0,0)], ['scattering', (1,1,0,0)], ['scattering', (2,4,0,0)]


def _dim_degree_tested():
    return (2,2)

from jax_sn.parametrisation import SingleGroupOperatorSigmaAllParametrised
def sum_f(sgop, rhs, theta, solver):        
    sgop_new = SingleGroupOperatorSigmaAllParametrised.reconstruct_operator(sgop, theta)
    return total_solve(sgop_new, rhs, solver).value.sum()  

@pytest.mark.slow
def test_single_group_ad():
    ''' This test computes the derivatives of a single group solve wrt all cross section parameters using two methods:
     - a finite difference method which replaces the cross section data and computes the difference in solution
     - an automatic differentiation method which uses jax's autodiff capabilities to compute the derivatives.

    The two methods are compared at some arbitrary cross section indices.
    
    '''
    dim, degree= _dim_degree_tested()
    dx = 1e-5
    mgop, source_eg0, solution_domain = setup_mgop(dim, degree)    
    sgop = mgop.to_single_group_operator(0)

    bicgstab = lx.BiCGStab(1e-10, 1e-10, max_steps = 100)
    
    sum_grad_f = eqx.filter_jit(jax.grad(sum_f, argnums = (2,)))

    all_derivatives = sum_grad_f(sgop, source_eg0, SingleGroupOperatorSigmaAllParametrised.theta_base(sgop), bicgstab)[0]

    def index_all_derivatives(xs_data_indices, all_derivatives):
        if xs_data_indices[0] == 'total':
            return all_derivatives.total[xs_data_indices[1]]
        elif xs_data_indices[0] == 'scattering':
            return all_derivatives.scattering[xs_data_indices[1]]
        else:
            raise ValueError("Invalid xs_data_indices")
        
    for i in all_testing_derivatives():
        xs_data_indices = i
        print(f"Testing finite difference vs. automatic differentiation for {xs_data_indices[0]} xs at index {xs_data_indices[1]}")
        print("Automatic differentiation:", index_all_derivatives(xs_data_indices, all_derivatives))
        print("Finite difference:        ", dx_mgop_mod_theta(dim, degree, xs_data_indices, dx, bicgstab) )
        onp.testing.assert_allclose(
            index_all_derivatives(xs_data_indices, all_derivatives),
            dx_mgop_mod_theta(dim, degree, xs_data_indices, dx, bicgstab),
            rtol=1e-4,
            atol=1e-4
        )
        


