import jax 
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp

import numpy as onp
from .test_transport_operator import _setup_domain, _verts_conn_reference, _source_reference, _cross_sections_reference
import jax_sn
import pytest

def _setup_all(element, degree, tn_order, _vertices_conn_function, _source_function, _cross_section_function):
    lagrangian_element, geometric_domain = _setup_domain(element, degree, _vertices_conn_function)
    cross_sections   = _cross_section_function(element, degree)
    
    domain = jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(geometric_domain, cross_sections)

    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(
        lagrangian_element, domain       
    )

    angles           = jax_sn.quadrature_set.tn.create_tn_quadrature_set(tn_order)
    toposort         = [jax_sn.operators.transport_operator.BasicSweepData.from_domain_angle(domain, angle)  for angle in angles.angles]  

    total_toposorts = jax.tree.map(lambda *xs : jnp.stack(xs, axis=0), *toposort)    
    source           = _source_function(element, degree)
    source_total     = jax.tree.map(lambda * xs : jnp.stack(xs), *[source for _ in toposort])

    sweep_array      = jnp.zeros((angles.n_angles, domain.n_elements, lagrangian_element.n_basis))
    
    return solution_domain, angles, total_toposorts,sweep_array, source_total

def _setup_all_reference(dimension, degree, tn_order):
    return _setup_all(dimension, degree, tn_order, _verts_conn_reference, _source_reference, _cross_sections_reference)

def _expected_value_reference(dimension, degree, tn_order, lmax):
    # Frozen moment reference (see tests/reference_data/_generate.py)
    from reference_data import load, key_moment
    return jnp.array(load(key_moment(dimension, degree, lmax, tn_order))['expected'])




def check_moment_operator(dimension, degree, tn_order, lmax, _verts_conn_function, _source_function, _cross_section_function, _expected_values_function):
    solution_domain, quadrature_set, total_toposorts, sweep_array, source_total = _setup_all(dimension, degree, tn_order, _verts_conn_function, _source_function, _cross_section_function)

    transport_operator = jax_sn.operators.transport_operator.TransportOperatorMultipleAngles(
        jax_sn.operators.transport_operator.TransportOperator(solution_domain.solution_domain_data, total_toposorts, quadrature_set.angles, 0, jax_sn.operators.transport_operator.BatchedForward(10), jax_sn.operators.transport_operator.BatchedSweep(10)), jax_sn.operators.transport_operator.VmapAngles())
    
    moment_operator = jax_sn.operators.moment_operator.MomentOperator.from_quadrature_set(quadrature_set, lmax, solution_domain.elements_basis_shape)

    mass_operator  = jax_sn.operators.mass_operator.MassOperator(solution_domain.fem_matrices, solution_domain.precomputed_geometric_data)
    source_x = mass_operator.mv(source_total.source_values) # this automatically batches
    psilm = moment_operator.mv(transport_operator.inv(source_x ))


    #psilm = moment_operator(sweep_angles)
    expected_values = _expected_values_function(dimension, degree, tn_order, lmax)

    onp.testing.assert_allclose(psilm, expected_values, rtol=1e-12, atol=1e-12)



@pytest.mark.slow
@pytest.mark.parametrize(("dimension", "degree", "lmax", "tn_order"),
                          [(1,5,3,3),
                           (2,3,5, 4),
                           (3,2,4, 3)])
def test_moment_operator(dimension, degree, lmax, tn_order):
    check_moment_operator(dimension, degree, tn_order, lmax, _verts_conn_reference, _source_reference, _cross_sections_reference, _expected_value_reference)

def test_moment_operator_transpose():
    
    quadrature_set = jax_sn.quadrature_set.create_tn_quadrature_set(3)    
    element_basis = jax.ShapeDtypeStruct(shape=(10, 15), dtype=quadrature_set.angles.dtype)
    moment_operator = jax_sn.operators.moment_operator.MomentOperator.from_quadrature_set(quadrature_set, 5, element_basis)
    from .operator_util import _check_adjoint_of_operator
    
    _check_adjoint_of_operator(moment_operator)

def test_discrete_operator_transpose():
    
    quadrature_set = jax_sn.quadrature_set.create_tn_quadrature_set(3)    
    element_basis = jax.ShapeDtypeStruct(shape=(10, 15), dtype=quadrature_set.angles.dtype)
    discrete_operator = jax_sn.operators.moment_operator.DiscreteOperator.from_quadrature_set(quadrature_set, 5, element_basis)
    from .operator_util import _check_adjoint_of_operator
    
    _check_adjoint_of_operator(discrete_operator)




def test_scatter_operator_transpose():
    onp.random.seed(5012026)
    quadrature_set = jax_sn.quadrature_set.create_tn_quadrature_set(3)    
    n_elem = 10

    xs_data_total = jnp.array(onp.random.rand(n_elem,2))
    xs_data_scat = jnp.array(onp.random.rand(n_elem, 16, 2,2))

    cross_section_data= jax_sn.domain.cross_sections.CrossSectionData(xs_data_total, xs_data_scat)
    cross_sections_indexed = jax_sn.domain.geometric_domain.CrossSectionsIndexed(jnp.arange(n_elem), cross_section_data)

    scatter_operator = jax_sn.operators.scatter_operator.ScatterOperator(cross_sections_indexed, jax.ShapeDtypeStruct((4,n_elem, 3), dtype = xs_data_total.dtype), energy_group_out=1, energy_group_in=0)
    from .operator_util import _check_adjoint_of_operator
    _check_adjoint_of_operator(scatter_operator)