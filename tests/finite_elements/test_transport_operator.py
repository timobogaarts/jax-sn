import jax 
import jax.numpy as jnp
from jax_sn.solution_domain.transport_operator import transport_inverse_sweep_complete_vmap
import jax_sn

import numpy as onp
from .test_finite_elements import create_permutation_dimension
import pytest
def check_transport_inverse_sweep_complete_vmap(element, degree, 
                                                _vertices_conn_function, 
                                                _angle_function,  
                                                _expected_values_function, 
                                                _source_function, 
                                                _cross_section_function):
    
    
    lagrangian_element, domain = _setup_domain(element, degree, _vertices_conn_function)
    angle                      = _angle_function()    
    toposort         = jax_sn.domain.geometric_domain.TopologicalSortedAngle.from_domain_angle(domain, angle)    
    femmatrices      = jax_sn.solution_domain.FEMMatrices.from_element(lagrangian_element)    
    precomputed_data = lagrangian_element.precompute_geometric_data(domain)    
    sweep_array      = jnp.zeros((domain.n_elements, lagrangian_element.n_basis))
    cross_sections   = _cross_section_function(element, degree)
    source           = _source_function(element, degree)
    

    sweep_array = transport_inverse_sweep_complete_vmap(
        femmatrices,
        domain,
        toposort,
        cross_sections, 
        precomputed_data,
        source,
        sweep_array)
    
    expected_values = _expected_values_function(element, degree)

    onp.testing.assert_allclose(sweep_array, expected_values, atol=1e-12)

    
def _setup_domain(element, degree, _vertices_conn_function):
    vertices, connectivity = _vertices_conn_function(element, degree)
    lagrangian_element     = jax_sn.solution_domain.basix_fem.BasixLagrangianSimplex(degree=degree, dimension=element)    
    domain           = jax_sn.domain.GeometricDomain.from_mesh((jnp.array(vertices)[..., :element], jnp.array(connectivity)), jnp.array(lagrangian_element.face_template))
    return lagrangian_element, domain


#===========================================================================================================================
#                                      StellBlanket comparison functions
#===========================================================================================================================
def _switch_element_type(dimension):
    import StellBlanket as SB
    if dimension == 1:
        return SB.Lines.Line_Element
    elif dimension == 2:
        return SB.Triangles.Triangle_Element
    elif dimension == 3:
        return SB.Tetrahedrons.Tetrahedron_Element
    else:
        raise ValueError("Dimension must be 1, 2 or 3.")
    
def _test_domain_from_sb(dimension, degree):
    import StellBlanket as SB
    material_0 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(0.673))
    material_1 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(2.5))
    region_sizes=  [0.8, 0.2]
    resolutions  = [4,3]
    size_other_dim = 0.754
    resolutions_other_dim = 2
    element = _switch_element_type(dimension)
    cartesian_domain = SB.Generic_Elements.Create_Slab(element, region_sizes, resolutions, [material_0, material_1], size_other_dim, resolutions_other_dim)    
    return cartesian_domain

def _arbitrary_source_values(n_elements):
    return onp.linspace(1.0, 10.0, n_elements)

def tn_order_index():
    return 4, 3

def _angle_function_indexed(tn_order, index):
    import StellBlanket as SB
    quad_set = SB.Generic_Elements.TN_Quadrature_Set(tn_order)
    return quad_set.Directions()[index:index+1,:]
    
def _matfree_nte_from_sb(dimension, degree, tn_order, angle_index):
    import StellBlanket as SB
    domain   = _test_domain_from_sb(dimension, degree)
    quad_set = SB.Generic_Elements.Quadrature_Set(onp.array(_angle_function_indexed(tn_order, angle_index)), onp.array([1.0]))

    energy_set = SB.Generic_Elements.Energy_Set([0,1])
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, domain, degree, 0)

    source_values = _arbitrary_source_values(domain.connectivity.shape[0])
    sources = [SB.Generic_Elements.Constant_Source(_switch_element_type(dimension), val, energy_set[0]) for val in source_values]
    total_source = SB.Generic_Elements.Per_Element_Source(sources, domain)
    return matfreente, total_source
    
def _verts_conn_from_sb(dimension, degree):
    import StellBlanket as SB
    domain   = _test_domain_from_sb(dimension, degree)
    return jnp.array(domain.vertices), jnp.array(domain.connectivity)

def _source_from_sb(dimension, degree):
    nte, total_source = _matfree_nte_from_sb(dimension, degree, *tn_order_index()) # angle doesn't matter for this
    source_values = _arbitrary_source_values(nte.geometric_domain.connectivity.shape[0])
    n_values = int(nte.solution_domain.total_size / nte.geometric_domain.connectivity.shape[0])
    return jax_sn.source.Source(jnp.stack([source_values] * n_values, axis=-1))

def _cross_sections_from_sb(dimension, degree):
    nte, total_source = _matfree_nte_from_sb(dimension, degree, *tn_order_index()) # angle doesn't matter for this
    cross_sections = jnp.array([nte.geometric_domain[i].Material().Cross_Section().Value_Total(nte.energy_set[0]) for i in range(nte.geometric_domain.connectivity.shape[0])])
    return jax_sn.domain.geometric_domain.CrossSectionData(cross_sections)

def _angle_function_tn():
    return _angle_function_indexed(*tn_order_index())[0]

def _n_degree_old_code(dimension):
    return [None, 3, 3, 2][dimension]


def _expected_values_from_sb(dimension, degree):
    import StellBlanket as SB
    nte, total_source = _matfree_nte_from_sb(dimension, degree, *tn_order_index())
    
    nte.simulation.Sweep(nte.energy_set[0], nte.quadrature_set[0], total_source)
    values = nte.global_solution.All_Solution_Values(nte.energy_set[0], nte.quadrature_set[0]).reshape(nte.geometric_domain.connectivity.shape[0], -1   )

    permutation = create_permutation_dimension(dimension, degree)
    
    return values[:, permutation]
    
@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_sweep_complete_vmap(dimension):
    degree = _n_degree_old_code(dimension)
    for i in range(degree + 1):
        check_transport_inverse_sweep_complete_vmap(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)

