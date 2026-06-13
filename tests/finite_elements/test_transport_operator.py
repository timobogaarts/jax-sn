import jax 
import jax.numpy as jnp
from jax_sn.operators.transport_operator import transport_inverse_sweep_complete_vmap
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
    
    
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(element, degree, _vertices_conn_function, _angle_function, _expected_values_function, _source_function, _cross_section_function)

    mass_operator = jax_sn.operators.mass_operator.MassOperator(femmatrices, precomputed_data)

    source_mass_op = mass_operator.mv(source.source_values)

    mapped_array = cross_sections.map_to_full_array(0)

    sweep_array = transport_inverse_sweep_complete_vmap(
        domain, 
        femmatrices,         
        precomputed_data,
        mapped_array, 
        toposort,                
        source_mass_op
    )
    
    onp.testing.assert_allclose(sweep_array, expected_values, atol=1e-12)

    
def _setup_domain(element, degree, _vertices_conn_function):
    vertices, connectivity = _vertices_conn_function(element, degree)
    lagrangian_element     = jax_sn.solution_domain.basix_fem.BasixLagrangianSimplex(degree=degree, dimension=element)    
    domain           = jax_sn.domain.GeometricDomain.from_mesh((jnp.array(vertices)[..., :element], jnp.array(connectivity)), jnp.array(lagrangian_element.face_template))
    return lagrangian_element, domain

def _setup_all(element, degree, _vertices_conn_function, _angle_function, _expected_values_function, _source_function, _cross_section_function):
    lagrangian_element, domain = _setup_domain(element, degree, _vertices_conn_function)
    angle                      = _angle_function()    
    toposort         = jax_sn.operators.transport_operator.BasicSweepData.from_domain_angle(domain, angle)    
    femmatrices      = jax_sn.solution_domain.FEMMatrices.from_element(lagrangian_element)    
    precomputed_data = lagrangian_element.precompute_geometric_data(domain)    
    sweep_array      = jnp.zeros((domain.n_elements, lagrangian_element.n_basis))
    cross_sections   = _cross_section_function(element, degree)
    source           = _source_function(element, degree)
    expected_values  = _expected_values_function(element, degree)
    
    return lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values


#====================================================================================================================================================================================
#                                      StellBlanket functions for testing & comparison [ugly but works]
#====================================================================================================================================================================================
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
    size_other_dim = 7.54
    resolutions_other_dim = 3
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
    n_elements = nte.geometric_domain.connectivity.shape[0]
    cross_sections = jnp.array([nte.geometric_domain[i].Material().Cross_Section().Value_Total(nte.energy_set[0]) for i in range(nte.geometric_domain.connectivity.shape[0])])

    cross_sections = cross_sections.reshape((n_elements, 1)) # [n_elements, n_energy_groups]
    
    vals, inverse = jnp.unique(cross_sections[:,0], return_inverse=True)
    
    scattering = jnp.zeros((vals.shape[0], 1, 1,1)) # Scattering is irrelevant for transport operator test [n_elements, n_L, n_energy_groups, n_energy_groups]
    total      = jnp.array(vals).reshape((vals.shape[0], 1))
    xsdata =  jax_sn.domain.cross_sections.CrossSectionData(total, scattering)
    xsdata.assert_valid_batched()
    
    return jax_sn.domain.geometric_domain.CrossSectionsIndexed(inverse, xsdata)

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

def _int_setup_solution_domain_data_and_toposort_sb(request):
    # If there is a better testing / ground truth, just replace this function.
    dimension = request.param
    degree = 2
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
    mapped_xs = cross_sections.map_to_full_array(0)
    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(lagrangian_element, jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections))

    
    quad_set = jax_sn.quadrature_set.create_tn_quadrature_set(2)
    toposorts = [jax_sn.operators.transport_operator.BasicSweepData.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)

    source_angle = jax.tree.map(lambda ts : jnp.repeat(ts, total_toposorts.angle.shape[0], axis=0).reshape((total_toposorts.angle.shape[0],)+ts.shape), source)
    return solution_domain, toposort, mapped_xs, source, expected_values, quad_set, total_toposorts, source_angle


@pytest.fixture(params=[1,2,3], scope='session')
def setup_solution_domain_data_and_toposort_sb(request):
    return _int_setup_solution_domain_data_and_toposort_sb(request)

#====================================================================================================================================================================================
    
@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_sweep_complete_vmap(dimension):
    # This tests the basic complete vmap sweep against StellBlanket results
    # Then, the other implementations are tested against the complete vmap sweep (transitive testing)
    degree = _n_degree_old_code(dimension)
    for i in range(degree + 1):
        check_transport_inverse_sweep_complete_vmap(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_local_real_inversion(dimension):
    # Tests BVH containing & local/real coordinate conversion for simplexes
    lagrangian_element, domain = _setup_domain(dimension, 1, _verts_conn_from_sb)
    aabb_min = jnp.min(domain.vertices, axis=0)
    aabb_max = jnp.max(domain.vertices, axis=0)
    points = jnp.linspace(aabb_min, aabb_max, 37)
    element_indices = domain.containing_cell(points)
    precomputed_data= lagrangian_element.precompute_geometric_data(domain)
    local_coords = lagrangian_element.local_coordinates(points, element_indices, domain, precomputed_data)
    
    atol = 1e-12
    onp.testing.assert_array_less(0 - atol, local_coords) # all coords >= 0 , simplex requirement 
    onp.testing.assert_array_less(local_coords.sum(axis=-1), 1.0 + atol) # sum of coords <= 1, simplex requirement
    
    real_coords  = lagrangian_element.real_coordinates(local_coords, element_indices, domain, precomputed_data)
    onp.testing.assert_allclose(points, real_coords, atol=1e-12)



#============================================================
# Transport operator tests
#===========================================================
# Including adjoint property tests & forward-backward consistency tests
# Optionally includes the Pallas implementations, --runpallas

def _all_transport_operators():
    return {jax_sn.operators.transport_operator.TransportOperatorVmap : {}, jax_sn.operators.transport_operator.TransportOperatorBatched : {'batch_size' : 5} }

def _all_transport_operators_pallas():
    return {jax_sn.operators.transport_operator.TransportOperatorVmap : {}, jax_sn.operators.transport_operator.TransportOperatorBatched : {'batch_size' : 5} , jax_sn.operators.transport_operator.TransportOperatorVmapPallas : {}, jax_sn.operators.transport_operator.TransportOperatorBatchedPallas : {'batch_size' : 5} }


@pytest.fixture(params=[1,2,3], scope='session')
def setup_all_transport_operators(request):        
    solution_domain, toposort, mapped_xs, source, expected_values, quad_set, total_toposorts, source_angle = _int_setup_solution_domain_data_and_toposort_sb(request)
    
    desired_ops = _all_transport_operators_pallas() if ( request.config.getoption("--runpallas")) else _all_transport_operators()
    operators = []
    for operator_class, operator_kwargs in desired_ops.items():        
        operators.append(operator_class(solution_domain.solution_domain_data, toposort, 0, **operator_kwargs))

    for operator_class, operator_kwargs in desired_ops.items():        
        operators.append(jax_sn.operators.transport_operator.transport_operator_base.TransportOperatorMultipleAnglesVmap(operator_class(solution_domain.solution_domain_data, total_toposorts, 0, **operator_kwargs)))
    
    return operators

@pytest.mark.slow
def test_transport_operator_adjoint(setup_all_transport_operators):
    # Test the adjoint property of the transport operator and its inverse:
    # <Tx, y> = <x, T'y>
    # <Tinvx, y> = <x, T'invy>
    onp.random.seed(12122025)
    operators = setup_all_transport_operators

    for operator in operators:
        x = jnp.array(onp.random.rand(*operator.in_structure().shape))
        y = jnp.array(onp.random.rand(*operator.in_structure().shape))
        Tx = operator.mv(x)
        Tty = operator.T.mv(y)    
        lhs = onp.einsum('...ij,...ij->...', Tx, y) # for batched angles the ... takes care of the angle dimension. Note that because each angle is independent, its adjoint property holds separately.
        rhs = onp.einsum('...ij,...ij->...', x, Tty)
        onp.testing.assert_allclose(lhs, rhs, atol=1e-12)

        Tinvx = operator.inv(x)
        Tinvty = operator.T.inv(y)  
        lhs_inv = onp.einsum('...ij,...ij->...', Tinvx, y)
        rhs_inv = onp.einsum('...ij,...ij->...', x, Tinvty)
        onp.testing.assert_allclose(lhs_inv, rhs_inv, atol=1e-12)

@pytest.mark.slow 
def test_transport_operator_forward_backward(setup_all_transport_operators):
    onp.random.seed(12122025)

    operators = setup_all_transport_operators

    for operator in operators:       
        x = jnp.array(onp.random.rand(*operator.in_structure().shape))        
        Tx = operator.mv(x)
        TinvTx = operator.inv(Tx)
        onp.testing.assert_allclose(x, TinvTx, atol=1e-12)
        Tty = operator.T.mv(x)    
        TinvTty = operator.T.inv(Tty)
        onp.testing.assert_allclose(x, TinvTty, atol=1e-12)


@pytest.mark.slow 
def test_transport_operator_consistency(setup_all_transport_operators):
    onp.random.seed(12122025)

    operators = setup_all_transport_operators
    n_basis   = operators[0].solution_domain_data.fem_matrices.n_basis
    n_elements = operators[0].solution_domain_data.domain.n_elements

    x_non_batch     = jnp.array(onp.random.rand(n_elements, n_basis))

    tx_gt = operators[0].mv(x_non_batch)
    ttx_gt = operators[0].T.mv(x_non_batch)
    tinv_gt = operators[0].inv(x_non_batch)
    tinvt_gt = operators[0].T.inv(x_non_batch)

    for operator_non_batched in operators[:len(operators)//2]:        
        tx = operator_non_batched.mv(x_non_batch)
        ttx = operator_non_batched.T.mv(x_non_batch)
        tinv = operator_non_batched.inv(x_non_batch)
        tinvt = operator_non_batched.T.inv(x_non_batch)  

        onp.testing.assert_allclose(tx_gt, tx, atol=1e-12)
        onp.testing.assert_allclose(ttx_gt, ttx, atol=1e-12)
        onp.testing.assert_allclose(tinv_gt, tinv, atol=1e-12)
        onp.testing.assert_allclose(tinvt_gt, tinvt, atol=1e-12)

    x_batch = jnp.array(onp.random.rand(*(operators[-1].in_structure().shape)))

    tx_gt = operators[len(operators)//2].mv(x_batch)
    ttx_gt = operators[len(operators)//2].T.mv(x_batch)
    tinv_gt = operators[len(operators)//2].inv(x_batch)
    tinvt_gt = operators[len(operators)//2].T.inv(x_batch)

    for operator_batched in operators[len(operators)//2:]:
        tx = operator_batched.mv(x_batch)
        ttx = operator_batched.T.mv(x_batch)
        tinv = operator_batched.inv(x_batch)
        tinvt = operator_batched.T.inv(x_batch)  

        onp.testing.assert_allclose(tx_gt, tx, atol=1e-12)
        onp.testing.assert_allclose(ttx_gt, ttx, atol=1e-12)
        onp.testing.assert_allclose(tinv_gt, tinv, atol=1e-12)
        onp.testing.assert_allclose(tinvt_gt, tinvt, atol=1e-12)        

@pytest.mark.slow 
def test_transport_operator_batching(setup_all_transport_operators):
    onp.random.seed(12122025)

    operators = setup_all_transport_operators
    
    x_batch         = jnp.array(onp.random.rand(*(operators[-1].in_structure().shape)))

    tx_singles = []    
    ttx_singles = []
    tinv_singles = []    
    tinvt_singles = []
    for i in range(x_batch.shape[0]):
        transport_operator_single = jax_sn.operators.transport_operator.TransportOperatorVmap(operators[0].solution_domain_data, operators[-1].transport_operator.sweep_data[i], 0)
        tx_singles.append(transport_operator_single.mv(x_batch[i]))
        ttx_singles.append(transport_operator_single.T.mv(x_batch[i]))
        tinv_singles.append(transport_operator_single.inv(x_batch[i]))
        tinvt_singles.append(transport_operator_single.T.inv(x_batch[i]))
        

    tx_single = jnp.stack(tx_singles, axis=0)
    ttx_single = jnp.stack(ttx_singles, axis=0)
    tinv_single = jnp.stack(tinv_singles, axis=0)
    tinvt_single = jnp.stack(tinvt_singles, axis=0)

    for operator_batched in operators[len(operators)//2:]:        
        tx_batched = operator_batched.mv(x_batch)
        ttx_batched = operator_batched.T.mv(x_batch)
        tinv_batched = operator_batched.inv(x_batch)
        tinvt_batched = operator_batched.T.inv(x_batch)  

        onp.testing.assert_allclose(tx_single, tx_batched, atol=1e-12)
        onp.testing.assert_allclose(ttx_single, ttx_batched, atol=1e-12)
        onp.testing.assert_allclose(tinv_single, tinv_batched, atol=1e-12)
        onp.testing.assert_allclose(tinvt_single, tinvt_batched, atol=1e-12)
