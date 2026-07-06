import jax 
import jax.numpy as jnp
from jax_sn.operators.transport_operator.implementations.transport_operator_jax import transport_inverse_sweep_complete_vmap
import jax_sn

import numpy as onp
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
        angle,
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
#   Reference-data providers  [load frozen reference arrays; see tests/reference_data/]
#====================================================================================================================================================================================
# These load the frozen mesh / source / cross sections / single-angle sweep reference and rebuild
# the jax-sn wrapper objects, so the suite needs no external solver. The moment test imports the
# mesh/source/cross-section providers from here too.
from reference_data import load as _load_ref, key_transport, key_transport_angle

def _verts_conn_reference(dimension, degree):
    b = _load_ref(key_transport(dimension, degree))
    return jnp.array(b['verts']), jnp.array(b['conn'])

def _source_reference(dimension, degree):
    b = _load_ref(key_transport(dimension, degree))
    return jax_sn.source.Source(jnp.array(b['source']))

def _cross_sections_reference(dimension, degree):
    b = _load_ref(key_transport(dimension, degree))
    xsdata = jax_sn.domain.cross_sections.CrossSectionData(jnp.array(b['xs_total']), jnp.array(b['xs_scattering']))
    xsdata.assert_valid_batched()
    return jax_sn.domain.geometric_domain.CrossSectionsIndexed(jnp.array(b['xs_indices']), xsdata)

def _angle_function_tn():
    return jnp.array(_load_ref(key_transport_angle())['angle'])

def _max_test_degree(dimension):
    return [None, 3, 3, 2][dimension]

def _expected_values_reference(dimension, degree):
    return jnp.array(_load_ref(key_transport(dimension, degree))['expected'])

def _int_setup_solution_domain_data_and_toposort(request):
    # If there is a better testing / ground truth, just replace this function.
    dimension = request.param
    degree = 2
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_reference, _angle_function_tn, _expected_values_reference, _source_reference, _cross_sections_reference)
    mapped_xs = cross_sections.map_to_full_array(0)
    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(lagrangian_element, jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections))

    
    quad_set = jax_sn.quadrature_set.create_tn_quadrature_set(2)
    toposorts = [jax_sn.operators.transport_operator.BasicSweepData.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)

    n_angles = quad_set.angles.shape[0]
    source_angle = jax.tree.map(lambda ts : jnp.repeat(ts, n_angles, axis=0).reshape((n_angles,)+ts.shape), source)
    return solution_domain, angle, toposort, mapped_xs, source, expected_values, quad_set, total_toposorts, source_angle


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_sweep_complete_vmap(dimension):
    # This tests the basic complete vmap sweep against the frozen reference
    # Then, the other implementations are tested against the complete vmap sweep (transitive testing)
    degree = _max_test_degree(dimension)
    check_transport_inverse_sweep_complete_vmap(dimension, degree, _verts_conn_reference, _angle_function_tn, _expected_values_reference, _source_reference, _cross_sections_reference)


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_local_real_inversion(dimension):
    # Tests BVH containing & local/real coordinate conversion for simplexes
    lagrangian_element, domain = _setup_domain(dimension, 1, _verts_conn_reference)
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

def _vmap_op(solution_domain_data, sweep_data, angle, energy_group):
    _to = jax_sn.operators.transport_operator
    return _to.TransportOperator(solution_domain_data, sweep_data, angle, energy_group, _to.VmapForward(), _to.VmapSweep())

def _batched_op(solution_domain_data, sweep_data, angle, energy_group, batch_size=None):
    _to = jax_sn.operators.transport_operator
    return _to.TransportOperator(solution_domain_data, sweep_data, angle, energy_group, _to.BatchedForward(batch_size), _to.BatchedSweep(batch_size))

def _vmap_pallas_op(solution_domain_data, sweep_data, angle, energy_group):
    _to = jax_sn.operators.transport_operator
    return _to.TransportOperator(solution_domain_data, sweep_data, angle, energy_group, _to.VmapForward(), _to.VmapPallasSweep())

def _batched_pallas_op(solution_domain_data, sweep_data, angle, energy_group, batch_size=None):
    _to = jax_sn.operators.transport_operator
    return _to.TransportOperator(solution_domain_data, sweep_data, angle, energy_group, _to.BatchedForward(batch_size), _to.BatchedPallasSweep(batch_size))

def _all_transport_operators():
    return {_vmap_op : {}, _batched_op : {'batch_size' : 5} }

def _all_transport_operators_pallas():
    return {_vmap_op : {}, _batched_op : {'batch_size' : 5} , _vmap_pallas_op : {}, _batched_pallas_op : {'batch_size' : 5} }


@pytest.fixture(params=[1,2,3], scope='session')
def setup_all_transport_operators(request):        
    solution_domain, angle, toposort, mapped_xs, source, expected_values, quad_set, total_toposorts, source_angle = _int_setup_solution_domain_data_and_toposort(request)

    desired_ops = _all_transport_operators_pallas() if ( request.config.getoption("--runpallas")) else _all_transport_operators()
    operators = []
    for operator_class, operator_kwargs in desired_ops.items():
        operators.append(operator_class(solution_domain.solution_domain_data, toposort, angle, 0, **operator_kwargs))

    for operator_class, operator_kwargs in desired_ops.items():
        operators.append(jax_sn.operators.transport_operator.TransportOperatorMultipleAngles(operator_class(solution_domain.solution_domain_data, total_toposorts, quad_set.angles, 0, **operator_kwargs), jax_sn.operators.transport_operator.VmapAngles()))
    
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
        transport_operator_single = _vmap_op(operators[0].solution_domain_data, operators[-1].transport_operator.sweep_data[i], operators[-1].transport_operator.angle[i], 0)
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
