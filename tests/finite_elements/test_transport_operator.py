import jax
import jax.numpy as jnp
import jax_sn

import numpy as onp
import pytest


def _setup_domain(element, degree, _vertices_conn_function):
    vertices, connectivity = _vertices_conn_function(element, degree)
    lagrangian_element     = jax_sn.solution_domain.basix_fem.BasixLagrangianSimplex(degree=degree, dimension=element)
    domain           = jax_sn.domain.GeometricDomain.from_mesh((jnp.array(vertices)[..., :element], jnp.array(connectivity)), jnp.array(lagrangian_element.face_template))
    return lagrangian_element, domain


#====================================================================================================================================================================================
#   Reference-data providers  [load frozen mesh; see tests/reference_data/]
#====================================================================================================================================================================================
# Rebuild the mesh from the frozen bundle for test_local_real_inversion (a filled simplex region
# where a linspace across the AABB stays inside the mesh -- the BB slab geometry doesn't qualify).
from reference_data import load as _load_ref, key_transport

def _verts_conn_reference(dimension, degree):
    b = _load_ref(key_transport(dimension, degree))
    return jnp.array(b['verts']), jnp.array(b['conn'])


def test_transport_sweep_pinned(bb_mgop):
    # Regression pin of the single-angle vmap sweep on the shared BB domain, frozen from the current
    # implementation (guards against silent numerical drift). External accuracy of this sweep is
    # anchored by test_bb, since VmapSweep is transport_inverse_sweep_complete_vmap.
    from reference_data import key_transport_sweep
    MGOP, _, _, bb_args = bb_mgop
    top0 = MGOP.transport_operator[0]                       # single-angle TransportOperator
    onp.random.seed(7072026)
    b = jnp.array(onp.random.rand(*top0.in_structure().shape))
    sweep = top0.inv(b)
    expected = _load_ref(key_transport_sweep(bb_args[0]))['sweep']
    onp.testing.assert_allclose(onp.array(sweep), expected, rtol=1e-10, atol=1e-12)


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


@pytest.fixture(scope='session')
def setup_all_transport_operators(bb_mgop, request):
    # These tests are algebraic (adjoint, forward-backward, backend consistency, batching) and
    # need no ground truth, so they reuse the shared BB domain/toposort/angles. Backend accuracy
    # is anchored externally elsewhere (test_sweep_complete_vmap / test_moment_operator / test_bb).
    MGOP                 = bb_mgop[0]
    solution_domain_data = MGOP.solution_domain_data
    top                  = MGOP.transport_operator
    single_sweep_data, single_angle = top[0].sweep_data, top[0].angle
    total_toposorts, angles         = top.sweep_data, top.angle

    desired_ops = _all_transport_operators_pallas() if request.config.getoption("--runpallas") else _all_transport_operators()
    operators = []
    for operator_class, operator_kwargs in desired_ops.items():
        operators.append(operator_class(solution_domain_data, single_sweep_data, single_angle, 0, **operator_kwargs))

    for operator_class, operator_kwargs in desired_ops.items():
        operators.append(jax_sn.operators.transport_operator.TransportOperatorMultipleAngles(operator_class(solution_domain_data, total_toposorts, angles, 0, **operator_kwargs), jax_sn.operators.transport_operator.VmapAngles()))

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
