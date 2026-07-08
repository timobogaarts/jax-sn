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

