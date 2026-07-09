'''Current basix-based FEM matrices vs. a frozen independent oracle.

Tests finite element matrices for n_degree_max orders for all simplex elements.
'''
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)

import jax_sn
from jax_sn.solution_domain.base_fem import compute_volume_matrices_linear_jacobian, _compute_face_outflow_matrices, _compute_face_inflow_matrices
from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex

from reference_data import load, key_fem

_DIMENSION = {"interval": 1, "triangle": 2, "tetrahedron": 3}


def _get_element_jsn(element_name: str, degree: int):
    return BasixLagrangianSimplex(degree=degree, dimension=_DIMENSION[element_name])


def n_degree_max():
    return 3


@pytest.mark.parametrize("element", ["interval", "triangle", "tetrahedron"])
def test_volume_matrix(element):
    for n_degree in range(0, n_degree_max() + 1):
        element_jsn = _get_element_jsn(element, n_degree)
        volume_matrices = compute_volume_matrices_linear_jacobian(element_jsn)  # (aij, ii, bij)
        ref = load(key_fem(element, n_degree))

        onp.testing.assert_allclose(volume_matrices[0], ref['aij'], atol=1e-12, rtol=1e-12)
        onp.testing.assert_allclose(volume_matrices[1], ref['ii'],  atol=1e-12, rtol=1e-12)
        onp.testing.assert_allclose(volume_matrices[2], ref['bij'], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("element", ["interval", "triangle", "tetrahedron"])
def test_area_matrix(element):
    for n_degree in range(0, n_degree_max() + 1):
        element_jsn = _get_element_jsn(element, n_degree)
        cij_new = _compute_face_outflow_matrices(element_jsn)  # [n_faces, n_basis, n_basis]
        ref = load(key_fem(element, n_degree))

        onp.testing.assert_allclose(cij_new, ref['cij'], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("element", ["interval", "triangle", "tetrahedron"])
def test_face_inflow_matrix(element):
    # Face inflow (connection) matrices over every (self-face, permutation, connected-face)
    # combination -- pinning the exact per-permutation values so an orientation/permutation
    # regression is caught (a row-sum check alone is permutation-invariant and would miss it).
    for n_degree in range(0, n_degree_max() + 1):
        element_jsn = _get_element_jsn(element, n_degree)
        dij_new = _compute_face_inflow_matrices(element_jsn)  # [n_faces*n_perms*n_faces, n_basis, n_basis]
        ref = load(key_fem(element, n_degree))

        onp.testing.assert_allclose(dij_new, ref['dij'], atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# BVH containment + local/real coordinate conversion for simplex elements. Rebuilds a mesh from the frozen
# reference bundle (a filled simplex region where a linspace across the AABB stays inside the mesh -- the
# BB slab geometry doesn't qualify). Moved here from the transport tests: it exercises element geometry,
# not an operator.
# ---------------------------------------------------------------------------------------------
def _verts_conn_reference(dimension, degree):
    from reference_data import load as _load_ref, key_transport
    b = _load_ref(key_transport(dimension, degree))
    return jnp.array(b['verts']), jnp.array(b['conn'])


def _setup_domain(element, degree):
    vertices, connectivity = _verts_conn_reference(element, degree)
    lagrangian_element = BasixLagrangianSimplex(degree=degree, dimension=element)
    domain = jax_sn.domain.GeometricDomain.from_mesh(
        (jnp.array(vertices)[..., :element], jnp.array(connectivity)), jnp.array(lagrangian_element.face_template))
    return lagrangian_element, domain


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_local_real_inversion(dimension):
    lagrangian_element, domain = _setup_domain(dimension, 1)
    aabb_min = jnp.min(domain.vertices, axis=0)
    aabb_max = jnp.max(domain.vertices, axis=0)
    points = jnp.linspace(aabb_min, aabb_max, 37)
    element_indices = domain.containing_cell(points)
    precomputed_data = lagrangian_element.precompute_geometric_data(domain)
    local_coords = lagrangian_element.local_coordinates(points, element_indices, domain, precomputed_data)

    atol = 1e-12
    onp.testing.assert_array_less(0 - atol, local_coords)              # all coords >= 0, simplex requirement
    onp.testing.assert_array_less(local_coords.sum(axis=-1), 1.0 + atol)  # sum of coords <= 1, simplex requirement

    real_coords = lagrangian_element.real_coordinates(local_coords, element_indices, domain, precomputed_data)
    onp.testing.assert_allclose(points, real_coords, atol=1e-12)
