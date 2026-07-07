'''Current basix-based FEM matrices vs. a frozen independent oracle.

The reference volume (aij/ii/bij) and face (cij) matrices were produced by a second,
independent FEM implementation (old_generate_basis.py) and frozen into
tests/reference_data/fem__*.npz. The generator is retired -- the frozen arrays are the
oracle now (see tests/reference_data/regenerate.py / __init__.py for provenance).
'''
import jax
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_sn.solution_domain.base_fem import compute_volume_matrices_linear_jacobian, _compute_face_outflow_matrices
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
