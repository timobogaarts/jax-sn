import jax 
import numpy as onp 
import jax.numpy as jnp
import pytest 
jax.config.update("jax_enable_x64", True)
from .old_generate_basis import Tetrahedron3D, Triangle2D, Line1D, create_permutation, get_factor_difference, get_jacobian_difference, get_area_difference
from jax_sn.solution_domain.base_fem import compute_volume_matrices_linear_jacobian
from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex

def _get_element_jsn(element_name: str, degree: int):
    if element_name == "tetrahedron":
        return BasixLagrangianSimplex(degree=degree, dimension=3)
    elif element_name == "triangle":
        return BasixLagrangianSimplex(degree=degree, dimension=2)
    elif element_name == "interval":
        return BasixLagrangianSimplex(degree=degree, dimension=1)
    else:
        raise ValueError(f"Unknown element name: {element_name}")
    
def _get_element_old(element_name: str, degree: int):
    if element_name == "tetrahedron":
        return Tetrahedron3D(degree)
    elif element_name == "triangle":
        return Triangle2D(degree)
    elif element_name == "interval":
        return Line1D(degree)
    else:
        raise ValueError(f"Unknown element name: {element_name}")
    
def n_degree_max():
    return 3

def _permute_volume_matrix(aij, permutation):
    return aij[permutation][:, permutation]

def _get_volume_matrix_from_old_generate_basis(element, degree):
    # We use the old code to generate the volume matrix and permute it to match the new ordering    
    # finally, the jacobians differ because in the old code, this is non normalised
    element_old = _get_element_old(element, degree=degree)
    permutation = create_permutation(element, element_old, _get_element_jsn(element, degree=degree))
    element_old.Calculate_Aij()
    aijold = onp.array(element_old.Aij, dtype=float)
    element_old.Calculate_Ii()
    element_old.Calculate_Bij()
    bijold = onp.array(element_old.mathcal_Bija, dtype=float)        
    
    iiold = onp.array(element_old.Ii, dtype=float)[:,0][permutation] * get_factor_difference(element)
    return aijold[permutation][:, permutation] * get_factor_difference(element), \
           iiold,\
           bijold[:, permutation][..., permutation] * get_factor_difference(element) * get_jacobian_difference(element)

def create_permutation_dimension(dimension, degree):
    element = [None, "interval", "triangle", "tetrahedron"][dimension]
    return create_permutation(element, _get_element_old(element, degree=degree), _get_element_jsn(element, degree=degree))

def _get_face_permutation(element):
    import basix
    element_old = _get_element_old(element, degree=1)
    element_new = _get_element_jsn(element, degree=1)

    n_faces     = len(element_old.surface_maps)    
    edge_permutation = []
    permutation_edges = create_permutation(element, element_old, element_new)
    for i in range(n_faces):
       surface_new = basix.cell.topology(element_new.element.cell_type)[-2][i]
       surface_maps = onp.array(element_old.surface_maps) # [i,j]  = [which old element faces i  contain which dofs j ]
       edge_permutation.append( onp.where( onp.all(onp.sort(surface_maps) == onp.sort(permutation_edges[surface_new]), axis=-1))[0][0])    
    return edge_permutation

def _get_face_matrices_from_old_generate_basis(element, degree):
    # We use the old code to generate the face matrix and permute it to match the new ordering
    # Furthermore, there is some difference between the coordinate definition between basix and the old code
    # finally, the jacobians differ because in the old code, this is nont normliased.
    element_old = _get_element_old(element, degree=degree)    
    permutation = create_permutation(element, element_old, _get_element_jsn(element, degree=degree))    
    element_old.Calculate_Cij()    
    cijold      = onp.array(element_old.baseCij, dtype=float)    
    N_dofs      = element_old.node_coords[0].shape[0]    
    n_faces     = len(element_old.surface_maps)    
    cij_e       = onp.zeros((n_faces, N_dofs, N_dofs))
    for e in range(n_faces):
        edge_i, edge_j = onp.meshgrid(element_old.surface_maps[e], element_old.surface_maps[e], indexing='ij')
        for i in range(edge_i.shape[0]):
            for j in range(edge_j.shape[1]):        
                cij_e[e, edge_i[i,j], edge_j[i,j]] = cijold[i,j]    
    cijexpected = cij_e[:, permutation][..., permutation] * get_area_difference(element)
    return [cijexpected[e] for e in _get_face_permutation(element)]

@pytest.mark.parametrize("element", ["interval", "triangle", "tetrahedron"])
def test_volume_matrix(element, get_other_function  = _get_volume_matrix_from_old_generate_basis ):    
    for n_degree in range(0, n_degree_max()+1):
        element_jsn = _get_element_jsn(element, degree=n_degree)
        volume_matrices = compute_volume_matrices_linear_jacobian(element_jsn)  #[n_basis, n_basis]
        volume_matrices_old = get_other_function(element, n_degree)
        
        onp.testing.assert_allclose(volume_matrices[0], volume_matrices_old[0], atol =1e-12, rtol=1e-12) 
        onp.testing.assert_allclose(volume_matrices[1], volume_matrices_old[1], atol =1e-12, rtol=1e-12)
        onp.testing.assert_allclose(volume_matrices[2], volume_matrices_old[2], atol =1e-12, rtol=1e-12)


@pytest.mark.parametrize("element", ["interval", "triangle", "tetrahedron"])
def test_area_matrix(element, get_other_function  = _get_face_matrices_from_old_generate_basis ):    
    from jax_sn.solution_domain.base_fem import _compute_face_outflow_matrices
    for n_degree in range(0, n_degree_max()+1):
        element_jsn = _get_element_jsn(element, degree=n_degree)
        cij_old     = get_other_function(element, n_degree)        #[n_faces, n_basis, n_basis]
        cij_new     = _compute_face_outflow_matrices(element_jsn)  #[n_faces, n_basis, n_basis]        
        onp.testing.assert_allclose(cij_old, cij_new, atol =1e-12, rtol=1e-12)


