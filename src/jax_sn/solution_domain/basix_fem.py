import basix 
import numpy as onp
from .base_fem import Element
from typing import Tuple, List
import math
from jax_sn.tools.permutation_tools import _get_all_permutations
from .base_fem import PrecomputedGeometricData
from jax_sn.domain.geometric_domain import GeometricDomain
import jax.numpy as jnp
import jax

def _map_simplex_point_to_different_permutation(permutation, coordinates):
    '''
    Map a point in the reference simplex to a different permutation of the simplex.

    If the permutation is a single point (i.e. the simplex has dimension 0), the coordinates are returned unchanged.

    Parameters
    ----------  
    permutation : onp.ndarray [n_vertices = n_dim + 1]
        The permutation to map to.
    coordinates : onp.ndarray [n_points, dim]
        The coordinates in the reference simplex.
    '''

    if permutation.shape[0] == 1:
        return coordinates

    assert coordinates.shape[-1] + 1 == permutation.shape[0], "The permutation must have n_dim + 1 vertices."
    
    barycentric_coordinates = onp.concatenate([1.0 - onp.sum(coordinates, axis=-1, keepdims=True), coordinates], axis=-1)    
    return barycentric_coordinates[..., permutation][..., 1:]

def _map_face_points_to_reference_cell(element : basix.finite_element.FiniteElement, face_idx : int, face_points : onp.ndarray) -> onp.ndarray:
    '''        
    Map points from reference face to reference cell
    This is done by the standard affine mapping, also for non-simplex cells.
    This is defined as follows:
    A d-dimensional facet with vertices {0, ..., k}
      that have coordinates in the reference cell v0, v1, ..., vk (these are d+1 points in for a d-dimensional facet)
    A point with reference coordinates (x1, x2, ..., xd) on the facet
    v0 + sum_{i=1}^{d} (vi - v0) * xi
    
    This always works for simplices.
    For non-simplices (i.e. a quadrilateral face in a hexahedron), this works as long as v_d is 
    v_d = v0 + sum_{i=1}^{d-1} (vi - v0)
        Since this functions only accepts basix elements, we can assume that this is valid (as this is how basix defines it)
    See also ffcx/ffcx/element_interface.py map_facet_points.

    Parameters
    ----------
    element : basix.finite_element.FiniteElement
        The basix finite element
    face_idx : int
        The face index
    face_points : onp.ndarray [n_points, face_dim]
        The points on the face in reference coordinates
    Returns
    -------
    onp.ndarray [num_points, cell_dim]
        The points mapped to the reference cell 
    '''
    face_vertices   = basix.cell.topology(element.cell_type)#[-2][face_idx] # -1 = cell, -2 = faces    
    
    face_vertices   = basix.cell.topology(element.cell_type)[-2][face_idx]  # -1 = cell, -2 = faces            
    vi              = basix.cell.geometry(element.cell_type)[face_vertices]    
    return vi[0] +  onp.sum( (vi[None, 1:face_points.shape[1]+1,:] - vi[0][None, None, : ]) * face_points[:, :,None ] , axis=1) 

def _create_face_quadrature_on_face(element : basix.finite_element.FiniteElement, face_idx : int) -> onp.ndarray:
    if element.cell_type == basix.CellType.interval:
        points = onp.array([[0.0]]) # basix cannot make a 'quadrature' for a 0D face. Since there is only one point, the only choice is a weight of 1.0 with a point at (arbitrary) 0.0.
        weights = onp.array([1.0])        
    else:
        points, weights = basix.make_quadrature(basix.cell.subentity_types(element.cell_type)[-2][face_idx], element.degree * 2) # -1 = cell, -2 = faces    
    return points, weights

def _create_face_quadrature(element : basix.finite_element.FiniteElement, face_idx : int) -> onp.ndarray:
    '''    
    Create a face quadrature for a particular element face.

    Parameters:
    ----------
    element : basix.finite_element.FiniteElement
        The basix finite element
    face_idx : int
        The face index
    Returns:
    -------
    onp.ndarray [n_points, cell_dim], onp.ndarray [n_points]
        The quadrature points mapped to the reference cell and the weights    
    '''
    points, weights = _create_face_quadrature_on_face( element, face_idx)
    return _map_face_points_to_reference_cell(element, face_idx, points), weights        



def _create_total_face_quadrature(element : basix.finite_element.FiniteElement) -> List[Tuple[onp.ndarray, onp.ndarray]]:
    '''
    Create face quadratures for all faces of the element.
    Returns a list of tuples (points, weights) for each face.
    Supports all types of basix elements; also those with faces of different types.

    Parameters
    ----------
    element : basix.finite_element.FiniteElement
        The basix finite element
    Returns
    -------
    List[Tuple[onp.ndarray, onp.ndarray]]
        A list of tuples (points, weights) for each face.
    '''
    n_faces = len(basix.cell.topology(element.cell_type)[-2])    
    return [_create_face_quadrature(element, face_idx) for face_idx in range(n_faces)]

def create_all_combined_face_quadratures(element : basix.finite_element.FiniteElement):
    '''
    Create all combined face quadratures for an element, i.e., for each face, for each permutation of the face vertices, and for each connected face.
    Returns a list of tuples (points_own, points_conn, weights) where points_own are the quadrature points on the own face, points_conn are the quadrature points on the connected face (after applying the permutation), and weights are the quadrature weights.

    Parameters
    ----------
    element : basix.finite_element.FiniteElement
        The finite element for which to create the combined face quadratures.   
    Returns
    -------
    list of tuples
        A list of tuples (points_own, points_conn, weights) for each combination of face, permutation, and connected face. Ordering is 
    '''
    n_face = len(basix.cell.topology(element.cell_type)[-2])    
    n_vertices_per_face = len(basix.cell.topology(element.cell_type)[-2][0])
    all_permutations = _get_all_permutations(n_vertices_per_face)
    all_face_quadratures = []
    
    for face_idx in range(n_face):
        points, weights = _create_face_quadrature_on_face(element, face_idx)
        points_own      = _map_face_points_to_reference_cell(element, face_idx, points)

        for permutation in all_permutations:
            permuted_points = _map_simplex_point_to_different_permutation(permutation, points)

            for face_idx_conn in range(n_face):
                points_conn = _map_face_points_to_reference_cell(element, face_idx_conn, permuted_points)
                all_face_quadratures.append( (points_own, points_conn, weights) )

    return all_face_quadratures


class BasixLagrangianSimplex(Element):
    '''
    Finite element based on Basix Lagrangian elements on simplices (interval, triangle, tetrahedron).

    This has a linear jacobian mapping between reference and real coordinates. Basix is used for the basis functions and quadrature generation.
    These do not change during the simulation, so non-jittable Basix functions can be used during setup and precomputation.

    '''
    
    def __init__(self, degree : int, dimension : int):
        from basix import ElementFamily, LagrangeVariant, CellType
        type = [CellType.interval, CellType.triangle, CellType.tetrahedron][dimension -1]
        self.element = basix.create_element(ElementFamily.P, type, degree= degree, lagrange_variant= LagrangeVariant.gll_warped, discontinuous=True)
        self.dimension = dimension

        if degree > 0:
            self.non_discontinuous_element = basix.create_element(ElementFamily.P, type, degree= degree, lagrange_variant= LagrangeVariant.gll_warped, discontinuous=False)
        else:
            self.non_discontinuous_element = self.element

    @property
    def volume_quadrature(self):
        return basix.make_quadrature(self.element.cell_type, 2 * self.element.degree)
    
    @property 
    def has_linear_jacobian(self) -> bool:
        return True
    
    @property
    def face_quadratures(self):                
        return _create_total_face_quadrature(self.element)
    
    @property 
    def all_connection_face_quadratures(self) -> List[Tuple[onp.ndarray, onp.ndarray, onp.ndarray]]:
        return create_all_combined_face_quadratures(self.element)

    
    @property    
    def n_faces(self) -> int:
        return self.dimension + 1
    
    @property     
    def n_basis(self) -> int:
        return self.element.dim
    
    @property    
    def face_template(self) -> List[Tuple]:
        return jnp.array(basix.cell.topology(self.element.cell_type)[-2])
        
    def n_vertices_per_face(self, face_idx) -> int:
        return self.dimension
    
    def tabulate_basis_functions_first_derivative(self, points):                
        assert points.shape[0] > 0        
        points = onp.atleast_2d(points)
        assert points.shape[1] == self.dimension, f"Points should be of shape (n_points, {self.dimension}) but is {points.shape}"

        return self.element.tabulate(1, points)[..., 0]
    
    def compatible(self, geometric_domain : GeometricDomain) -> bool:
        if not (geometric_domain.connectivity.shape[-1] == self.dimension + 1):
            return False 
        
        face_template_here = onp.array(self.face_template)
        
        face_template_geom = onp.array(geometric_domain.face_template)
        if not onp.array_equal(face_template_here, face_template_geom):
            return False
        return True
            
    
    def precompute_geometric_data(self, geometric_domain):
        self.assert_compatible(geometric_domain)
        return _precompute_jacobians_simplex(geometric_domain)
    
    def local_coordinates(self, real_coordinates : jnp.ndarray, element_indices : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData):
        assert real_coordinates.shape[-1] == self.dimension, f"Real coordinates should be of shape (..., {self.dimension}) but is {real_coordinates.shape}"
        assert element_indices.shape == real_coordinates.shape[:-1], f"Element indices should be of shape {real_coordinates.shape[:-1]} but is {element_indices.shape}"
        return _local_coordinate_simplex_vec(real_coordinates, element_indices, geometric_domain, precomputed_geometric_data)

    def real_coordinates(self, local_coordinates : jnp.ndarray, element_indices : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData):
        assert local_coordinates.shape[-1] == self.dimension, f"Real coordinates should be of shape (..., {self.dimension}) but is {local_coordinates.shape}"
        assert element_indices.shape == local_coordinates.shape[:-1], f"Element indices should be of shape {local_coordinates.shape[:-1]} but is {element_indices.shape}"
        return _real_coordinate_simplex_vec(local_coordinates, element_indices, geometric_domain, precomputed_geometric_data)
    
    
    def volume_quadrature_order(self, n_deg : int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return basix.make_quadrature(self.element.cell_type, n_deg)
        


def _local_coordinate_simplex(real_coordinate : jnp.ndarray, element_index : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData):
    '''
    Compute the local coordinate of a point in a simplex element.

    Uses the standard affine mapping from reference simplex to physical coordinates:
            x = v0 + J * xi = v0 + sum_{i=1}^{d} (vi - v0) * xi
    where J is the Jacobian matrix with columns (vi - v0) for i=1..d
    Thus, the local coordinate is given by:
            xi = J_inv * (x - v0)
    This inverse  jacobian is part of the precomputed geometric data.

    For a vectorized version, use _local_coordinate_simplex_vec.

    Parameters:
    -----------
    real_coordinate : jnp.ndarray [n_dim]
        The real coordinate of the point.
    element_index : int
        The index of the element containing the point.
    geometric_domain : GeometricDomain
        The geometric domain.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data.
    Returns:
    --------
    jnp.ndarray [n_dim]
        The local coordinate of the point in the reference simplex.
    '''
    connectivity_indices = geometric_domain.connectivity[element_index]
    element_vertices     = geometric_domain.vertices[connectivity_indices]        # [n_vertices, n_dim]
    J_inv = precomputed_geometric_data.vol_jacobians_inv[element_index]           # [n_dim, n_dim]
    l0 = (real_coordinate - element_vertices[0, :])                               # [n_dim]
    return J_inv @ l0

_local_coordinate_simplex_vec = jax.jit(jnp.vectorize(_local_coordinate_simplex, excluded=(2,3), signature=('(d),()->(d)')))
        
def _real_coordinate_simplex(local_coordinate : jnp.ndarray, element_index : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData):
    '''
    Compute the real coordinate of a point in a simplex element.

    Uses the standard affine mapping from reference simplex to physical coordinates:
            x = v0 + J * xi = v0 + sum_{i=1}^{d} (vi - v0) * xi
    where J is the Jacobian matrix with columns (vi - v0) for i=1..d
    Thus, the real coordinate is given by:
            x = v0 + J * xi
    This jacobian is part of the precomputed geometric data.    

    For a vectorized version, use _real_coordinate_simplex_vec.

    Parameters:
    ----------- 
    local_coordinate : jnp.ndarray [n_dim]
        The local coordinate of the point in the reference simplex.
    element_index : int
        The index of the element containing the point.
    geometric_domain : GeometricDomain
        The geometric domain.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data.
    Returns:
    --------
    jnp.ndarray [n_dim]
        The real coordinate of the point.
    '''

    connectivity_indices = geometric_domain.connectivity[element_index]
    element_vertices     = geometric_domain.vertices[connectivity_indices]  # [n_vertices, n_dim]
    J                    = precomputed_geometric_data.vol_jacobians[element_index] # [n_dim, n_dim]
    return element_vertices[0, :] + J @ local_coordinate

_real_coordinate_simplex_vec = jax.jit(jnp.vectorize(_real_coordinate_simplex, excluded=(2,3), signature=('(d),()->(d)')))

        
        
    
    
    
def _jacobian_simplex(vertices : onp.ndarray) -> onp.ndarray:
    r'''
    Compute the Jacobian matrix for a simplex defined by its vertices.
    
    From the standard simplex coordinate system to physical coordinates:
    v0 + sum_{i=1}^{d} (vi - v0) * xi
    The Jacobian is thus the matrix with columns (vi - v0) for i=1..d:

    J_ij = d x_i / d xi_j = (vj - v0)_i

    For a simplex of dimension 0 (a point), we return a 1x1 matrix with value 1.

    Parameters:
    -----------
    vertices : jnp.ndarray [..., n_vertices, n_dim]
        Array of simplex vertices. n_vertices does not have to be n_dim + 1 (embedded simplices are allowed)

    Returns:
    --------
    jnp.ndarray [..., n_dim, n_vertices - 1]
        Jacobian matrices for each simplex: J_ij = d x_i / d xi_j
    '''

    n_vertices = vertices.shape[-2]
    if n_vertices == 1:        
        return jnp.ones(vertices.shape[:-2] + (1,1))    
    else:        
        v0 = vertices[..., 0:1, :]
        diffs = vertices[..., 1:, :] - v0
        return jnp.moveaxis(diffs, -1, -2)  # Move the last axis (n_dim) to second last position


def _precompute_jacobians_simplex(domain : GeometricDomain) -> PrecomputedGeometricData:
    '''
    Precomputes the Jacobian matrices, their inverses, and determinants for a simplex domain.

    Parameters
    ----------
    domain : GeometricDomain
        The geometric domain consisting of simplex elements.    
    Returns
    -------
    PrecomputedGeometricData
        The precomputed geometric data including Jacobians, their inverses, and determinants.
    '''
    vertices           = domain.vertices[domain.connectivity]
    volume_J_total     = _jacobian_simplex(vertices)
    volume_J_det_total = jnp.abs(jnp.linalg.det(volume_J_total))
    volume_J_inv_total = jnp.linalg.inv(volume_J_total)    
    face_conn          = domain.connectivity[:, domain.face_template]    # [n_element, n_faces, n_face_vertices]    
    face_vertices      =  domain.vertices[face_conn]
    face_jacobians     = _jacobian_simplex(face_vertices)
    face_jacobians_det = jnp.sqrt(jnp.abs(jnp.linalg.det(onp.einsum('...ji,...jk->...ik', face_jacobians, face_jacobians))))    
    return PrecomputedGeometricData(volume_J_total, volume_J_inv_total, volume_J_det_total, face_jacobians_det)
    
