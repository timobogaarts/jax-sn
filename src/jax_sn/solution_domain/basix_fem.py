import basix 
import numpy as onp
from .base_fem import Element
from typing import Tuple, List
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
    face_vertices   = basix.cell.topology(element.cell_type)[-2][face_idx] # -1 = cell, -2 = faces    
    vi              = basix.cell.geometry(element.cell_type)[face_vertices]
    return vi[0] +  onp.sum( (vi[None, 1:face_points.shape[1]+1,:] - vi[0][None, None, : ]) * face_points[:, :,None ] , axis=1)

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
    points, weights = basix.make_quadrature(basix.cell.subentity_types(element.cell_type)[-2][face_idx], element.degree * 2) # -1 = cell, -2 = faces    
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


class BasixLagrangianSimplex(Element):
    
    def __init__(self, degree : int, dimension : int):
        from basix import ElementFamily, LagrangeVariant, CellType
        type = [CellType.interval, CellType.triangle, CellType.tetrahedron][dimension -1]
        self.element = basix.create_element(ElementFamily.P, type, degree= degree, lagrange_variant= LagrangeVariant.gll_warped, discontinuous=True)
        self.dimension = dimension
        self.non_discontinuous_element = basix.create_element(ElementFamily.P, type, degree= degree, lagrange_variant= LagrangeVariant.gll_warped, discontinuous=False)

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
    
    def n_faces(self) -> int:
        return self.dimension + 1
    
    def tabulate_basis_functions_first_derivative(self, points):                
        assert points.shape[0] > 0        
        points = onp.atleast_2d(points)
        assert points.shape[1] == self.dimension, f"Points should be of shape (n_points, {self.dimension}) but is {points.shape}"
        return self.element.tabulate(1, points)[..., 0]
        
