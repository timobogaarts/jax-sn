import jax
import jax.numpy as jnp 
from dataclasses import dataclass
from jax_sn.tools.mesh_tools import compute_face_normal_vectors_from_3d_simplices, _compute_adjacency, _compute_face_permutations, _pad_last_dimension
from functools import lru_cache
from typing import Tuple
from .cross_sections import CrossSectionDataSingleGroup

@jax.jit
def compute_domain_setup_from_mesh(mesh, face_template):
    '''
    Computes the setup for a geometric domain from a mesh and a face template (see `jax_sn.tools.mesh_tools._face_template` for an example template).

    Computes normals of each face and the element-to-element and element-to-face adjacency matrices.

    Parameters
    ----------
    mesh : tuple
        A tuple containing vertices and connectivity of the mesh.
    face_template : jnp.ndarray [n_faces_per_element, n_vertices_per_face]
        A 2D array representing the face template.
    '''
    assert face_template.ndim == 2, "Face template must be a 2D array."
    vertices, connectivity = mesh

    face_normals = compute_face_normal_vectors_from_3d_simplices(vertices[connectivity], face_template)
    etoe, etof   = _compute_adjacency(connectivity, face_template)
    face_perm_idx = _compute_face_permutations(connectivity, etoe, etof, face_template)

    return vertices, connectivity, face_normals, etoe, etof, face_perm_idx



@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False) # we use eq=False to use default (id-based) hashing.
class GeometricDomain:
    vertices      : jnp.ndarray
    connectivity  : jnp.ndarray
    face_normals  : jnp.ndarray    
    etoe          : jnp.ndarray
    etof          : jnp.ndarray
    face_perm_idx : jnp.ndarray
    face_template : jnp.ndarray

    @classmethod 
    def from_mesh(cls, mesh, face_template):
        return cls(
            *compute_domain_setup_from_mesh(mesh, face_template),
            face_template=face_template
        )
    @property
    def n_elements(self):
        return self.connectivity.shape[0]
    
    @property
    def n_faces_per_element(self):
        return self.face_template.shape[0]
    
    def containing_cell(self, points : jnp.ndarray):
        '''
        Containing cell for each point in points.
        Uses a cached BVH, so cannot be used in a jitted function

        Parameters
        ----------
        points : jnp.ndarray [n_points, n_dim]
            The points for which to find the containing cell.
        Returns
        -------
        element_indices : jnp.ndarray [n_points]
            The indices of the containing cell for each point. -1 if no containing cell is found
        '''
        return _containing_cell(self, points)

from .cross_sections import CrossSectionData

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossSectionIndexed:
    indices            : jnp.ndarray      #[n_elements]
    cross_section_data : CrossSectionData #[batched over n_cross_sections (where it should hold that n_cross_sections >= max(indices)+1])]

    def map_to_full_array(self, energy_group : int) -> CrossSectionDataSingleGroup:
        return _map_cross_sections_to_full_array(self, energy_group)

@jax.jit
def _map_cross_sections_to_full_array(cross_sections : CrossSectionIndexed, energy_group : int) -> CrossSectionDataSingleGroup:
    return CrossSectionDataSingleGroup(cross_sections.cross_section_data.total[cross_sections.indices,energy_group], cross_sections.cross_section_data.scattering[cross_sections.indices, : , energy_group, energy_group])
    
@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Domain(GeometricDomain):
    '''
    Geometric domain with indexed cross sections.
    This is seperated from GeometricDomain because many algorithms do not depend on cross sections and therefore we want to keep the cross sections distinct.
    '''
    cross_sections_indexed : CrossSectionIndexed
    
    @classmethod
    def from_mesh_and_cross_sections(cls, mesh : Tuple[jnp.ndarray, jnp.ndarray], face_template : jnp.ndarray, cross_sections_indexed : CrossSectionIndexed):
        
        _check_cross_section_compatibility(mesh[1].shape[0], cross_sections_indexed)
        return cls(
            *compute_domain_setup_from_mesh(mesh, face_template),
            face_template         = face_template,
            cross_sections_indexed = cross_sections_indexed
        )    
    @classmethod 
    def from_geometric_domain_and_cross_sections(cls, geometric_domain : GeometricDomain, cross_sections_indexed : CrossSectionIndexed):        
        return cls(geometric_domain.vertices, 
                   geometric_domain.connectivity, 
                   geometric_domain.face_normals, 
                   geometric_domain.etoe, 
                   geometric_domain.etof, 
                   geometric_domain.face_perm_idx, 
                   geometric_domain.face_template,
                   cross_sections_indexed=cross_sections_indexed
                  )    
    
def _check_cross_section_compatibility(n_elements : int, cross_sections : CrossSectionIndexed):
    '''
    Checks whether the cross sections are compatible with the geometric domain.
    Only is static array shape checking, not values.
    '''
    assert cross_sections.indices.ndim == 1, f"Cross section indices must be a 1D array [ n_elements ]. Provided shape: {cross_sections.indices.shape}"
    assert cross_sections.indices.shape[0] == n_elements, f"Cross section indices must have length equal to number of elements in geometric domain. Number of elements in geometric domain: {n_elements}, provided length: {cross_sections.indices.shape[0]}"
    cross_sections.cross_section_data.assert_valid_batched()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TopologicalSortedAngle:
    levels         : jnp.ndarray #[n_elements]
    angle          : jnp.ndarray #[n_dim]
    levels_argsort : jnp.ndarray #[n_elements]
    
    @classmethod 
    def from_domain_angle(cls, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        from .graph_generation import compute_levels_domain_safe
        levels = compute_levels_domain_safe(geometric_domain, angle, n_iterations_per_loop)
        levels_argsort = jnp.argsort(levels)
        return cls(levels=levels, angle=angle, levels_argsort=levels_argsort)
    
    def __getitem__(self, idx):
        assert self.levels.ndim == 2, "Indexing is only supported for batched TopologicalSortedAngle instances."
        return TopologicalSortedAngle(
            levels         = self.levels[idx],
            angle          = self.angle[idx],
            levels_argsort = self.levels_argsort[idx]
        )
    

@lru_cache(maxsize=16)
def _build_bvh(geometric_domain : GeometricDomain):    
    '''
    Builds a BVH for the given geometric domain. 
    This function uses caching to avoid rebuilding the BVH for the same domain multiple times.
    Therefore, it should not be used in side jitted functions.

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain for which to build the BVH.    
    Returns
    -------
    bvh : BVH
        The built BVH.
    '''
    from jax_sn.tools.mesh_tools import create_bvh
    return create_bvh(geometric_domain.vertices, geometric_domain.connectivity)

def _containing_cell(geometric_domain : GeometricDomain, points : jnp.ndarray):    
    if isinstance(points, jax.core.Tracer):
        raise RuntimeError("The function 'containing_cell' cannot be used inside jitted functions because it relies on a cached BVH.")    
        
    bvh = _build_bvh(geometric_domain)
    from jax_sn.tools.mesh_tools import probe_bvh 
    possible_hits =  probe_bvh(bvh, points)        
    elements_containing_point = _check_point_in_element_vec(points[..., None, :], possible_hits, geometric_domain)    
    element_indices           = jnp.where(elements_containing_point, possible_hits, -1)    
    first_elements             = jax.vmap(lambda x : x[jnp.where(x>=0, size=1)[0]], in_axes=0)(element_indices)
    return first_elements[:,0]
    


def _check_point_in_element(point : jnp.ndarray, element : int, geometric_domain : GeometricDomain):    
    '''
    Checks whether a point is inside the element of the geometric domain. 

    Assumes the domain contains *convex* elements. 

    Checks whether the point is inside all half-spaces defined by the elements faces.

    Parameters
    ----------
    point : jnp.ndarray [n_dim]
        The point to check.
    element : int
        The index of the element to check against.
    geometric_domain : GeometricDomain
        The geometric domain containing the element.
    Returns
    -------
    inside : bool
        True if the point is inside the element, False otherwise.
    '''
    vertices_ref_face = geometric_domain.connectivity[element, geometric_domain.face_template[:,0]] #[n_faces]    
    x_min_ref_face    = point - geometric_domain.vertices[vertices_ref_face]                        #[n_faces, n_dim]    
    face_normals = geometric_domain.face_normals[element] # n_faces, n_dim    
    inside = jnp.sum(x_min_ref_face * face_normals[:, :x_min_ref_face.shape[-1]], axis=-1) <= 0.0    
    return jnp.all(inside)

_check_point_in_element_vec = jnp.vectorize(_check_point_in_element, excluded=(2,), signature='(n_dim),()->()')
    