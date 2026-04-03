import jax
import jax.numpy as jnp 
from dataclasses import dataclass
from jax_sn.tools.graph_tools import multiple_angle_level_bounds_safe
from jax_sn.tools.mesh_tools import compute_face_normal_vectors_from_3d_simplices, _compute_adjacency, _compute_face_permutations, _pad_last_dimension
from functools import lru_cache
from typing import Tuple
from .cross_sections import CrossSectionDataSingleGroup
from jax_sn.quadrature_set import QuadratureSet
from typing import Tuple 
import equinox as eqx
from typing import Union

@jax.jit
def _compute_domain_setup_from_mesh(mesh : Tuple[jnp.ndarray, jnp.ndarray], face_template : jnp.ndarray):
    '''
    Computes the setup for a geometric domain from a mesh and a face template (see `jax_sn.tools.mesh_tools._face_template` for an example template).

    Computes normals of each face and the element-to-element and element-to-face adjacency matrices.

    Parameters
    ----------
    mesh : tuple
        A tuple containing vertices and connectivity of the mesh. vertices is a jnp.ndarray of shape [n_vertices, n_dim] and connectivity is a jnp.ndarray of shape [n_elements, n_vertices_per_element].
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
    '''
    Dataclass representing a geometric domain. 
    These contain precomputed connectivity, face normals, and adjacency information. 
    We use id-based hasing to be able to cache BVH structures based on the domain. 

    '''
    
    vertices      : jnp.ndarray
    '''Vertices of the domain. Shape is [n_vertices, n_dim]'''

    connectivity  : jnp.ndarray
    '''Connectivity of the domain. Shape is [n_elements, n_vertices_per_element]'''

    face_normals  : jnp.ndarray    
    '''Face normals of the domain. Shape is [n_elements, n_faces_per_element, n_dim]'''
    
    etoe          : jnp.ndarray
    '''Element-to-element adjacency. Shape is [n_elements, n_faces_per_element]. -1 indicates a boundary face (unconnected to another element).'''

    etof          : jnp.ndarray
    '''Element-to-face adjacency. Shape is [n_elements, n_faces_per_element]. Each face has a unique index in the domain.'''

    face_perm_idx : jnp.ndarray
    '''Face permutation indices. Shape is [n_elements, n_faces_per_element]. Indicates how the face vertices are permuted between adjacent elements.'''

    face_template : jnp.ndarray
    '''Face template used to define faces of elements. Shape is [n_faces_per_element, n_vertices_per_face].'''

    @classmethod 
    def from_mesh(cls, mesh : Tuple[jnp.ndarray, jnp.ndarray], face_template : jnp.ndarray):
        '''
        Constructs a GeometricDomain from a mesh and a face template (see `jax_sn.tools.mesh_tools._face_template` for an example template).
        
        Parameters
        ----------
        mesh : tuple
            A tuple containing vertices and connectivity of the mesh. vertices is a jnp.ndarray of shape [n_vertices, n_dim] and connectivity is a jnp.ndarray of shape [n_elements, n_vertices_per_element].
        face_template : jnp.ndarray [n_faces_per_element, n_vertices_per_face]
            A 2D array representing the face template.
        Returns
        -------
        domain : GeometricDomain
            The constructed GeometricDomain.
        '''

        return cls(
            *_compute_domain_setup_from_mesh(mesh, face_template),
            face_template=face_template
        )
    
    @classmethod 
    def from_mesh_numpy(cls, mesh : Tuple[jnp.ndarray, jnp.ndarray], face_template : jnp.ndarray):
        '''
        Constructs a GeometricDomain from a mesh and a face template (see `jax_sn.tools.mesh_tools._face_template` for an example template).

        Uses a numpy implementation of the domain setup. This avoids compilation overheads of especially the lexsort, which can take significant time.
        
        Parameters
        ----------
        mesh : tuple
            A tuple containing vertices and connectivity of the mesh. vertices is a jnp.ndarray of shape [n_vertices, n_dim] and connectivity is a jnp.ndarray of shape [n_elements, n_vertices_per_element].
        face_template : jnp.ndarray [n_faces_per_element, n_vertices_per_face]
            A 2D array representing the face template.
        Returns
        -------
        domain : GeometricDomain
            The constructed GeometricDomain.
        '''
        from jax_sn.tools.mesh_tools_numpy import _compute_domain_setup_from_mesh_numpy

        domain_setup = _compute_domain_setup_from_mesh_numpy(mesh, face_template)

        return cls(
            *[jnp.array(x) for x in domain_setup],
            face_template=jnp.array(face_template)
        )
    @property
    def n_elements(self):
        '''Number of elements in the domain'''
        return self.connectivity.shape[0]
    
    @property
    def n_faces_per_element(self):
        '''Number of faces per element in the domain'''
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

from .cross_sections import CrossSectionData, CrossSectionDataSingleGroup, CrossSectionsIndexed

@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Domain(GeometricDomain):
    '''
    Geometric domain with indexed cross sections.
    This is seperated from GeometricDomain because many algorithms do not depend on cross sections and therefore we want to keep the cross sections distinct.
    '''

    
    cross_sections_indexed : CrossSectionsIndexed
    '''Indexed cross sections for the domain.'''

    
    @classmethod
    def from_mesh_and_cross_sections(cls, mesh : Tuple[jnp.ndarray, jnp.ndarray], face_template : jnp.ndarray, cross_sections_indexed : CrossSectionsIndexed):
        '''
        Constructs a Domain from a mesh, face template, and indexed cross sections. Checks compatibility between the mesh and cross sections.
        
        Parameters
        ----------
        mesh : tuple
            A tuple containing vertices and connectivity of the mesh. vertices is a jnp.ndarray of shape [n_vertices, n_dim] and connectivity is a jnp.ndarray of shape [n_elements, n_vertices_per_element].
        face_template : jnp.ndarray [n_faces_per_element, n_vertices_per_face]
            A 2D array representing the face template.
        cross_sections_indexed : CrossSectionsIndexed
            Indexed cross sections for the domain.
        Returns
        -------
        domain : Domain
            The constructed Domain.
        '''
        
        _check_cross_section_compatibility(mesh[1].shape[0], cross_sections_indexed)
        return cls(
            *_compute_domain_setup_from_mesh(mesh, face_template),
            face_template         = face_template,
            cross_sections_indexed = cross_sections_indexed
        )    
    
    @classmethod
    def from_mesh_and_cross_sections_numpy(cls, mesh : Tuple[jnp.ndarray, jnp.ndarray], face_template : jnp.ndarray, cross_sections_indexed : CrossSectionsIndexed):
        '''
        Constructs a Domain from a mesh, face template, and indexed cross sections. Checks compatibility between the mesh and cross sections.
        Uses a numpy implementation of the domain setup. This avoids compilation overheads of especially the lexsort, which can take significant time.

        Parameters
        ----------
        mesh : tuple
            A tuple containing vertices and connectivity of the mesh. vertices is a jnp.ndarray of shape [n_vertices, n_dim] and connectivity is a jnp.ndarray of shape [n_elements, n_vertices_per_element].
        face_template : jnp.ndarray [n_faces_per_element, n_vertices_per_face]
            A 2D array representing the face template.
        cross_sections_indexed : CrossSectionsIndexed
            Indexed cross sections for the domain.
        Returns
        -------
        domain : Domain
            The constructed Domain.
        '''
        from jax_sn.tools.mesh_tools_numpy import _compute_domain_setup_from_mesh_numpy

        _check_cross_section_compatibility(mesh[1].shape[0], cross_sections_indexed)
        domain_setup = _compute_domain_setup_from_mesh_numpy(mesh, face_template)

        return cls(
            *[jnp.array(x) for x in domain_setup],
            face_template=jnp.array(face_template),
            cross_sections_indexed=cross_sections_indexed
        )
    
    @classmethod 
    def from_geometric_domain_and_cross_sections(cls, geometric_domain : GeometricDomain, cross_sections_indexed : CrossSectionsIndexed):        
        '''
        Constructs a Domain from a GeometricDomain and indexed cross sections. Checks compatibility between the geometric domain and cross sections.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain.
        cross_sections_indexed : CrossSectionsIndexed
            Indexed cross sections for the domain.
        Returns
        -------
        domain : Domain
            The constructed Domain.
        '''
        assert isinstance(geometric_domain, GeometricDomain), f"geometric_domain must be an instance of GeometricDomain. Provided type: {type(geometric_domain)}"
        _check_cross_section_compatibility(geometric_domain.connectivity.shape[0], cross_sections_indexed)
        return cls(geometric_domain.vertices, 
                   geometric_domain.connectivity, 
                   geometric_domain.face_normals, 
                   geometric_domain.etoe, 
                   geometric_domain.etof, 
                   geometric_domain.face_perm_idx, 
                   geometric_domain.face_template,
                   cross_sections_indexed=cross_sections_indexed
                  )    
    
def _check_cross_section_compatibility(n_elements : int, cross_sections : CrossSectionsIndexed):
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
    '''
    Topologically sorted angle for a geometric domain. Contains information on the topological sort of elements for a given angle.    
    
    '''
    
    levels         : jnp.ndarray #[n_elements]
    '''
    Levels of the elements in the topological sort. Shape is [n_elements]. Each level can be computed in parallel.
    '''

    angle          : jnp.ndarray #[n_dim]
    '''
    The angle for which the topological sort is computed. Shape is [n_dim].
    '''

    levels_argsort : jnp.ndarray #[n_elements]
    '''
    The indices that sort the elements by their levels. Shape is [n_elements].
    '''
    
    @classmethod 
    def from_domain_angle(cls, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        '''
        Computes a topological sort for the given angle and geometric domain. 
        This function is not jittable because it uses Python control flow to safely compute levels, which can throw if there is a cycle in the graph.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain for which to compute the topological sort.
        angle : jnp.ndarray [n_dim]
            The angle for which to compute the topological sort.
        n_iterations_per_loop : int, optional
            The number of iterations per loop for the level computation. The default is 100.

        Returns
        -------
        toposort : TopologicalSortedAngle
            The computed topological sort.
        '''
        from .graph_generation import compute_levels_domain_safe
        levels = compute_levels_domain_safe(geometric_domain, angle, n_iterations_per_loop)
        levels_argsort = jnp.argsort(levels)
        return cls(levels=levels, angle=angle, levels_argsort=levels_argsort)
    
    @classmethod
    def from_domain_quadrature_set(cls, geometric_domain : GeometricDomain, quadrature_set : QuadratureSet, n_iterations_per_loop : int = 100):
        '''
        Computes topological sorts for all angles in the given quadrature set and geometric domain.
        This function is not jittable because it uses Python control flow to safely compute levels, which can throw if there is a cycle in the graph.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain for which to compute the topological sorts.
        quadrature_set : QuadratureSet
            The quadrature set containing the angles for which to compute the topological sorts.
        n_iterations_per_loop : int, optional
            The number of iterations per loop for the level computation. The default is 100.
        Returns
        -------
        toposorts : TopologicalSortedAngle
            The computed topological sorts for all angles in the quadrature set. Batched over angles.
        '''
        return cls(*_topologicalsorted_angle_from_domain_quadrature_set_scan(geometric_domain, quadrature_set, n_iterations_per_loop))        

    
    def __getitem__(self, idx : int) -> "TopologicalSortedAngle":
        '''
        Indexing for batched TopologicalSortedAngle instances.
        Allows to get a single TopologicalSortedAngle from a batched instance.

        Parameters
        ----------
        idx : int
            The index of the angle to get.
        Returns
        -------
        toposort : TopologicalSortedAngle
            The TopologicalSortedAngle instance for the given index.
        '''
        assert self.levels.ndim == 2, "Indexing is only supported for batched TopologicalSortedAngle instances."
        return TopologicalSortedAngle(
            levels         = self.levels[idx],
            angle          = self.angle[idx],
            levels_argsort = self.levels_argsort[idx]
        )
    
    @property
    def reverse(self):
        '''
        Reverses the angle and topological sort. This is also correct for batched angles.        
        The levels are reversed by subtracting from the maximum level. This is non-optimal, but produces a valid topological sort.

        Returns
        -------
        reversed_toposort : TopologicalSortedAngle
            The reversed topological sort.
        '''        
        return TopologicalSortedAngle(
            levels         = jnp.max(self.levels, axis = -1, keepdims=True) - self.levels, # technically not optimal, but produces valid levels [not entirely sure, but in general we don't use this anyway except for level-based parallelism....]
            angle          = -1.0 * self.angle,
            levels_argsort = self.levels_argsort[..., ::-1] 
        )
    
@eqx.filter_jit
def _topologicalsorted_angle_from_domain_quadrature_set_scan(geometric_domain : GeometricDomain, quadrature_set : QuadratureSet, n_iterations_per_loop : int = 100):
    '''
    Jittable version of TopologicalSortedAngle.from_domain_quadrature_set.
    Computes topological sorts for all angles in the given quadrature set and geometric domain.

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain for which to compute the topological sorts.
    quadrature_set : QuadratureSet
        The quadrature set containing the angles for which to compute the topological sorts.
    n_iterations_per_loop : int, optional
        The number of iterations per loop for the level computation. The default is 100.
    Returns
    -------
    toposorts : TopologicalSortedAngle
        The computed topological sorts for all angles in the quadrature set. Batched over angles.
    '''
    from jax_sn.domain.graph_generation import compute_levels_domain_safe_jax

    def body_fn(carry, angle):
        levels = compute_levels_domain_safe_jax(geometric_domain, angle, n_iterations_per_loop)
        return carry, levels
    
    levels = jax.lax.scan(body_fn, None, quadrature_set.angles)[1] #[n_angles, n_elements]        
    levels_argsort = jnp.argsort(levels, axis=-1) #[n_angles, n_elements]
    return levels, quadrature_set.angles, levels_argsort

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
    '''
    Containing cell for each point in points.
    Uses a cached BVH, so cannot be used in a jitted function

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain for which to find the containing cells.
    points : jnp.ndarray [n_points, n_dim]
        The points for which to find the containing cell.
    Returns
    -------
    element_indices : jnp.ndarray [n_points]
        The indices of the containing cell for each point. -1 if no containing cell is found    
    '''
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
    

class LevelBasedParallelism(eqx.Module):
    '''
    Class providing utilities for level-based parallelism in the context of transport sweeps. Stores the first occurrence, end, and size of each level for each angle, based on the topological sorted angles provided by the transport operator.

    Since JAX does not allow dynamic shapes, the class method `from_topological_sorted_angle` optionally is given a maximum number of levels, which is used to pad the resulting arrays to a fixed shape. If this is not 
    sufficiently high enough, an equinox error is given.
        
    '''
    first_occurences : jnp.ndarray 
    '''
    First occurrence of each level, per angle. Shape [..., max_levels]
    '''
    ends : jnp.ndarray 
    '''
    End of each level (i.e. start of next level), per angle. Shape [..., max_levels]
    '''
    sizes : jnp.ndarray 
    '''
    Number of elements in each level, per angle. Shape [..., max_levels]
    '''

    @classmethod
    def from_topological_sorted_angle(cls, topological_sorted_angle : TopologicalSortedAngle, maximum_number_of_levels : Union[int, None] = None):
        '''
        Create a LevelBasedParallelism instance from a TopologicalSortedAngle, which contains the levels and their argsort for each angle. Optionally, a maximum number of levels can be provided to pad the resulting arrays to a fixed shape. This has 
        the advantage that array sizes are known at compile time. If not sufficiently high enough, it will raise an equinox error.

        Parameters
        ----------
        topological_sorted_angle : TopologicalSortedAngle
            The topological sorted angle containing the levels and their argsort for each angle.
        maximum_number_of_levels : int, optional
            The maximum number of levels to pad to. If None, it will be set to the maximum number of levels present in the topological sorted angle. Default is None.   

        Returns 
        -------
        LevelBasedParallelism
            An instance of LevelBasedParallelism containing the first occurrence, end, and size of each level for each angle, based on the topological sorted angles provided by the transport operator.
        '''
        first_occurences, ends, sizes = multiple_angle_level_bounds_safe(topological_sorted_angle.levels, topological_sorted_angle.levels_argsort, maximum_number_of_levels)
        return cls(first_occurences=first_occurences, ends=ends, sizes=sizes)
    
    @property 
    def n_levels(self):
        '''
        Number of distinct levels
        '''
        return self.first_occurences.shape[-1]
    