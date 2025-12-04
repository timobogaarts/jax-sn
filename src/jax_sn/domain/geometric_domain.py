import jax
import jax.numpy as jnp 
from dataclasses import dataclass
from jax_sn.tools.mesh_tools import compute_face_normal_vectors_from_3d_simplices, _compute_adjacency, _compute_face_permutations

#@jax.jit
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
@dataclass(frozen=True)
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

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TopologicalSortedAngle:
    levels         : jnp.ndarray
    angle          : jnp.ndarray
    levels_argsort : jnp.ndarray
    
    @classmethod 
    def from_domain_angle(cls, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        from .graph_generation import compute_levels_domain_safe
        levels = compute_levels_domain_safe(geometric_domain, angle, n_iterations_per_loop)
        levels_argsort = jnp.argsort(levels)
        return cls(levels=levels, angle=angle, levels_argsort=levels_argsort)
    
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PrecomputedGeometricData:
    vol_jacobians         : jnp.ndarray
    vol_jacobians_inv     : jnp.ndarray
    vol_jacobians_det     : jnp.ndarray
    face_jacobians_det    : jnp.ndarray

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossSectionData:
    total           : jnp.ndarray    