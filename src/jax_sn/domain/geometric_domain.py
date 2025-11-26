import jax
import jax.numpy as jnp 
from dataclasses import dataclass
from jax_sn.tools.mesh_tools import create_face_normal_vectors_from_3d_simplices, _compute_adjacency


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

    return vertices, connectivity, create_face_normal_vectors_from_3d_simplices(vertices[connectivity], face_template), *_compute_adjacency(connectivity, face_template)



@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GeometricDomain:
    vertices     : jnp.ndarray
    connectivity : jnp.ndarray
    normals      : jnp.ndarray    
    etoe         : jnp.ndarray
    etof         : jnp.ndarray
    face_template: jnp.ndarray

    @classmethod 
    def from_mesh(cls, mesh, face_template):
        return cls(
            *compute_domain_setup_from_mesh(mesh, face_template),
            face_template=face_template
        )
    @property
    def n_elements(self):
        return self.connectivity.shape[0]


