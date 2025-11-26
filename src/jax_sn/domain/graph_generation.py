import jax
import jax.numpy as jnp
from .geometric_domain import GeometricDomain
from functools import partial
from jax_sn.tools.graph_tools import _compute_levels_parallel_scan, _compute_levels_parallel_while

def _compute_outflow_bool(face_normals : jnp.ndarray, angle : jnp.ndarray) -> jnp.ndarray:
    normal_dot_angle = jnp.sum(face_normals * angle[None, None, :], axis=-1) 
    outflow          = normal_dot_angle > 0    
    return outflow

def compute_face_outflow(geometric_domain : GeometricDomain, angle : jnp.ndarray) -> jnp.ndarray:
    return _compute_outflow_bool(geometric_domain.face_normals, angle)

@partial(jax.jit, static_argnums = (2,3))
def compute_levels_domain(geometric_domain : GeometricDomain, angle : jnp.ndarray, n_max_iterations : int = 1500, while_loop : bool = True) -> jnp.ndarray:
    outflows                      = compute_face_outflow(geometric_domain, angle) #[n_elements, n_faces_per_element]    
    
    # Mask out outflows on domainboundary faces
    outflows_connected_to_element = (geometric_domain.etof != -1) & outflows # [n_elements, n_faces_per_element]     

    # matrix that for each element i and each face k gives the element connected to element i through face k if that face is an outflow face, else -1    
    outflow_face_to_element_matrix = jnp.where(outflows_connected_to_element, geometric_domain.etoe, geometric_domain.n_elements)    
    if while_loop:
        levels = _compute_levels_parallel_while(outflow_face_to_element_matrix,  n_max_iterations)
    else:
        levels = _compute_levels_parallel_scan(outflow_face_to_element_matrix, n_max_iterations)
    return levels
