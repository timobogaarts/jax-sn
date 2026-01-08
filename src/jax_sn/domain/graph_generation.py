'''Module for graph generation utilities in the context of geometric domains.
This contains internal functions to compute inflow faces and element levels based on angles.

'''
import jax
import jax.numpy as jnp
from .geometric_domain import GeometricDomain
from functools import partial
from jax_sn.tools.graph_tools import _compute_levels_parallel_scan, _compute_levels_parallel_while, compute_levels_iterative

def _compute_inflow_bool(face_normals : jnp.ndarray, angle : jnp.ndarray) -> jnp.ndarray:
    normal_dot_angle = jnp.sum(face_normals * angle[None, None, :], axis=-1) 
    inflow          = normal_dot_angle < 0    
    return inflow

def compute_face_inflow(geometric_domain : GeometricDomain, angle : jnp.ndarray) -> jnp.ndarray:
    return _compute_inflow_bool(geometric_domain.face_normals, angle)

@jax.jit
def _get_inflow_indices_compressed(geometric_domain : GeometricDomain, angle : jnp.ndarray) -> jnp.ndarray:
    inflows                      = compute_face_inflow(geometric_domain, angle) #[n_elements, n_faces_per_element]    
    
    # Mask out inflows on domainboundary faces
    inflows_connected_to_element = (geometric_domain.etof != -1) & inflows # [n_elements, n_faces_per_element]     

    # matrix that for each element i and each face k gives the element connected to element i through face k if that face is an inflow face, else -1    
    dependency_matrix = jnp.where(inflows_connected_to_element, geometric_domain.etoe, geometric_domain.n_elements)    

    # We don't need all connections since not every face can be an outflow face for a closed element.
    def get_outflow_indices_compressed(inflows_connected_to_element_1):        
        indices = jnp.where(inflows_connected_to_element_1 < geometric_domain.n_elements, size= geometric_domain.n_faces_per_element -1, fill_value=-1)[0]        
        return jnp.where(indices >= 0, inflows_connected_to_element_1[indices], geometric_domain.n_elements)
        
    connection_one_less = jax.vmap(get_outflow_indices_compressed, in_axes=(0))(dependency_matrix)
    return connection_one_less

@partial(jax.jit, static_argnums = (2,3))
def compute_levels_domain(geometric_domain : GeometricDomain, angle : jnp.ndarray, n_max_iterations : int, while_loop : bool) -> jnp.ndarray:    
    '''
    Computes computation levels for elements in a geometric domain given an angle.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).

    See also `compute_levels_iterative` for a safer alternative that checks for cycles.

    Parameters:
    -----------
    geometric_domain : GeometricDomain
        The geometric domain containing the mesh and adjacency information.
    angle : jnp.ndarray
        A 1D array representing the angle for which to compute the levels.
    n_max_iterations : int
        The maximum number of iterations to perform. Default is 1500.
    while_loop : bool
        If True, use a jax.while_loop to compute levels; otherwise, use jax.lax.scan.    
    Returns:
    --------
    jnp.ndarray
        An array of shape (n_elements,) containing the computed levels for each element.
    '''
    connection_indices = _get_inflow_indices_compressed(geometric_domain, angle)
    f_levels           = _compute_levels_parallel_scan if not while_loop else _compute_levels_parallel_while    
    return f_levels(connection_indices, n_max_iterations)


def compute_levels_domain_safe(geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int ) -> jnp.ndarray:
    '''
    Computes computation levels for elements in a geometric domain given an angle.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).
        
    This function uses a safe iterative approach that checks for cycles in the graph.

    See also `compute_levels_domain` for a jittable version (also for i.e. vmaps) that does not check for cycles.

    Parameters:
    -----------
    geometric_domain : GeometricDomain
        The geometric domain containing the mesh and adjacency information.
    angle : jnp.ndarray
        A 1D array representing the angle for which to compute the levels.
    n_max_iterations : int
        The maximum number of iterations to perform. Default is 1500.
    while_loop : bool
        If True, use a jax.while_loop to compute levels; otherwise, use jax.lax.scan.    
    Returns:
    --------
    jnp.ndarray
        An array of shape (n_elements,) containing the computed levels for each element.
    '''
    connection_indices = _get_inflow_indices_compressed(geometric_domain, angle)
    return compute_levels_iterative(connection_indices, n_iterations_per_loop=n_iterations_per_loop)

