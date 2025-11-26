import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums = (1))
def _compute_levels_parallel_scan(outflow_face_to_element_matrix : jnp.ndarray, n_max_iterations : int = 1500) -> jnp.ndarray:
    '''
    Compute levels of elements in a graph defined by outflow_face_to_element_matrix using a parallel scan.
    Each element's level is determined by the maximum level of its downstream elements plus one (zero upstream elements -> 0 level).

    Due to JAX's compilation requiring static shapes, all elements compute for all faces every iteration. Therefore, this is not 
    an optimized algorithm in any way; but it is sufficiently fast for n_elements < 1e6. [<200 ms per angle for 1e6 elements]

    Parameters:
    -----------
    outflow_face_to_element_matrix : jnp.ndarray
        A (n_elements, n_faces_per_element) array where each row contains the indices of downstream elements for each element.
        Use n_elements as a dummy index for faces that do not connect to any downstream element.
    n_elements : int
        The total number of elements in the graph.
    n_max_iterations : int, optional
        The maximum number of iterations to perform. Default is 1500.
    Returns:
    --------
    jnp.ndarray
        An array of shape (n_elements,) containing the computed levels for each element.

    '''
    # initial levels of all elements is set to n_elements (one larger than would be actually possible!).    
    n_elements = outflow_face_to_element_matrix.shape[0]
    levels_elements  = jnp.full(n_elements + 1, n_elements, dtype = outflow_face_to_element_matrix.dtype)
    levels_elements  = levels_elements.at[-1].set(0) # dummy element that is always computable (i.e. no connection -> level 0 available)    
    def level_body(state, i):
        return jnp.concatenate([jnp.max(state[outflow_face_to_element_matrix], axis=-1) + 1, jnp.array([0])]), None                        
    levels_final, _ = jax.lax.scan(level_body, levels_elements, jnp.arange(n_max_iterations))
    return levels_final[:-1] - 1


@partial(jax.jit, static_argnums = (1))
def _compute_levels_parallel_while(outflow_face_to_element_matrix : jnp.ndarray, n_max_iterations : int = 1500) -> jnp.ndarray:
    '''
    Compute levels of elements in a graph defined by outflow_face_to_element_matrix using a parallel while loop.
    Each element's level is determined by the maximum level of its downstream elements plus one (zero upstream elements -> 0 level).

    Computes levels until all levels are computed or n_max_iterations is reached.

    Due to JAX's compilation requiring static shapes, all elements compute all levels every iteration. Therefore, this is not 
    an optimized algorithm in any way; but it is sufficiently fast for n_elements < 1e6. [<200 ms per angle for 1e6 elements]

    Parameters:
    -----------
    outflow_face_to_element_matrix : jnp.ndarray
        A (n_elements, n_faces_per_element) array where each row contains the indices of downstream elements for each element.
        Use n_elements as a dummy index for faces that do not connect to any downstream element.
    n_elements : int
        The total number of elements in the graph.
    n_max_iterations : int, optional
        The maximum number of iterations to perform. Default is 1500.
    Returns:
    --------
    jnp.ndarray
        An array of shape (n_elements,) containing the computed levels for each element.

    '''
    n_elements = outflow_face_to_element_matrix.shape[0]
    # initial levels of all elements is set to n_elements (one larger than would be actually possible!).    
    levels_elements  = jnp.full(n_elements + 1, n_elements, dtype = outflow_face_to_element_matrix.dtype)
    levels_elements  = levels_elements.at[-1].set(0) # dummy element that is always computable (i.e. no connection -> level 0 available)    
    def cond(state):
        return jnp.any(state[0][:-1] >= n_elements) & (state[-1] < n_max_iterations)
    
    def body(state):
        levels_elements, loop_i = state
        return jnp.concatenate([jnp.max(levels_elements[outflow_face_to_element_matrix], axis=-1) + 1, jnp.array([0])]), loop_i + 1
    
    levels_final, _ = jax.lax.while_loop(cond, body, (levels_elements, 0))
    return levels_final[:-1] - 1