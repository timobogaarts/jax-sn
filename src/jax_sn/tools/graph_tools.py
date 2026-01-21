import jax
import jax.numpy as jnp
from functools import partial
import equinox as eqx

def _increment_levels_including_dummy(levels_elements : jnp.ndarray, dependency_matrix : jnp.ndarray) -> jnp.ndarray:
    '''
    *internal*
    Increment levels of elements in a graph defined by dependency_matrix.   
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).
    ''' 
    return jnp.concatenate([jnp.max(levels_elements[dependency_matrix], axis=-1) + 1, jnp.array([0])])

@partial(jax.jit, static_argnums = (2))
def _scan_loop_levels_including_dummy(levels_elements : jnp.ndarray, dependency_matrix : jnp.ndarray, n_iterations : int) -> jnp.ndarray:
    '''
    *internal*
    Compute levels of elements in a graph defined by dependency_matrix using a scan loop.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).
    This wraps a scan operation for a fixed number of iterations.
    '''
    def body(state, i):
        return _increment_levels_including_dummy(state, dependency_matrix), None
    levels_final, _ = jax.lax.scan(body, levels_elements, jnp.arange(n_iterations))
    return levels_final

@partial(jax.jit, static_argnums = (2))
def _while_loop_levels_including_dummy(levels_elements : jnp.ndarray, dependency_matrix : jnp.ndarray, n_max_iterations : int) -> jnp.ndarray:
    '''
    *internal*
    Compute levels of elements in a graph defined by dependency_matrix using a while loop.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).
    This wraps a while loop that continues until all levels are computed or n_max_iterations is reached.
    Possibly significantly slower than the scan version; use with care. Does not check for cycles in the graph.    
    '''
    def cond(state):
        return jnp.any(state[0][:-1] >= dependency_matrix.shape[0]) & (state[-1] < n_max_iterations)    
    def body(state):        
        return _increment_levels_including_dummy(state[0], dependency_matrix), state[1] + 1    
    levels_final, _ = jax.lax.while_loop(cond, body, (levels_elements, 0))
    return levels_final

@partial(jax.jit)
def _setup_levels_including_dummy(dependency_matrix : jnp.ndarray) -> jnp.ndarray:
    '''
    *internal* 
    Setup initial levels of elements in a graph defined by dependency_matrix.   
    Initial levels of all elements is set to n_elements (one larger than would be actually possible!).
    A dummy element is added at the end that is always computable (i.e. no connection -> level 0 available).
    '''
    n_elements = dependency_matrix.shape[0]
    levels_elements  = jnp.full(n_elements + 1, n_elements, dtype = dependency_matrix.dtype)
    levels_elements  = levels_elements.at[-1].set(0) # dummy element that is always computable (i.e. no connection -> level 0 available)    
    return levels_elements

@partial(jax.jit)
def _levels_including_dummy_to_levels(levels_including_dummy : jnp.ndarray) -> jnp.ndarray:
    '''
    *internal* 
    Converts the levels including dummy element to levels without dummy element.
    Simply slices to last element and decreases all levels by one.
    '''
    return levels_including_dummy[:-1] - 1

@partial(jax.jit, static_argnums = (1))
def _compute_levels_parallel_scan(dependency_matrix : jnp.ndarray, n_iterations : int) -> jnp.ndarray:
    '''
    Compute levels of elements in a graph defined by dependency_matrix using a parallel scan.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).

    Due to JAX's compilation requiring static shapes, all elements compute for all faces every iteration. Therefore, this is not 
    an optimized algorithm in any way; but it is sufficiently fast for n_elements < 1e6. [<200 ms per angle for 1e6 elements]

    Parameters:
    -----------
    dependency_matrix : jnp.ndarray
        A (n_elements, n_faces_per_element) array where each row contains the indices of upstream elements for each element.
        Use n_elements as a dummy index for faces that do not connect to any upstream element.
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
    return _levels_including_dummy_to_levels(_scan_loop_levels_including_dummy(_setup_levels_including_dummy(dependency_matrix), dependency_matrix, n_iterations))

@partial(jax.jit, static_argnums = (1))
def _compute_levels_parallel_while(dependency_matrix : jnp.ndarray, n_max_iterations : int) -> jnp.ndarray:
    '''
    Compute levels of elements in a graph defined by dependency_matrix using a parallel while loop.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).

    Computes levels until all levels are computed or n_max_iterations is reached.

    Due to JAX's compilation requiring static shapes, all elements compute all levels every iteration. Therefore, this is not 
    an optimized algorithm in any way; but it is sufficiently fast for n_elements < 1e6. [<200 ms per angle for 1e6 elements]

    Parameters:
    -----------
    dependency_matrix : jnp.ndarray
        A (n_elements, n_faces_per_element) array where each row contains the indices of upstream elements for each element.
        Use n_elements as a dummy index for faces that do not connect to any upstream element.
    n_elements : int
        The total number of elements in the graph.
    n_max_iterations : int, optional
        The maximum number of iterations to perform. Default is 1500.
    Returns:
    --------
    jnp.ndarray
        An array of shape (n_elements,) containing the computed levels for each element.
    '''
    return _levels_including_dummy_to_levels(_while_loop_levels_including_dummy(_setup_levels_including_dummy(dependency_matrix), dependency_matrix, n_max_iterations))

def compute_levels_iterative(dependency_matrix : jnp.ndarray, n_iterations_per_loop : int) -> jnp.ndarray:
    '''
    Compute levels of elements in a graph defined by dependency_matrix using a safe while loop.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).

    This function uses a Python loop to iteratively compute levels until all levels are computed or no new levels can be computed.
    This uses static number of iterations for each JAX-compiled step to avoid recompilation and to be able to use jax.lax.scan. 
    This checks for cycles in the graph and raises an error if a cycle is detected.

    For large enough graphs and n_iterations_per_loop, the Python overhead is insignificant compared to the JAX computation time.
    Since this function is safer than the other two level computation functions, it is recommended to use this function when possible.
    
    :param dependency_matrix: A (n_elements, n_faces_per_element) array where each row contains the indices of upstream elements for each element.
                              Use n_elements as a dummy index for faces that do not connect to any upstream element.
    :param n_iterations_per_loop: The number of iterations to perform in each JAX-compiled step.  
    :return: An array of shape (n_elements,) containing the computed levels for each element.    
    
    '''
    levels = _setup_levels_including_dummy(dependency_matrix)
    n_elements = dependency_matrix.shape[0]
    
    absolute_maximum_loops = (n_elements + 1) // n_iterations_per_loop + 1
        
    n_elements_computed    = 0 
    for _ in range(absolute_maximum_loops):                                
        levels = _scan_loop_levels_including_dummy(levels, dependency_matrix, n_iterations_per_loop)
        n_elements_computed_new = jnp.sum(levels < n_elements)        
        if n_elements_computed_new >= n_elements:
            break
        if n_elements_computed_new == n_elements_computed:
            raise RuntimeError("No new elements are in the next level while not all elements can be computed. Graph cannot be computed: there are cycles present in the graph.")
        n_elements_computed = n_elements_computed_new
    return _levels_including_dummy_to_levels(levels)
        

@eqx.filter_jit
def compute_levels_iterative_jax(dependency_matrix : jnp.ndarray, n_iterations_per_loop : int) -> jnp.ndarray:
    '''
    Compute levels of elements in a graph defined by dependency_matrix using a safe while loop.
    Each element's level is determined by the maximum level of its upstream elements plus one (zero upstream elements -> 0 level).
    
    Uses a while loop and equinox runtime errors to check for cycles in the graph.    
    
    :param dependency_matrix: A (n_elements, n_faces_per_element) array where each row contains the indices of upstream elements for each element.
                              Use n_elements as a dummy index for faces that do not connect to any upstream element.
    :param n_iterations_per_loop: The number of iterations to perform in each JAX-compiled step.  
    :return: An array of shape (n_elements,) containing the computed levels for each element.    
    
    '''
    
    n_elements             = dependency_matrix.shape[0]    
    absolute_maximum_loops = (n_elements + 1) // n_iterations_per_loop + 1
    levels0                = _setup_levels_including_dummy(dependency_matrix)
    n_computed_0           = jnp.sum(levels0 < n_elements)

    state0 = (
        levels0,                                          # levels start
        0,                                                # loop idx
        n_computed_0,                                     # n_computed_0
        0                                                 # n_computed_prev        
        )   
    
    def cond_fn(state):
        levels, loop_idx, n_computed, n_computed_prev = state
        return jnp.logical_and(loop_idx < absolute_maximum_loops, jnp.logical_and(n_computed < n_elements, n_computed != n_computed_prev))
    
    def body_fn(state):
        levels, loop_idx, n_computed, n_computed_prev = state
        levels_new = _scan_loop_levels_including_dummy(levels, dependency_matrix, n_iterations_per_loop)
        n_computed_new = jnp.sum(levels_new < n_elements)
        return (
            levels_new,
            loop_idx + 1,
            n_computed_new,
            n_computed
        )
    
    def error_fn(state):
        state = eqx.error_if(state, jnp.logical_and(state[2] < n_elements, state[1] >= absolute_maximum_loops), f"Absolute maximum number of loops {absolute_maximum_loops} reached while not all elements could be computed. Graph cannot be computed: there may be cycles present in the graph.")
        state = eqx.error_if(state, jnp.logical_and(state[2] < n_elements, state[2] == state[3]), f"No new elements are in the next level while not all elements can be computed. Graph cannot be computed: there are cycles present in the graph.")
        return state 
    
    state = error_fn(jax.lax.while_loop(cond_fn, body_fn, state0))
    return _levels_including_dummy_to_levels(state[0])

