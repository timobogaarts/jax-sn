import jax 
import jax.numpy as jnp
from functools import partial
import math
def _obtain_permutation(indices_0 : jnp.ndarray, indices_1 : jnp.ndarray):
    sorted_0_idx  = jnp.argsort(indices_0)    
    pos_in_sorted = jnp.searchsorted(indices_0[sorted_0_idx], indices_1)
    return sorted_0_idx[pos_in_sorted]

@jax.jit
def _permutation_to_number(P):
    n = P.shape[0]
    def count_smaller(i, Pi):
        return jnp.sum(P[i+1:] < Pi)            
    L = jnp.array([count_smaller(i, P[i]) for i in range(n)])    
    factors = jnp.array([math.factorial(i) for i in range(n)][::-1])    
    return jnp.sum(L * factors)

@jax.jit
def _indices_to_permutation_number(indices_0 : jnp.ndarray, indices_1 : jnp.ndarray):
    permutation = _obtain_permutation(indices_0, indices_1)
    return _permutation_to_number(permutation)

@partial(jax.jit, static_argnums=(1,))
def _number_to_permutation(number : int, permutation_length : int):
    factors = jnp.array([math.factorial(i) for i in range(permutation_length)][::-1])    
    def _factorial_digits_function(carry, f):        
        return carry % f, carry // f    
    _, factorial_digits = jax.lax.scan(_factorial_digits_function, number, factors)
        
    perm_available = jnp.arange(permutation_length)
    def select_step(unused, digit):        
        indices      = jnp.cumsum(unused) # maps available elements to indices
        target_index = digit + 1                
        # this obtains the index in the original array where the target element is located (argmax gives *first* value)
        idx          = jnp.argmax(indices == target_index)         
        new_used     = unused.at[idx].set(False)
        return new_used, perm_available[idx]

    initial_unused = jnp.ones(permutation_length, dtype=bool)    
    _, perm = jax.lax.scan(select_step, initial_unused, factorial_digits)
    return perm

@partial(jax.jit, static_argnums=(0,))
def _get_all_permutations(permutation_length : int):    
    return jax.vmap(_number_to_permutation, in_axes=(0, None))(jnp.arange(math.factorial(permutation_length)), permutation_length)    
    