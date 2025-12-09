import jax 
import numpy as onp 
import jax.numpy as jnp
from dataclasses import dataclass

from jax_sn.domain.geometric_domain import TopologicalSortedAngle
from typing import Tuple


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Level based parallelism
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TopologicalSortedAnglePrecomputedLevels(TopologicalSortedAngle):    
    padded_levels               : jnp.ndarray = None #[n_padded_elements]    
    sizes                       : jnp.ndarray = None #[n_levels]
    unique_sizes                : jnp.ndarray = None #[n_unique_sizes]
    starting_indices_padded     : jnp.ndarray = None #[n_levels]
    starting_indices_not_padded : jnp.ndarray = None #[n_levels]
    level_to_unique_size        : jnp.ndarray = None #[n_levels]

    @classmethod
    def from_topological_sorted_angle(cls, topological_sorted_angle: TopologicalSortedAngle) -> 'TopologicalSortedAnglePrecomputedLevels':
        return cls(*precompute_levels_padded_pow2(topological_sorted_angle))

def precompute_levels_padded_pow2(topological_sorted_angle: TopologicalSortedAngle):
    padded_levels, sizes, unique_sizes = _setup_levels_padded_pow2_angle(topological_sorted_angle)
    starting_indices_padded            = jnp.concatenate([jnp.array([0]), jnp.cumsum(sizes)[:-1]])
    starting_indices_not_padded        = jnp.concatenate([jnp.array([0]), jnp.cumsum(jnp.bincount(topological_sorted_angle.levels))[:-1]])            
    mapping = onp.zeros(onp.max(unique_sizes) + 1, dtype=int)
    mapping[onp.array(unique_sizes,dtype=int)] = onp.arange(unique_sizes.shape[0])
    level_to_unique_size = mapping[onp.array(sizes,dtype=int)]
    return topological_sorted_angle.levels, topological_sorted_angle.angle, topological_sorted_angle.levels_argsort, jnp.array(padded_levels),  jnp.array(sizes),  jnp.array(unique_sizes), starting_indices_padded, starting_indices_not_padded, jnp.array(level_to_unique_size)

def _setup_levels_padded_pow2(levels : onp.ndarray, levels_argsort : onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    '''
    Setups level arrays for level-based parallelism. For each level, the element indices are stored in a padded array such that each level has a size that is a power of two.
    This allows for efficient parallel processing of elements at the same level without compiling too many different sizes.

    This uses ordinary numpy because it involves multiple dynamic slices and creations and this function is only called once during setup. 
    It furthermore does not need to be differentiable as this is not a parameter; just a setup step.

    In other words, if we have levels: [1,1,1, 0, 2,2, 3, 3, 3, 3, 3], we create the following arrays:
    
    1. padded array of power of two of each level indices:    
                [3, 0, 1, 2,-1, 5, 6, 7, 8, 9, -1, -1, -1] 
    (corr lvl:)  0, 1, 1, 1, 1, 2, 2, 2, 2, 2,  2,  2,  2
    2. power of two padded size for each level           
                 [1, 4, 2, 8] 
    (corr lvl:)  [0, 1, 2, 3]    

    3. all different sizes used for padding:                                                
                 [1, 4,2,8]

    Parameters:
    -----------
    levels: jnp.ndarray [n_elements]
        Array of levels for each element.
    levels_argsort: jnp.ndarray [n_elements]
        Array of element indices sorted by levels.

    Returns:
    --------
    total_array: jnp.ndarray [sum(padded_sizes)]
        Padded array of element indices for each level.
    padded_sizes: jnp.ndarray [n_levels]
        Padded sizes for each level.
    unique_sizes: jnp.ndarray [n_unique_sizes]
        Unique padded sizes used.
    -----------
    '''
    levels               = onp.array(levels)
    levels_argsort       = onp.array(levels_argsort)
    levels_bincount      = onp.bincount(levels)    
    max_level            = onp.max(levels)

    levels_bincount_log2 = onp.ceil(onp.log2(levels_bincount)).astype(onp.int32)    
    padded_sizes         = 2**levels_bincount_log2
    total_array_size     = onp.sum(padded_sizes)

    total_array          = onp.full(total_array_size, -1, dtype = onp.int32)

    levels_start_original_array = onp.cumsum(onp.concatenate(([0], levels_bincount)))
    levels_start_padded_array   = onp.cumsum(onp.concatenate(([0], padded_sizes)))

    for lvl in range(max_level + 1):
        total_array[levels_start_padded_array[lvl]:levels_start_padded_array[lvl] + levels_bincount[lvl]] = levels_argsort[levels_start_original_array[lvl] : levels_start_original_array[lvl + 1]]
    return total_array, padded_sizes, onp.unique(padded_sizes)
    

def _setup_levels_padded_pow2_angle(toposort : TopologicalSortedAngle) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    '''
    Setups level arrays for level-based parallelism for a given topologically sorted angle.

    See _setup_levels_padded_pow2 for more information.

    Parameters:
    -----------
    toposort: TopologicalSortedAngle
        The topologically sorted angle data.

    Returns:
    --------
    total_array: jnp.ndarray [sum(padded_sizes)]
        Padded array of element indices for each level.
    padded_sizes: jnp.ndarray [n_levels]
        Padded sizes for each level.
    unique_sizes: jnp.ndarray [n_unique_sizes]
        Unique padded sizes used.
    -----------
    '''
    return _setup_levels_padded_pow2(toposort.levels, toposort.levels_argsort)
