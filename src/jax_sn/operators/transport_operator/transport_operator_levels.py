import jax 
import numpy as onp 
import jax.numpy as jnp
from dataclasses import dataclass

from jax_sn.domain.geometric_domain import TopologicalSortedAngle
from typing import Tuple
from .transport_operator_imp import GeometricDomain, FEMMatrices, PrecomputedGeometricData, Source, CrossSectionDataSingleGroup, _setup_elements_idx_in_precomp_with_omega_n_calculation
from .transport_operator_imp import _collapse_dij, _collapse_dij_vmap
from functools import partial

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Level based parallelism
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# This file contains functions to precompute and use level-based parallelism for transport sweeps.
# The idea is to compute levels of elements in the dependency graph such that elements in the same level can be computed in parallel.
# This is done by first topologically sorting the elements and then grouping them into levels based on their dependencies.
# The levels are then padded to the next power of two for efficient parallel processing.
# The transport sweep is then performed level by level, with all elements in a level being computed in parallel.
# For a single angle, this is significantly faster than a regular topological sweep with vmap, especially for large problems with many elements.
# For multiple angles however, this is not the case, since this cannot be independently batched over angles due to the level structure which has different sizes for each angle.
# Since the angles have to be computed sequentially, the overhead of the level-based parallelism is not worth it for many angles.
# A simple test illustrates this:
# - For a 3D problem with ~60k elements and 2nd order basis functions, the following timings:
#    - Single angle:
#       - Regular topological sweep with vmap & fori: ~600 ms 
#       - Level-based parallelism sweep: ~20 - 75 ms
#    - 128 angles:
#       - Batched multiple angle sweep with vmap & fori : ~3.2s
#       - Level-based parallelism sweep: ~8.s (~128 * 63 ms)
# Thus, level-based parallelism is only beneficial for single angle sweeps or very few angles, or in the case of multiple devices where angles can be distributed over devices.

# It could be improved further by grouping levels with the same size together and processing them in a single batch, but this is not implemented yet.
# (there are always at least two angles with the same level structure: the angle and its opposite.)
# Finally, the minimum level size could be used to batch just over that. However, in my testing, the minimum level size is often 1, so this does not help much in practice.
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TopologicalSortedAnglePrecomputedLevels(TopologicalSortedAngle):    
    '''
    Topologically sorted angle with precomputed level data for level-based parallelism.

    Padded arrays for level-based parallelism where each level has a size that is a power of two for efficient parallel processing.
    '''
    padded_levels               : jnp.ndarray = None #[n_padded_elements]    
    sizes                       : jnp.ndarray = None #[n_levels]
    unique_sizes                : jnp.ndarray = None #[n_unique_sizes]
    starting_indices_padded     : jnp.ndarray = None #[n_levels]
    starting_indices_not_padded : jnp.ndarray = None #[n_levels]
    level_log2_size             : jnp.ndarray = None #[n_levels]

    @classmethod
    def from_topological_sorted_angle(cls, topological_sorted_angle: TopologicalSortedAngle) -> 'TopologicalSortedAnglePrecomputedLevels':
        return cls(*precompute_levels_padded_pow2(topological_sorted_angle))






def _single_level(level_size : int, level_i : int, precomputed_levels : TopologicalSortedAnglePrecomputedLevels,
                  gij_inv_level : jnp.ndarray, qm_i_level : jnp.ndarray, dij_f_p_fm_level : jnp.ndarray, abs_omega_dot_n_Je_level : jnp.ndarray, 
                  etoe : jnp.ndarray, sweep_result : jnp.ndarray):
    '''
    Perform a single level sweep in parallel. This assumes the elements in the level are independent.
    The level_size should be a power of two for compilation efficiency. This can be padded with dummy elements, see _setup_levels_padded_pow2.
    Note: compared to a normal sweep, this requires an extra dummy element in the sweep_result array. This dummy element is set by padded elements.

    Parameters:
    -----------------------
    level_size          : int
        Number of elements in the level (should be a power of two for compilation efficiency) [this is necessarily equal to precomputed_levels.sizes[level_i] but passed here because it allows us to precompile for different level sizes]
        However, we cannot check this during compilation, so it is assumed the precomputed levels are constructed correctly (otherwise you will index beyond the current level, which is not out of bound but will produce incorrect results)
    level_i            : int
        Index of the level to be swept
    precomputed_levels : TopologicalSortedAnglePrecomputedLevels
        Precomputed levels data structure    
    gij_inv_level      : jnp.ndarray
        Precomputed inverse transport operator for the elements in the level [level_size, n_basis, n_basis] (also padded)
    qm_i_level        : jnp.ndarray
        Precomputed total source for the elements in the level [level_size, n_basis] (also padded)
    dij_f_p_fm_level  : jnp.ndarray
        Precomputed incoming flux coefficients for the elements in the level [level_size, n_faces, n_basis] (also padded)
    abs_omega_dot_n_Je_level : jnp.ndarray
        Absolute value of the dot product of the angle and the element face normals for the elements in the level [level_size, n_faces] (also padded)
    etoe              : jnp.ndarray
        Element to element connectivity array [n_elements, n_faces] (with -1 for boundary faces). From GeometricDomain.
    sweep_result     : jnp.ndarray
        Current sweep result array [n_elements + 1, n_basis]
    '''
    assert level_size == gij_inv_level.shape[0] == qm_i_level.shape[0] == dij_f_p_fm_level.shape[0] == abs_omega_dot_n_Je_level.shape[0], \
        "Input arrays must have the same number of elements in the level but got: level_size {}, gij_inv_level {}, qm_i_level {}, dij_f_p_fm_level {}, abs_omega_dot_n_Je_level {}".format(\
            level_size, gij_inv_level.shape[0], qm_i_level.shape[0], dij_f_p_fm_level.shape[0], abs_omega_dot_n_Je_level.shape[0])    
    assert sweep_result.shape[0] == etoe.shape[0] + 1, \
        "Sweep result array must have one extra element compared to etoe (number of elements) but got: sweep_result {}, etoe {}".format(sweep_result.shape[0], etoe.shape[0])
    
    level_start                = precomputed_levels.starting_indices_padded[level_i]
    level_indices              = jax.lax.dynamic_slice(precomputed_levels.padded_levels, (level_start,), (level_size,)) # Get the indices for this level [level_size] (elem_idx in geometric domain)
    s_dj                       = _collapse_dij_vmap(level_indices, dij_f_p_fm_level, sweep_result, etoe, abs_omega_dot_n_Je_level)        
    result                     = jnp.einsum("mij, mj->mi", gij_inv_level, (qm_i_level + s_dj))    
    return sweep_result.at[level_indices].set(result)

def _single_level_from_complete_vmap(level_size : int, level_i : int, precomputed_levels : TopologicalSortedAnglePrecomputedLevels,
                                      gij_inv_all, qm_i_all, dij_f_p_fm_all, abs_omega_dot_n_Je_all, 
                                      etoe, sweep_result):
    ''' 
    Perform a single level sweep in parallel. This assumes the elements in the level are independent. Here, we pass the entire vmap array (over all elements) (as computed by a vmap over all elements in topological order) and slice the relevant level data.
    '''    
    level_start_not_padded     = precomputed_levels.starting_indices_not_padded[level_i] #[starting index of the level in the non-padded array (for indexing in the complete vmap arrays)]    
    indices_desired            = jnp.arange(0,level_size) +  level_start_not_padded 
    
    safe_mask                  = indices_desired < gij_inv_all.shape[0]
    indices_safe               = jnp.minimum(indices_desired, gij_inv_all.shape[0]) 
    gij_inv_safe               = jnp.where(safe_mask[:, None, None],       gij_inv_all[indices_safe],            0)
    qm_i_safe                  = jnp.where(safe_mask[:, None],             qm_i_all[indices_safe],               0)
    dij_f_p_fm_safe            = jnp.where(safe_mask[:, None, None, None], dij_f_p_fm_all[indices_safe],         0)
    abs_omega_dot_n_Je_safe    = jnp.where(safe_mask[:, None],             abs_omega_dot_n_Je_all[indices_safe], 0)
    
    return _single_level(level_size, level_i, precomputed_levels, gij_inv_safe, qm_i_safe, dij_f_p_fm_safe, abs_omega_dot_n_Je_safe, etoe, sweep_result)

from typing import Callable
def transport_inverse_sweep_complete_vmap_levels(geometric_domain           : GeometricDomain, 
                                                 fem_matrices               : FEMMatrices, 
                                                 precomputed_geometric_data : PrecomputedGeometricData, 
                                                 cross_sections             : CrossSectionDataSingleGroup,                                                
                                                 topological_sorted_angle   : TopologicalSortedAnglePrecomputedLevels, 
                                                 precomputed_source         : Source, 
                                                 sweep_result               : jnp.ndarray, 
                                                 sweep_level_f_sizes        : Tuple[Callable]) -> jnp.ndarray:    
    gij_inv_all, qm_i_all, dij_f_p_fm_all, _, abs_omega_dot_n_Je = _setup_elements_idx_in_precomp_with_omega_n_calculation(topological_sorted_angle.levels_argsort, 
                                                                    geometric_domain, fem_matrices,
                                                                    precomputed_geometric_data, cross_sections, topological_sorted_angle.angle,
                                                                    precomputed_source)            
    def sweep_level(sweep_result, level_i):
        sweep_result = jax.lax.switch(topological_sorted_angle.level_log2_size[level_i], sweep_level_f_sizes, level_i, topological_sorted_angle, gij_inv_all, qm_i_all, dij_f_p_fm_all, abs_omega_dot_n_Je, geometric_domain.etoe, sweep_result)
        return sweep_result, None    
    return jax.lax.scan(sweep_level, sweep_result, jnp.arange(topological_sorted_angle.sizes.shape[0]))[0]



@partial(jax.jit, static_argnums = (7,))
def transport_inverse_sweep_complete_vmap_levels_max(geometric_domain       : GeometricDomain, 
                                                 fem_matrices               : FEMMatrices, 
                                                 precomputed_geometric_data : PrecomputedGeometricData, 
                                                 cross_sections             : CrossSectionDataSingleGroup,                                                
                                                 topological_sorted_angle   : TopologicalSortedAnglePrecomputedLevels, 
                                                 precomputed_source         : Source, 
                                                 sweep_result               : jnp.ndarray, 
                                                 max_levels_log2            : int) -> jnp.ndarray: 
    callables = tuple([partial(_single_level_from_complete_vmap, int(2**i)) for i in range(max_levels_log2+1)])        
    return transport_inverse_sweep_complete_vmap_levels(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angle, precomputed_source, sweep_result, callables)



#===--------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Level based parallelism setup functions
#===--------------------------------------------------------------------------------------------------------------------------------------------------------------
def precompute_levels_padded_pow2(topological_sorted_angle: TopologicalSortedAngle):
    padded_levels, sizes, unique_sizes = _setup_levels_padded_pow2_angle(topological_sorted_angle)
    starting_indices_padded            = jnp.concatenate([jnp.array([0]), jnp.cumsum(sizes)[:-1]])
    starting_indices_not_padded        = jnp.concatenate([jnp.array([0]), jnp.cumsum(jnp.bincount(topological_sorted_angle.levels))[:-1]])            
    level_log2_size                    = jnp.ceil(jnp.log2(sizes)).astype(jnp.int32)

    return topological_sorted_angle.levels, topological_sorted_angle.angle, topological_sorted_angle.levels_argsort, jnp.array(padded_levels),  jnp.array(sizes),  jnp.array(unique_sizes), starting_indices_padded, starting_indices_not_padded, jnp.array(level_log2_size)

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
