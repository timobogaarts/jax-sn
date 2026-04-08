
import jax.numpy as jnp
from .transport_operator_base import TransportOperatorBatched, TransportOperatorMultipleAngles, vmap_multiple_angles_forward

from .linear_tetrahedron_warp_kernel import single_angle_warp_sweep, multi_angle_warp_sweep, NBASIS, NFACES

class TransportOperatorSequentialWarpSingleAngle(TransportOperatorBatched):
    '''
    Transport operator for a single angle using a sequential sweep in Warp.

    Forward mode is using batched forward as in :class:`TransportOperatorBatched`, while inverse mode is using the single-angle sequential sweep kernel in Warp.
    The inverse does not use any batching.
    '''

    def __check_init__(self):
        assert self.solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape[-2:] == (3,3), f"For Warp kernel, expected vol_jacobians_inv shape (3,3), got {self.solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape}"
        assert self.solution_domain_data.n_basis == NBASIS, f"For warp kernel, expected {NBASIS} basis functions, got {self.solution_domain_data.n_basis}"
        assert self.solution_domain_data.domain.etoe.shape[-1] == NFACES, f"For warp kernel, expected {NFACES} faces per elements, got {self.solution_domain_data.domain.etoe.shape[-1]}"
    
    @property 
    def _inv_op_args(self):
        return single_angle_warp_sweep
    
    def _inv_args(self, b):
        return self._base_inv_args(b)
    
    @property
    def _vmap_for_inv_multiple_angles(self):
        return (None, None, None, None, 0,0)

class TransportOperatorSequentialWarpMultipleAngles(TransportOperatorMultipleAngles):
    '''
    Transport operator for multiple angles using vmap over angles.
    '''
    def __check_init__(self):
        assert isinstance(self.transport_operator, TransportOperatorSequentialWarpSingleAngle), f"Expected transport_operator to be a TransportOperatorSequentialWarpSingleAngle, got {type(self.transport_operator)}"

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_forward(self.transport_operator, x)
    
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return multi_angle_warp_sweep(*self._base_inv_args(b))
    


import equinox as eqx

from jax_sn.domain import TopologicalSortedAngle, LevelBasedParallelism
from jax_sn.solution_domain import SolutionDomainData
from typing import Union
from .linear_tetrahedron_warp_kernel import level_single_angle_warp_sweep, level_multi_angle_warp_sweep




class TransportOperatorLevelsWarpSingleAngle(TransportOperatorBatched):
    '''
    Transport operator for a single angle using a level-based parallelism sweep in Warp.

    We compute and store both the level-based parallelism for the original angle and for the reversed angle, so that we can use the same class for both forward and inverse mode by just swapping the levels.
    Forward mode is using batched forward as in :class:`TransportOperatorBatched`, while inverse mode is using the single-angle level-based parallelism sweep kernel in Warp.
    '''

    levelsetup         : LevelBasedParallelism    
    levelsetup_reverse : LevelBasedParallelism

    def __init__(self, solution_domain_data : SolutionDomainData, topological_sorted_angle : TopologicalSortedAngle, energy_group : int, batch_size : int, max_n_levels : Union[int, None] = None):
        self.solution_domain_data = solution_domain_data
        self.topological_sorted_angle = topological_sorted_angle
        self.energy_group = energy_group
        self.batch_size = batch_size
        self.levelsetup         = LevelBasedParallelism.from_topological_sorted_angle(topological_sorted_angle, max_n_levels)
        self.levelsetup_reverse = LevelBasedParallelism.from_topological_sorted_angle(topological_sorted_angle.reverse, max_n_levels)

    def transpose(self):
        return eqx.tree_at(
            lambda op: (op.topological_sorted_angle, op.levelsetup, op.levelsetup_reverse),
            self,
            (self.topological_sorted_angle.reverse, self.levelsetup_reverse, self.levelsetup),
        )

    @property
    def _inv_op_args(self):
        return level_single_angle_warp_sweep

    def _inv_args(self, b):
        return (*self._base_inv_args(b), self.levelsetup)
        
class TransportOperatorLevelsWarpMultipleAngles(TransportOperatorMultipleAngles):
    '''
    Transport operator for multiple angles using vmap over angles.
    '''
    def __check_init__(self):
        assert isinstance(self.transport_operator, TransportOperatorLevelsWarpSingleAngle), f"Expected transport_operator to be a TransportOperatorLevelsWarpSingleAngle, got {type(self.transport_operator)}"

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_forward(self.transport_operator, x)

    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return level_multi_angle_warp_sweep(*self.transport_operator._inv_args(b))
    
    def __getitem__(self, idx):
        '''Overloads for the level based parallelism transport operator.'''   
        return eqx.tree_at(lambda t_operator : [t_operator.topological_sorted_angle, t_operator.levelsetup, t_operator.levelsetup_reverse], self.transport_operator, 
                           [self.transport_operator.topological_sorted_angle[idx], self.transport_operator.levelsetup[idx], self.transport_operator.levelsetup_reverse[idx]])
    

    