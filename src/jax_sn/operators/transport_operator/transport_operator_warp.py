'''
Warp strategies and convenience constructors for the transport operator.

Warp provides two inverse-sweep families -- a sequential sweep and a level-based-parallelism sweep --
each with a single-angle kernel (an :class:`InversionStrategy`) and a *fused* multi-angle kernel (an
:class:`AngleLifting`). The fused multi-angle inverse is not a vmap of the single-angle one; it is a
dedicated kernel tied to the warp data layout, so it lives in an ``AngleLifting`` whose ``forward``
still defers to the generic :class:`VmapAngles`.

The warp forward is the batched forward (``BatchedForward``), so warp single-angle operators are just
``TransportOperator`` with a batched forward strategy + a warp inversion strategy -- no special
subclass. The level-parallel variant carries its level setup inside a
:class:`TopologicalSortedAngleLevelParallel` toposort.
'''

from jax_sn.domain import TopologicalSortedAngleBase, TopologicalSortedAngleLevelParallel
from jax_sn.solution_domain import SolutionDomainData
from typing import Union

from .transport_operator_base import TransportOperator, TransportOperatorMultipleAngles
from .transport_operator_strategies import InversionStrategy, AngleLifting, BatchedForward, VmapAngles
from .linear_tetrahedron_warp_kernel import (
    single_angle_warp_sweep, multi_angle_warp_sweep,
    level_single_angle_warp_sweep, level_multi_angle_warp_sweep,
    NBASIS, NFACES,
)


def _assert_warp_compatible(solution_domain_data: SolutionDomainData):
    '''Validate that the solution-domain data matches what the warp tetrahedron kernels expect.'''
    assert solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape[-2:] == (3, 3), \
        f"For Warp kernel, expected vol_jacobians_inv shape (3,3), got {solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape}"
    assert solution_domain_data.n_basis == NBASIS, \
        f"For warp kernel, expected {NBASIS} basis functions, got {solution_domain_data.n_basis}"
    assert solution_domain_data.domain.etoe.shape[-1] == NFACES, \
        f"For warp kernel, expected {NFACES} faces per elements, got {solution_domain_data.domain.etoe.shape[-1]}"


# ---------------------------------------------------------------------------
# Inversion strategies (single angle)
# ---------------------------------------------------------------------------
class WarpSequentialSweep(InversionStrategy):
    '''Single-angle inverse via the sequential Warp sweep kernel (no batching).'''

    def __call__(self, operator, b):
        return single_angle_warp_sweep(*operator._sweep_data(), b)


class WarpLevelSweep(InversionStrategy):
    '''Single-angle inverse via the level-based-parallelism Warp sweep kernel.'''

    def __call__(self, operator, b):
        return level_single_angle_warp_sweep(*operator._sweep_data(), b, operator.topological_sorted_angle.level_fwd)


# ---------------------------------------------------------------------------
# Angle lifting (fused multi-angle warp kernels; forward stays vmap)
# ---------------------------------------------------------------------------
class WarpSequentialAngles(AngleLifting):
    '''Multi-angle lift: fused sequential Warp sweep for the inverse, vmap for the forward.'''

    def forward(self, single_op, x):
        return VmapAngles().forward(single_op, x)

    def invert(self, single_op, b):
        return multi_angle_warp_sweep(*single_op._sweep_data(), b)


class WarpLevelAngles(AngleLifting):
    '''Multi-angle lift: fused level-based-parallelism Warp sweep for the inverse, vmap for the forward.'''

    def forward(self, single_op, x):
        return VmapAngles().forward(single_op, x)

    def invert(self, single_op, b):
        return level_multi_angle_warp_sweep(*single_op._sweep_data(), b, single_op.topological_sorted_angle.level_fwd)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------
def TransportOperatorSequentialWarpSingleAngle(solution_domain_data: SolutionDomainData, topological_sorted_angle: TopologicalSortedAngleBase, energy_group: int, batch_size: int) -> TransportOperator:
    '''Single-angle operator: batched forward + sequential Warp inverse sweep.'''
    _assert_warp_compatible(solution_domain_data)
    return TransportOperator(solution_domain_data, topological_sorted_angle, energy_group, BatchedForward(batch_size), WarpSequentialSweep())


def TransportOperatorLevelsWarpSingleAngle(solution_domain_data: SolutionDomainData, topological_sorted_angle: TopologicalSortedAngleBase, energy_group: int, batch_size: int, max_n_levels: Union[int, None] = None) -> TransportOperator:
    '''
    Single-angle operator: batched forward + level-based-parallelism Warp inverse sweep. Builds the
    :class:`TopologicalSortedAngleLevelParallel` toposort (which carries the level setup for both sweep
    directions) from the given plain toposort.
    '''
    _assert_warp_compatible(solution_domain_data)
    level_parallel_toposort = TopologicalSortedAngleLevelParallel.from_topological_sorted_angle(topological_sorted_angle, max_n_levels)
    return TransportOperator(solution_domain_data, level_parallel_toposort, energy_group, BatchedForward(batch_size), WarpLevelSweep())


def TransportOperatorSequentialWarpMultipleAngles(transport_operator: TransportOperator) -> TransportOperatorMultipleAngles:
    '''Multi-angle operator using the fused sequential Warp sweep.'''
    return TransportOperatorMultipleAngles(transport_operator, WarpSequentialAngles())


def TransportOperatorLevelsWarpMultipleAngles(transport_operator: TransportOperator) -> TransportOperatorMultipleAngles:
    '''Multi-angle operator using the fused level-based-parallelism Warp sweep.'''
    return TransportOperatorMultipleAngles(transport_operator, WarpLevelAngles())
