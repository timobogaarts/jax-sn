'''
Warp strategies and convenience constructors for the transport operator.
'''

from ..sweep_data_implementations import BasicSweepData, LeveledSweepData, ProvidesLevels
from jax_sn.solution_domain import SpatialData

from ..transport_operator_strategies import SingleAngleInversionStrategy, MultiAngleInversionStrategy
from ..transport_data import TransportData
from .linear_tetrahedron_warp_kernel import (
    single_angle_warp_sweep, multi_angle_warp_sweep,
    level_single_angle_warp_sweep, level_multi_angle_warp_sweep,
    NBASIS, NFACES,
)

def _assert_warp_compatible(spatial_data: SpatialData):
    '''Validate that the solution-domain data matches what the warp tetrahedron kernels expect.'''
    assert spatial_data.precomputed_geometric_data.vol_jacobians_inv.shape[-2:] == (3, 3), \
        f"For Warp kernel, expected vol_jacobians_inv shape (3,3), got {spatial_data.precomputed_geometric_data.vol_jacobians_inv.shape}"
    assert spatial_data.n_basis == NBASIS, \
        f"For warp kernel, expected {NBASIS} basis functions, got {spatial_data.n_basis}"
    assert spatial_data.domain.etoe.shape[-1] == NFACES, \
        f"For warp kernel, expected {NFACES} faces per elements, got {spatial_data.domain.etoe.shape[-1]}"

# ---------------------------------------------------------------------------
# Inversion strategies (single angle)
# ---------------------------------------------------------------------------
class WarpSequentialSweep(SingleAngleInversionStrategy):
    '''Single-angle inverse via the sequential Warp sweep kernel (no batching).'''

    def invert(self, data: TransportData, b):
        return single_angle_warp_sweep(*data.inversion_args, b)

    def check_compatible(self, data):
        _assert_warp_compatible(data.spatial_data)

    @classmethod
    def vmappable(cls):
        return False   # opaque warp kernel: an outer vmap serialises

    @classmethod
    def native_multi_angle(cls):
        return WarpSequentialAngles   # the fused multi-angle kernel, not a vmap of this one


class WarpLevelSweep(SingleAngleInversionStrategy):
    '''
    Single-angle inverse via the level-based-parallelism Warp sweep kernel. Requires sweep data
    providing the :class:`~jax_sn.operators.transport_operator.ProvidesLevels` capability (the level setup for both sweep
    directions). The base :meth:`~SingleAngleInversionStrategy.check_compatible` enforces that capability; this
    strategy only adds the warp-data-layout check. The minimal concrete builder is
    :class:`~jax_sn.operators.transport_operator.LeveledSweepData`.
    '''

    def invert(self, data: TransportData, b):
        return level_single_angle_warp_sweep(*data.inversion_args, b, data.sweep_data.level_fwd)

    @classmethod
    def required_capabilities(cls):
        return (ProvidesLevels,)

    @classmethod
    def vmappable(cls):
        return False   # opaque warp kernel: an outer vmap serialises

    @classmethod
    def default_sweep_data_type(cls):
        return LeveledSweepData

    @classmethod
    def native_multi_angle(cls):
        return WarpLevelAngles   # the fused multi-angle kernel, not a vmap of this one

    def check_compatible(self, data):
        super().check_compatible(data)  # enforces required_capabilities (ProvidesLevels)
        _assert_warp_compatible(data.spatial_data)


# ---------------------------------------------------------------------------
# Native multi-angle angle inversions (fused warp kernels; forward always vmaps)
# ---------------------------------------------------------------------------
# These are *native* AngleInversions: they own no inner SingleAngleInversionStrategy, they invoke a fused
# multi-angle kernel directly. single_angle_strategy() names the matching single-angle kernel, which is
# both what the wrapped single-angle operator is built with (so the pairing holds by construction) and
# what resolves the required sweep-data capabilities via MultiAngleInversionStrategy's delegation.
class WarpSequentialAngles(MultiAngleInversionStrategy):
    '''Native multi-angle inverse: fused sequential Warp sweep.'''

    def single_angle_strategy(self):
        return WarpSequentialSweep()

    def invert(self, data: TransportData, b):
        return multi_angle_warp_sweep(*data.inversion_args, b)

    def check_compatible(self, data):
        _assert_warp_compatible(data.spatial_data)


class WarpLevelAngles(MultiAngleInversionStrategy):
    '''Native multi-angle inverse: fused level-based-parallelism Warp sweep. Requires sweep data
    providing the :class:`~jax_sn.operators.transport_operator.ProvidesLevels` capability.'''

    def single_angle_strategy(self):
        return WarpLevelSweep()

    def invert(self, data: TransportData, b):
        return level_multi_angle_warp_sweep(*data.inversion_args, b, data.sweep_data.level_fwd)

    def required_capabilities(self):
        return (ProvidesLevels,)

    def default_sweep_data_type(self):
        return LeveledSweepData

    def check_compatible(self, data):
        super().check_compatible(data)  # enforces required_capabilities (ProvidesLevels)
        _assert_warp_compatible(data.spatial_data)
