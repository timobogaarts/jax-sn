'''
Transport operator implementations; both single angle and multi-angle. 


transport_operator_base defines the single-angle and multi-angle lineax operators, strategies, and data.
'''

from .sweep_data_implementations import SweepData, BasicSweepData, LeveledSweepData, LevelBasedParallelism, ProvidesLevels

from .transport_data import TransportData
from .transport_operator_base import (
    AbstractTransportOperator, TransportOperatorSingleAngle, TransportOperatorMultiAngle,
)

from .transport_operator_strategies import (
    SingleAngleForwardStrategy, MultiAngleForwardStrategy, ComposedMultiAngleForwardStrategy,
    AbstractInversionStrategy,
    SingleAngleInversionStrategy, MultiAngleInversionStrategy, ComposedMultiAngleInversionStrategy,
)
from .implementations.transport_operator_forward import VmapForward, BatchedForward
from .implementations.transport_operator_jax import VmapSweep, BatchedSweep
from .implementations.multi_angle_inversions import VmapAnglesInversion, ScanAnglesInversion, VmapAnglesForward, ScanAnglesForward


from .implementations.transport_operator_pallas import (
    VmapPallasSweep, BatchedPallasSweep,
)


# Warp is an optional extra (`pip install jax-sn[warp]`); without it these strategies are absent.
try:
    from .implementations.transport_operator_warp import (
        WarpSequentialSweep, WarpLevelSweep, WarpSequentialAngles, WarpLevelAngles,
    )
except ImportError:
    _WARP = []
else:
    _WARP = ['WarpSequentialSweep', 'WarpLevelSweep', 'WarpSequentialAngles', 'WarpLevelAngles']

__all__ = [
    'SweepData',
    'BasicSweepData',
    'LeveledSweepData',
    'LevelBasedParallelism',
    'ProvidesLevels',
    'TransportData',
    'AbstractTransportOperator',
    'TransportOperatorSingleAngle',
    'TransportOperatorMultiAngle',
    'SingleAngleForwardStrategy',
    'MultiAngleForwardStrategy',
    'ComposedMultiAngleForwardStrategy',
    'AbstractInversionStrategy',
    'SingleAngleInversionStrategy',
    'MultiAngleInversionStrategy',
    'ComposedMultiAngleInversionStrategy',
    'VmapForward',
    'BatchedForward',
    'VmapSweep',
    'BatchedSweep',
    'VmapAnglesInversion',
    'ScanAnglesInversion',
    'VmapAnglesForward',
    'ScanAnglesForward',
    'VmapPallasSweep',
    'BatchedPallasSweep',
] + _WARP
