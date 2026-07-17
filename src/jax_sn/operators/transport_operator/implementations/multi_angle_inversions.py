'''
Composed multi-angle strategies, which lift a single-angle strategy over the angle axis. Both axes
(forward and inverse sweep) get the same two lifts: ``vmap`` and ``lax.map``.
'''
import jax
import equinox as eqx
from ..transport_data import TransportData
from ..transport_operator_strategies import (
    ComposedMultiAngleForwardStrategy, ComposedMultiAngleInversionStrategy,
)

def _scan_over_angles(data: TransportData, leading, fn):
    '''``lax.map`` ``fn`` over the angle axis, serialising it (one angle live at a time).'''
    def one_angle(angle_sweep_v):
        angle_i, sweep_i, v_i = angle_sweep_v
        return fn(eqx.tree_at(lambda a : [a.angle, a.sweep_data], data, (angle_i, sweep_i)), v_i)
    return jax.lax.map(one_angle, (data.angle, data.sweep_data, leading))

def map_over_angles(data: TransportData, leading, fn):
    '''vmap closure over angle & sweep data. Closure because we don't know a priori other arguments (i.e. batching.)'''
    def one_angle(angle_i, sweep_i, v_i):
        return fn(eqx.tree_at(lambda a : [a.angle, a.sweep_data], data, (angle_i, sweep_i)), v_i)
    return eqx.filter_vmap(one_angle, in_axes=(0, 0, 0))(data.angle, data.sweep_data, leading)



# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
class VmapAnglesForward(ComposedMultiAngleForwardStrategy):
    '''Lift the single-angle forward by ``vmap``-ing it over the angle axis.'''

    def forward(self, data: TransportData, x):
        return map_over_angles(data, x, self.inner.forward)


class ScanAnglesForward(ComposedMultiAngleForwardStrategy):
    '''Lift the single-angle forward by ``lax.map``-ing it over the angle axis.'''

    def forward(self, data: TransportData, x):
        return _scan_over_angles(data, x, self.inner.forward)


# ---------------------------------------------------------------------------
# Inverse sweep
# ---------------------------------------------------------------------------
class VmapAngles(ComposedMultiAngleInversionStrategy):
    '''Lift the single-angle inverse sweep by ``vmap``-ing it over the angle axis.'''

    def __check_init__(self):
        assert self.inner.vmappable(), \
            f"VmapAngles over {type(self.inner).__name__} would silently serialise (an opaque kernel is " \
            f"invisible to vmap); use ScanAngles({type(self.inner).__name__}()) or the kernel's fused native lift."

    def invert(self, data: TransportData, b):
        return map_over_angles(data, b, self.inner.invert)


class ScanAngles(ComposedMultiAngleInversionStrategy):
    '''Lift the single-angle inverse sweep by ``lax.map``-ing it over the angle axis.'''

    def invert(self, data: TransportData, b):
        return _scan_over_angles(data, b, self.inner.invert)
