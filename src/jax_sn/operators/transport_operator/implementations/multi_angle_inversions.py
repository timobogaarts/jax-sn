'''
Composed MultiAngleInversionStrategy implementations, which lift a single-angle strategy to a multi-angle one.
'''
import jax

from ..transport_operator_interface import _map_over_angles
from ..transport_data import InversionData
from ..transport_operator_strategies import ComposedMultiAngleInversionStrategy

class VmapAngles(ComposedMultiAngleInversionStrategy):
    '''Lift the single-angle inverse sweep by ``vmap``-ing it over the angle axis.'''

    def __check_init__(self):
        assert self.inner.vmappable(), \
            f"VmapAngles over {type(self.inner).__name__} would silently serialise (an opaque kernel is " \
            f"invisible to vmap); use ScanAngles({type(self.inner).__name__}()) or the kernel's fused native lift."

    def invert(self, data: InversionData, b):
        return _map_over_angles(data, b, lambda d, v: self.inner.invert(d, v))

class ScanAngles(ComposedMultiAngleInversionStrategy):
    '''Lift the single-angle inverse sweep by ``lax.map``-ing it over the angle axis.'''

    def invert(self, data: InversionData, b):
        def one_angle(angle_sweep_b):
            angle_i, sweep_i, b_i = angle_sweep_b
            return self.inner.invert(data._replace(angle=angle_i, sweep_data=sweep_i), b_i)
        return jax.lax.map(one_angle, (data.angle, data.sweep_data, b))