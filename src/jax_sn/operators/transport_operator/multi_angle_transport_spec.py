import equinox as eqx

from jax_sn.operators.operator_data import OperatorData
from .transport_operator_interface import TransportOperatorMultiAngle
from .transport_operator_strategies import SingleAngleForwardStrategy, MultiAngleInversionStrategy, SingleAngleInversionStrategy

class MultiAngleTransportSpec(eqx.Module):
    r'''
    Configuration of a multi-angle transport operator. Only contains strategies.
    '''

    forward_strategy   : SingleAngleForwardStrategy
    '''How to apply the single-angle forward operator. Vmapped to multile angles'''

    multi_angle_inversion_strategy    : MultiAngleInversionStrategy
    '''How to apply multi-angle version. Fused kernels are different than a Vmap of a single-angle version.'''

    energy_group       : int = 0
    '''Energy group index for which the transport operator is defined.'''

    def build(self, setup : OperatorData) -> TransportOperatorMultiAngle:
        '''Construct the multi-angle transport operator for this spec on the given :class:`OperatorData`.'''
        return TransportOperatorMultiAngle(
            setup.spatial_data,
            setup.sweep_data,
            setup.angular_data.quadrature_set.angles,
            self.energy_group,
            self.forward_strategy,
            self.multi_angle_inversion_strategy,
        )

    @classmethod
    def from_transport_operator(cls, transport_operator : TransportOperatorMultiAngle, energy_group : int = None) -> "MultiAngleTransportSpec":
        '''Extract the (data-free) spec from an existing operator.'''
        return cls(
            transport_operator.forward_strategy,
            transport_operator.multi_angle_inversion_strategy,
            transport_operator.energy_group if energy_group is None else energy_group,
        )

    def to_energy_group(self, energy_group : int) -> "MultiAngleTransportSpec":
        '''Return this spec for a different energy group (a cheap integer swap).'''
        return eqx.tree_at(lambda spec: spec.energy_group, self, energy_group)
