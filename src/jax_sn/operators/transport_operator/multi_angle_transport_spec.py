from jax_sn.operators.operator_data import OperatorData
from .transport_operator_interface import TransportOperatorMultiAngle
from .transport_operator_strategies import SingleAngleForwardStrategy, MultiAngleInversionStrategy
import equinox as eqx

class MultiAngleTransportSpec(eqx.Module):
    r'''
    Configuration of a multi-angle transport operator: purely the strategies (which forward kernel,
    which multi-angle inversion). The energy group is **not** part of the spec -- it is a build
    parameter, since the same strategies build the operator for any group.
    '''

    forward_strategy   : SingleAngleForwardStrategy
    '''How to apply the single-angle forward operator. Vmapped to multile angles'''

    multi_angle_inversion_strategy    : MultiAngleInversionStrategy
    '''How to apply multi-angle version. Fused kernels are different than a Vmap of a single-angle version.'''

    def build(self, operator_data : OperatorData, energy_group : int) -> TransportOperatorMultiAngle:
        '''Construct the multi-angle transport operator for this spec, group and :class:`OperatorData`.'''
        return TransportOperatorMultiAngle(
            operator_data.spatial_data,
            operator_data.sweep_data,
            operator_data.angular_data.quadrature_set.angles,
            energy_group,
            self.forward_strategy,
            self.multi_angle_inversion_strategy,
        )

    @classmethod
    def from_transport_operator(cls, transport_operator : TransportOperatorMultiAngle) -> "MultiAngleTransportSpec":
        '''Extract the (data-free) spec from an existing operator (its energy group is dropped -- the spec
        is group-agnostic).'''
        return cls(
            transport_operator.forward_strategy,
            transport_operator.multi_angle_inversion_strategy,
        )
