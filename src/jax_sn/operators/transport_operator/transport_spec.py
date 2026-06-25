import equinox as eqx

from .solution_setup import SolutionSetup
from .transport_operator_interface import TransportOperator, TransportOperatorMultipleAngles
from .transport_operator_strategies import (
    ForwardStrategy, InversionStrategy, AngleLifting,    
)

class TransportSpec(eqx.Module):
    r'''
    Configuration of a multi-angle transport operator. Contains no data; only strategies (how to invert single angle and how to 
    lift to multiple angles) and the energy group index. The operator's data is provided by a :class:`SolutionSetup` 
    '''

    forward_strategy   : ForwardStrategy 
    '''How to apply the single-angle forward operator. See :mod:`.transport_operator_strategies`.'''

    inversion_strategy : InversionStrategy
    '''How to sweep-invert the single-angle operator.'''

    angle_lifting      : AngleLifting
    '''How to lift the single-angle operator over the angle axis.'''

    energy_group       : int = 0
    '''Energy group index for which the transport operator is defined.'''

    def build(self, setup : SolutionSetup) -> TransportOperatorMultipleAngles:
        '''Construct the multi-angle transport operator for this spec on the given :class:`SolutionSetup`.

        The transport direction is injected from ``setup.angular_setup`` (the single home of the angle);
        the sweep data carries only the operator-agnostic ordering.'''
        single_angle = TransportOperator(
            setup.solution_domain_data,
            setup.sweep_data,
            setup.angular_setup.quadrature_set.angles,
            self.energy_group,
            self.forward_strategy,
            self.inversion_strategy,
        )
        return TransportOperatorMultipleAngles(single_angle, self.angle_lifting)

    @classmethod
    def from_transport_operator(cls, transport_operator : TransportOperatorMultipleAngles, energy_group : int = None) -> "TransportSpec":
        '''Extract the (data-free) spec from an existing operator. The operator's data goes into a :class:`SolutionSetup`.'''
        single_angle = transport_operator.transport_operator
        return cls(
            single_angle.forward_strategy,
            single_angle.inversion_strategy,
            transport_operator.angle_lifting,
            single_angle.energy_group if energy_group is None else energy_group,
        )

    def to_energy_group(self, energy_group : int) -> "TransportSpec":
        '''Return this spec for a different energy group (a cheap integer swap).'''
        return eqx.tree_at(lambda spec: spec.energy_group, self, energy_group)
