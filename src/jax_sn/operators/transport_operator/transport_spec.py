import equinox as eqx

from .solution_setup import SolutionSetup
from .transport_operator_base import TransportOperator, TransportOperatorMultipleAngles
from .transport_operator_strategies import (
    ForwardStrategy, InversionStrategy, AngleLifting,
    VmapForward, VmapSweep, VmapAngles,
)


class TransportSpec(eqx.Module):
    r'''
    Per-operator *configuration* of a (multi-angle) transport operator -- the ``spec`` in the
    ``operator = setup + spec + strategy`` decomposition.

    This holds **no data**: only the strategy selections and the energy group. All precomputed data
    (spatial *and* the sweep execution plan) lives in the :class:`SolutionSetup` that a composite
    operator owns; :meth:`build` injects that setup to produce the operator. Keeping the spec
    data-free means a composite can hold and forward it without knowing any transport-strategy
    internals, and it partitions cleanly as a static pytree (no arrays) under ``jit``/``grad``.
    '''

    forward_strategy   : ForwardStrategy   = eqx.field(default_factory=VmapForward)
    '''How to apply the single-angle forward operator. See :mod:`.transport_operator_strategies`.'''

    inversion_strategy : InversionStrategy = eqx.field(default_factory=VmapSweep)
    '''How to sweep-invert the single-angle operator.'''

    angle_lifting      : AngleLifting      = eqx.field(default_factory=VmapAngles)
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
