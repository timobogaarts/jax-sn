from jax_sn.solution_domain import SolutionDomain
import jax.numpy as jnp
from jax_sn.operators.transport_operator import SingleAngleForwardStrategy, SingleAngleInversionStrategy, MultiAngleForwardStrategy, MultiAngleInversionStrategy,TransportOperatorSingleAngle, TransportOperatorMultiAngle, TransportData
from .builders import Builder
from jax_sn.operators.sweep_policy import Resort

from typing import Dict, Union

def create_transport_operator(solution_domain : SolutionDomain, angle : jnp.ndarray, forward_strategy : Union[SingleAngleForwardStrategy, MultiAngleForwardStrategy],
                              inversion_strategy : Union[SingleAngleInversionStrategy, MultiAngleInversionStrategy], energy_group : int, sweep_data_kwargs : Dict = {}) -> Union[TransportOperatorSingleAngle, TransportOperatorMultiAngle]:
    '''
    Creates a transport operator (single- or multi-angle, dispatched on ``angle.ndim``) for the given
    solution domain and angle(s), using the default sweep data type of the given strategies.

    Thin adapter over :meth:`Builder.transport_operator`; no quadrature set is
    needed, since the operator sweeps the given ``angle``.
    '''
    builder = Builder.from_solution_domain(solution_domain)
    return builder.transport_operator(angle, forward_strategy, inversion_strategy, energy_group,
                                      Resort(**sweep_data_kwargs))
