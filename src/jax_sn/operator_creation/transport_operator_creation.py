from jax_sn.solution_domain import SolutionDomain
import jax.numpy as jnp
from jax_sn.operators.transport_operator import SingleAngleForwardStrategy, SingleAngleInversionStrategy, MultiAngleForwardStrategy, MultiAngleInversionStrategy,TransportOperatorSingleAngle, TransportOperatorMultiAngle, TransportData

from typing import Dict, Union

def create_transport_operator(solution_domain : SolutionDomain, angle : jnp.ndarray, forward_strategy : Union[SingleAngleForwardStrategy, MultiAngleForwardStrategy], 
                              inversion_strategy : Union[SingleAngleInversionStrategy, MultiAngleInversionStrategy], energy_group : int, sweep_data_kwargs : Dict = {}) -> Union[TransportOperatorSingleAngle, TransportOperatorMultiAngle]:
    '''
    Creates a transport operator (single- or multi-angle, dispatched on ``angle.ndim``) for the given
    solution domain and angle(s), using the default sweep data type of the given strategies.
    '''
    if angle.ndim == 1:
        assert angle.shape[0] == 3, "For single angle transport operator, angle must be a 1D array of shape (3,) but got {}".format(angle.shape)
        assert isinstance(forward_strategy, SingleAngleForwardStrategy), "For single angle transport operator, forward strategy must be a SingleAngleForwardStrategy but got {}".format(type(forward_strategy))
        assert isinstance(inversion_strategy, SingleAngleInversionStrategy), "For single angle transport operator, inversion strategy must be a SingleAngleInversionStrategy but got {}".format(type(inversion_strategy))
    elif angle.ndim == 2:
        assert angle.shape[1] == 3, "For multi angle transport operator, angle must be a 2D array of shape (N, 3) but got {}".format(angle.shape)
        assert isinstance(forward_strategy, MultiAngleForwardStrategy), "For multi angle transport operator, forward strategy must be a MultiAngleForwardStrategy but got {}".format(type(forward_strategy))
        assert isinstance(inversion_strategy, MultiAngleInversionStrategy), "For multi angle transport operator, inversion strategy must be a MultiAngleInversionStrategy but got {}".format(type(inversion_strategy))
    else:
        raise ValueError("Angle must be either a 1D array of shape (3,) for single angle or a 2D array of shape (N, 3) for multi angle but got {}".format(angle.shape))

    sweep_data = inversion_strategy.default_sweep_data_type().from_domain_angle(solution_domain.domain, angle, **sweep_data_kwargs)        
    transport_data = TransportData(solution_domain.spatial_data, energy_group, angle, sweep_data)

    if angle.ndim == 1:
        return TransportOperatorSingleAngle(transport_data, forward_strategy, inversion_strategy)
    else:
        return TransportOperatorMultiAngle(transport_data, forward_strategy, inversion_strategy)
        