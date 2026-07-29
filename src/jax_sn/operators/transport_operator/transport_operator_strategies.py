'''
Strategy objects for the transport operator. Here, we have subtypes that compose from a inner single-angle strategy.
'''

import equinox as eqx
from abc import abstractmethod
from typing import Type, Tuple
from jax_sn.operators.base_linear_operator import Strategy
from .sweep_data_implementations import BasicSweepData, SweepData
from .transport_data import TransportData
from .transport_operator_base import (
    AbstractInversionStrategy,
    MultiAngleForwardStrategy, MultiAngleInversionStrategy,
    SingleAngleForwardStrategy, SingleAngleInversionStrategy,
)


class ComposedMultiAngleForwardStrategy(MultiAngleForwardStrategy):
    '''A multi-angle forward strategy that is a combinator over a single-angle strategy.'''

    inner: SingleAngleForwardStrategy

    def single_angle_strategy(self) -> SingleAngleForwardStrategy:
        return self.inner

class ComposedMultiAngleInversionStrategy(MultiAngleInversionStrategy):
    '''A multi-angle inversion strategy that is a combinator over a single-angle strategy.'''

    inner: SingleAngleInversionStrategy

    def single_angle_strategy(self) -> SingleAngleInversionStrategy:
        return self.inner

    def required_capabilities(self) -> Tuple[type, ...]:
        return self.inner.required_capabilities()

    def default_sweep_data_type(self) -> Type[SweepData]:
        return self.inner.default_sweep_data_type()

    def check_compatible(self, data: TransportData):
        self.inner.check_compatible(data)
