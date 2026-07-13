'''
Strategy objects for the transport operator.
'''

import equinox as eqx
from abc import abstractmethod
from typing import TYPE_CHECKING, Type, Tuple, Optional
from .sweep_data_implementations import BasicSweepData, SweepData
from .transport_data import ForwardData, InversionData
if TYPE_CHECKING:
    from .transport_operator_interface import TransportOperatorSingleAngle


class SingleAngleForwardStrategy(eqx.Module):
    '''How to apply the single-angle forward operator.'''

    @abstractmethod
    def forward(self, data: ForwardData, x):
        ...

class SingleAngleInversionStrategy(eqx.Module):
    '''
    How to sweep-invert the single-angle operator. 
    '''

    @classmethod
    def required_capabilities(cls) -> Tuple[type, ...]:
        '''Sweep-data capability interfaces this strategy requires (e.g. ``(ProvidesLevels,)``). Empty
        (the default) means any :class:`~jax_sn.operators.transport_operator.SweepData` is acceptable.'''
        return ()

    @classmethod
    @abstractmethod
    def vmappable(cls) -> bool:
        '''Whether an outer ``vmap`` over this inversion vectorises.'''
        ...

    @abstractmethod
    def invert(self, data: InversionData, b):
        ...

    @classmethod
    def default_sweep_data_type(cls) -> Type[SweepData]:
        '''The concrete, minimal sweep-data type a creation helper should build for this strategy.
        Default: the plain sort (provides no extra capability).'''
        return BasicSweepData
    
    def check_compatible(self, operator: "TransportOperatorSingleAngle"):
        '''Assert the operator's sweep data provides every capability in :meth:`required_capabilities`.
        Permissive: a richer sweep data (more capabilities) passes. Run from
        ``TransportOperatorSingleAngle.__check_init__``, so it fires at construction and on every ``eqx.tree_at``.'''
        for capability in self.required_capabilities():
            assert isinstance(operator.sweep_data, capability), \
                f"{type(self).__name__} requires sweep data providing {capability.__name__}, " \
                f"but got {type(operator.sweep_data).__name__}. Build it via " \
                f"{type(self).__name__}.default_sweep_data_type().from_domain_quadrature_set(...)."

class MultiAngleInversionStrategy(eqx.Module):
    '''
    How to sweep-invert over the angle axis. Consumes a batched :class:`InversionData` (angle axis leading).
    '''

    @abstractmethod
    def invert(self, data: InversionData, b):
        '''Invert over the angle axis: ``data`` is a batched :class:`InversionData` (angle axis leading).'''
        ...

    @abstractmethod
    def single_angle_strategy(self) -> SingleAngleInversionStrategy:
        '''The single-angle :class:`SingleAngleInversionStrategy` this angle inversion corresponds to (used for composited strategy, error if does not exist)'''
        ...

    @abstractmethod
    def required_capabilities(self) -> Tuple[type, ...]:
        '''Sweep-data capability interfaces this multi-angle inversion requires (see
        :meth:`SingleAngleInversionStrategy.required_capabilities`).'''
        ...

    @abstractmethod
    def default_sweep_data_type(self) -> Type[SweepData]:
        '''The concrete, minimal sweep-data type a creation helper should build for this strategy.'''
        ...

    @abstractmethod
    def check_compatible(self, operator):
        ...


class ComposedMultiAngleInversionStrategy(MultiAngleInversionStrategy):
    '''A multi-angle inversion strategy that is a combinator over a single-angle strategy.'''

    inner: SingleAngleInversionStrategy

    def single_angle_strategy(self) -> SingleAngleInversionStrategy:
        return self.inner

    def required_capabilities(self) -> Tuple[type, ...]:
        return self.inner.required_capabilities()

    def default_sweep_data_type(self) -> Type[SweepData]:
        return self.inner.default_sweep_data_type()

    def check_compatible(self, operator):
        self.inner.check_compatible(operator)
