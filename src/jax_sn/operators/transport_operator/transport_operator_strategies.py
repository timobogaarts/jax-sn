'''
Strategy objects for the transport operator.

The transport operator is parameterized by composable strategy objects rather than by subclassing:

- :class:`ForwardStrategy`   -- how to apply the (single-angle) forward operator ``mv``.
- :class:`InversionStrategy` -- how to sweep-invert the (single-angle) operator ``_invert_internal``.
- :class:`AngleLifting`      -- how to lift a single-angle operator to multiple angles.

A new method is a single small class implementing one action method named after that action:
``forward`` (forward strategies), ``invert`` (inversion strategies), or the ``forward``/``invert``
pair (lifting). Forward and inversion strategies are independent fields on the operator, so any
forward composes with any inversion (``F + I`` classes, not ``F x I`` operators).

This module holds only the *abstract* strategy bases. Each concrete implementation lives next to
its kernel under ``implementations/`` (e.g. ``implementations/transport_operator_forward.py``,
``transport_operator_pallas.py``, ``transport_operator_warp.py``).
'''

import equinox as eqx
from abc import abstractmethod
from typing import TYPE_CHECKING, Type, ClassVar, Tuple

from .sweep_data import BasicSweepData, SweepData

if TYPE_CHECKING:
    from .transport_operator_interface import TransportOperator


# ---------------------------------------------------------------------------
# Forward strategies
# ---------------------------------------------------------------------------
class ForwardStrategy(eqx.Module):
    '''How to apply the single-angle forward operator. Consumes ``operator._forward_data()``.'''

    @abstractmethod
    def forward(self, operator: "TransportOperator", x):
        ...


# ---------------------------------------------------------------------------
# Inversion strategies
# ---------------------------------------------------------------------------
class InversionStrategy(eqx.Module):
    '''
    How to sweep-invert the single-angle operator. Consumes ``operator._inversion_args()`` and
    ``operator.sweep_data``.

    A strategy may need *richer sweep data* than the plain order (e.g. the level-based-parallelism
    sweep needs the :class:`~jax_sn.operators.transport_operator.ProvidesLevels` capability). It answers two separate
    questions about that:

    - :attr:`required_capabilities` -- the :class:`~jax_sn.operators.transport_operator.SweepData` *capabilities* the sweep
      data must provide (interfaces, not concrete types). :meth:`check_compatible` accepts *any*
      sweep data providing them, so a richer plan (more capabilities) is always acceptable -- this is
      the *acceptance* question, answered at construction / on every ``eqx.tree_at``.
    - :meth:`default_sweep_data_type` (classmethod) -- the *concrete*, minimal sweep-data type a
      creation helper should build to satisfy those capabilities. This is the *build* question,
      answered *before* the operator exists (no instance data), so a helper can construct one.

    Building the sweep data is an explicit, visible step (the helper builds ``default_sweep_data_type``
    from the domain, the caller can build it by hand, and it is rebuilt via ``SweepData.reconstruct``)
    -- never hidden work inside operator construction.
    '''

    required_capabilities: ClassVar[Tuple[type, ...]] = ()
    '''Sweep-data capability interfaces this strategy requires (e.g. ``(ProvidesLevels,)``). Empty
    means any :class:`~jax_sn.operators.transport_operator.SweepData` is acceptable.'''

    vmappable: ClassVar[bool] = True
    '''Whether an outer ``vmap`` over this inversion vectorises. ``False`` for an opaque external
    kernel (e.g. warp), where ``vmap`` cannot see into the call and falls back to a serial loop --
    so a vmap-based lift (see :attr:`AngleLifting.vmaps_inner`) over it silently serialises. This is
    a *performance* fact, kept separate from the *correctness* check in :meth:`check_compatible`.'''

    @abstractmethod
    def invert(self, operator: "TransportOperator", b):
        ...

    @classmethod
    def default_sweep_data_type(cls) -> Type[SweepData]:
        '''The concrete, minimal sweep-data type a creation helper should build for this strategy.
        Default: the plain sort (provides no extra capability).'''
        return BasicSweepData

    @classmethod
    @abstractmethod
    def default_angle_lifting(cls) -> "AngleLifting":
        '''The multi-angle lift a creation helper uses by default for this inversion. Each strategy
        declares its own: the generic inversions return :class:`VmapAngles`; an opaque-kernel inversion
        (one with ``vmappable = False``, e.g. warp) returns its matching fused lift. Declared abstract
        here -- not defaulted -- so this module stays free of the concrete lift classes that live under
        ``implementations/``.'''
        ...

    def check_compatible(self, operator: "TransportOperator"):
        '''Assert the operator's sweep data provides every capability in :attr:`required_capabilities`.
        Permissive: a richer sweep data (more capabilities) passes. Run from
        ``TransportOperator.__check_init__``, so it fires at construction and on every ``eqx.tree_at``.'''
        for capability in self.required_capabilities:
            assert isinstance(operator.sweep_data, capability), \
                f"{type(self).__name__} requires sweep data providing {capability.__name__}, " \
                f"but got {type(operator.sweep_data).__name__}. Build it via " \
                f"{type(self).__name__}.default_sweep_data_type().from_domain_quadrature_set(...)."

# ---------------------------------------------------------------------------
# Angle lifting
# ---------------------------------------------------------------------------
class AngleLifting(eqx.Module):
    '''
    How to lift a single-angle operator to multiple angles. The single-angle operator is used
    strictly as a black-box F-morphism (via ``mv`` / ``_invert_internal``); the multi-angle operator
    owns the lift, so the single-angle class stays ignorant of the multi-angle space.
    '''

    vmaps_inner: ClassVar[bool] = False
    '''Whether this lift works by ``vmap``-ing the wrapped single-angle op. ``True`` for the generic
    :class:`VmapAngles`; ``False`` for a fused kernel that handles the angles itself. A ``True`` lift
    over an inversion with ``vmappable = False`` serialises -- the one combination a benchmark should
    not select (it is still constructible; this only flags it).'''

    @abstractmethod
    def forward(self, single_op: "TransportOperator", x):
        ...

    @abstractmethod
    def invert(self, single_op: "TransportOperator", b):
        ...

    def check_compatible(self, single_op: "TransportOperator"):
        '''Assert the wrapped single-angle operator satisfies this lift's precondition. Default: nothing to check.'''
        pass

