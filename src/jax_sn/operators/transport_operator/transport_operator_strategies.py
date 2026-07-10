'''
Strategy objects for the transport operator.

The transport operator is parameterized by composable strategy objects rather than by subclassing:

- :class:`ForwardStrategy`         -- how to apply the (single-angle) forward operator ``mv``.
- :class:`InversionStrategy`       -- how to sweep-invert the (single-angle) operator ``_invert_internal``.
- :class:`ForwardAngleLifting`     -- how to lift the single-angle *forward* to multiple angles.
- :class:`InversionAngleLifting`   -- how to lift the single-angle *inverse* to multiple angles.

A new method is a single small class implementing one action method named after that action:
``forward`` (forward strategies / forward lifts) or ``invert`` (inversion strategies / inversion
lifts). The two axes are independent fields on the operator, so any forward composes with any
inversion (``F + I`` classes, not ``F x I`` operators) -- at the single-angle level *and* over the
angle lift. The lift is split the same way: the forward lift has a single implementation (vmap),
while the inversion lift carries the fused-kernel variants, so bundling them would force every fused
inversion lift to re-declare the same vmap forward.

This module holds only the *abstract* strategy bases. Each concrete implementation lives next to
its kernel under ``implementations/`` (e.g. ``implementations/transport_operator_forward.py``,
``transport_operator_pallas.py``, ``transport_operator_warp.py``).
'''

import equinox as eqx
from abc import abstractmethod
from typing import TYPE_CHECKING, Type, Tuple

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

    - :meth:`required_capabilities` -- the :class:`~jax_sn.operators.transport_operator.SweepData` *capabilities* the sweep
      data must provide (interfaces, not concrete types). :meth:`check_compatible` accepts *any*
      sweep data providing them, so a richer plan (more capabilities) is always acceptable -- this is
      the *acceptance* question, answered at construction / on every ``eqx.tree_at``.
    - :meth:`default_sweep_data_type` -- the *concrete*, minimal sweep-data type a
      creation helper should build to satisfy those capabilities. This is the *build* question,
      answered *before* the operator exists (no instance data), so a helper can construct one.

    Building the sweep data is an explicit, visible step (the helper builds ``default_sweep_data_type``
    from the domain, the caller can build it by hand, and it is rebuilt via ``SweepData.reconstruct``)
    -- never hidden work inside operator construction.

    Every fixed-per-class fact a subclass declares to self-classify is a ``classmethod`` (answerable
    without an instance). Those with a fail-safe default are concrete (override only if special);
    :meth:`vmappable` has no safe default -- forgetting it would silently re-arm a performance footgun
    -- so it is ``@abstractmethod``, forcing each concrete strategy to state its value.
    '''

    @classmethod
    def required_capabilities(cls) -> Tuple[type, ...]:
        '''Sweep-data capability interfaces this strategy requires (e.g. ``(ProvidesLevels,)``). Empty
        (the default) means any :class:`~jax_sn.operators.transport_operator.SweepData` is acceptable.'''
        return ()

    @classmethod
    @abstractmethod
    def vmappable(cls) -> bool:
        '''Whether an outer ``vmap`` over this inversion vectorises. ``False`` for an opaque external
        kernel (e.g. warp), where ``vmap`` cannot see into the call and falls back to a serial loop --
        so a vmap-based lift (see :meth:`InversionAngleLifting.vmaps_inner`) over it silently serialises. This is
        a *performance* fact, kept separate from the *correctness* check in :meth:`check_compatible`.
        Abstract (no safe default): each concrete inversion must declare it.'''
        ...

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
    def default_inversion_lifting(cls) -> "InversionAngleLifting":
        '''The multi-angle inverse lift a creation helper uses by default for this inversion. Each
        strategy declares its own: the generic inversions return :class:`VmapInversionAngles`; an
        opaque-kernel inversion (one with ``vmappable = False``, e.g. warp) returns its matching fused
        lift. Declared abstract here -- not defaulted -- so this module stays free of the concrete lift
        classes that live under ``implementations/``. (The *forward* lift has a single implementation,
        :class:`VmapForwardAngles`, so it needs no per-strategy default.)'''
        ...

    def check_compatible(self, operator: "TransportOperator"):
        '''Assert the operator's sweep data provides every capability in :meth:`required_capabilities`.
        Permissive: a richer sweep data (more capabilities) passes. Run from
        ``TransportOperator.__check_init__``, so it fires at construction and on every ``eqx.tree_at``.'''
        for capability in self.required_capabilities():
            assert isinstance(operator.sweep_data, capability), \
                f"{type(self).__name__} requires sweep data providing {capability.__name__}, " \
                f"but got {type(operator.sweep_data).__name__}. Build it via " \
                f"{type(self).__name__}.default_sweep_data_type().from_domain_quadrature_set(...)."

# ---------------------------------------------------------------------------
# Angle lifting
# ---------------------------------------------------------------------------
# The lift is split into a *forward* lift and an *inversion* lift -- two independent fields on the
# multi-angle operator, mirroring the single-angle ``ForwardStrategy`` + ``InversionStrategy`` split.
# The single-angle operator is used strictly as a black-box F-morphism (via ``mv`` / ``_invert_internal``);
# the multi-angle operator owns the lift, so the single-angle class stays ignorant of the multi-angle
# space. The forward lift has a single implementation (:class:`VmapForwardAngles`); all the variation --
# fused warp kernels -- lives on the inversion lift, which is why the two are separate axes.
class ForwardAngleLifting(eqx.Module):
    '''How to lift the single-angle *forward* (``mv``) over the angle axis.'''

    @abstractmethod
    def forward(self, single_op: "TransportOperator", x):
        ...

    def check_compatible(self, single_op: "TransportOperator"):
        '''Assert the wrapped single-angle operator satisfies this lift's precondition. Default: nothing to check.'''
        pass


class InversionAngleLifting(eqx.Module):
    '''How to lift the single-angle *inverse* (``_invert_internal``) over the angle axis.'''

    @classmethod
    @abstractmethod
    def vmaps_inner(cls) -> bool:
        '''Whether this lift works by ``vmap``-ing the wrapped single-angle inverse. ``True`` for the
        generic :class:`VmapInversionAngles`; ``False`` for a fused kernel that handles the angles
        itself. A ``True`` lift over an inversion with ``vmappable() == False`` serialises -- the one
        combination a benchmark should not select (it is still constructible; this only flags it).
        Abstract (no safe default): each concrete lift must declare it.'''
        ...

    @abstractmethod
    def invert(self, single_op: "TransportOperator", b):
        ...

    def check_compatible(self, single_op: "TransportOperator"):
        '''Assert the wrapped single-angle operator satisfies this lift's precondition. Default: nothing to check.'''
        pass

