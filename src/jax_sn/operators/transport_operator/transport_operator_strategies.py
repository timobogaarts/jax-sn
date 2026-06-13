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

The pallas and warp strategies live next to their kernels (``transport_operator_pallas.py``,
``transport_operator_warp.py``) so this module stays free of optional/heavy dependencies.
'''

import equinox as eqx
from abc import abstractmethod
from typing import TYPE_CHECKING, Type, ClassVar, Tuple

from .sweep_data import BasicSweepData, SweepData
from .transport_operator_imp import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap
from .transport_operator_forward import forward_transport_operator_vmap, forward_transport_operator_batch

if TYPE_CHECKING:
    from .transport_operator_base import TransportOperator


# ---------------------------------------------------------------------------
# Forward strategies
# ---------------------------------------------------------------------------
class ForwardStrategy(eqx.Module):
    '''How to apply the single-angle forward operator. Consumes ``operator._forward_data()``.'''

    @abstractmethod
    def forward(self, operator: "TransportOperator", x):
        ...


class VmapForward(ForwardStrategy):
    '''Forward via vmap over all elements.'''

    def forward(self, operator: "TransportOperator", x):
        return forward_transport_operator_vmap(*operator._forward_data(), x)


class BatchedForward(ForwardStrategy):
    '''Forward via batched vmap + fori_loop. ``batch_size=None`` reproduces the non-batched version.'''

    batch_size: int = None

    def forward(self, operator: "TransportOperator", x):
        return forward_transport_operator_batch(*operator._forward_data(), x, self.batch_size)


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

    @abstractmethod
    def invert(self, operator: "TransportOperator", b):
        ...

    @classmethod
    def default_sweep_data_type(cls) -> Type[SweepData]:
        '''The concrete, minimal sweep-data type a creation helper should build for this strategy.
        Default: the plain sort (provides no extra capability).'''
        return BasicSweepData

    def check_compatible(self, operator: "TransportOperator"):
        '''Assert the operator's sweep data provides every capability in :attr:`required_capabilities`.
        Permissive: a richer sweep data (more capabilities) passes. Run from
        ``TransportOperator.__check_init__``, so it fires at construction and on every ``eqx.tree_at``.'''
        for capability in self.required_capabilities:
            assert isinstance(operator.sweep_data, capability), \
                f"{type(self).__name__} requires sweep data providing {capability.__name__}, " \
                f"but got {type(operator.sweep_data).__name__}. Build it via " \
                f"{type(self).__name__}.default_sweep_data_type().from_domain_quadrature_set(...)."


class VmapSweep(InversionStrategy):
    '''Inverse sweep with all per-element matrices computed via vmap (high memory).'''

    def invert(self, operator: "TransportOperator", b):
        return transport_inverse_sweep_complete_vmap(*operator._inversion_args(), b)


class BatchedSweep(InversionStrategy):
    '''Inverse sweep over batches of elements. ``batch_size=None`` reproduces the non-batched version.'''

    batch_size: int = None

    def invert(self, operator: "TransportOperator", b):
        return transport_inverse_sweep_batched(*operator._inversion_args(), b, self.batch_size)


# ---------------------------------------------------------------------------
# Angle lifting
# ---------------------------------------------------------------------------
class AngleLifting(eqx.Module):
    '''
    How to lift a single-angle operator to multiple angles. The single-angle operator is used
    strictly as a black-box F-morphism (via ``mv`` / ``_invert_internal``); the multi-angle operator
    owns the lift, so the single-angle class stays ignorant of the multi-angle space.
    '''

    @abstractmethod
    def forward(self, single_op: "TransportOperator", x):
        ...

    @abstractmethod
    def invert(self, single_op: "TransportOperator", b):
        ...

    def check_compatible(self, single_op: "TransportOperator"):
        '''Assert the wrapped single-angle operator satisfies this lift's precondition. Default: nothing to check.'''
        pass


class VmapAngles(AngleLifting):
    '''
    Lift by vmapping the single-angle reduction over the angle axis. The angle axis lives in the
    toposort, so we vmap over the toposort's leading axis and the leading axis of the input, rebuilding
    a single-angle operator per angle. The base operator therefore never needs to know how to vmap.
    '''

    @staticmethod
    def _map_over_angles(single_op: "TransportOperator", leading, fn):
        def one_angle(toposort_i, v_i):
            op_i = eqx.tree_at(lambda o: o.sweep_data, single_op, toposort_i)
            return fn(op_i, v_i)
        return eqx.filter_vmap(one_angle, in_axes=(0, 0))(single_op.sweep_data, leading)

    def forward(self, single_op: "TransportOperator", x):
        return self._map_over_angles(single_op, x, lambda op, v: op.mv(v))

    def invert(self, single_op: "TransportOperator", b):
        return self._map_over_angles(single_op, b, lambda op, v: op._invert_internal(v))
