'''
Strategy objects for the transport operator.

The transport operator is parameterized by composable strategy objects rather than by subclassing:

- :class:`ForwardStrategy`   -- how to apply the (single-angle) forward operator ``mv``.
- :class:`InversionStrategy` -- how to sweep-invert the (single-angle) operator ``_invert_internal``.
- :class:`AngleLifting`      -- how to lift a single-angle operator to multiple angles.

A new method is a single small class implementing one ``__call__`` (forward/inversion) or one
``forward``/``invert`` pair (lifting). Forward and inversion strategies are independent fields on the
operator, so any forward composes with any inversion (``F + I`` classes, not ``F x I`` operators).

The pallas and warp strategies live next to their kernels (``transport_operator_pallas.py``,
``transport_operator_warp.py``) so this module stays free of optional/heavy dependencies.
'''

import equinox as eqx
from abc import abstractmethod
from typing import TYPE_CHECKING, Type

from jax_sn.domain.geometric_domain import TopologicalSortedAngle, TopologicalSortedAngleBase
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
    def __call__(self, operator: "TransportOperator", x):
        ...


class VmapForward(ForwardStrategy):
    '''Forward via vmap over all elements.'''

    def __call__(self, operator: "TransportOperator", x):
        return forward_transport_operator_vmap(*operator._forward_data(), x)


class BatchedForward(ForwardStrategy):
    '''Forward via batched vmap + fori_loop. ``batch_size=None`` reproduces the non-batched version.'''

    batch_size: int = None

    def __call__(self, operator: "TransportOperator", x):
        return forward_transport_operator_batch(*operator._forward_data(), x, self.batch_size)


# ---------------------------------------------------------------------------
# Inversion strategies
# ---------------------------------------------------------------------------
class InversionStrategy(eqx.Module):
    '''
    How to sweep-invert the single-angle operator. Consumes ``operator._sweep_data()``.

    A strategy may have a *precondition on the toposort* (e.g. the level-based-parallelism sweep needs
    a :class:`TopologicalSortedAngleLevelParallel`). It answers two separate questions about that:

    - :meth:`required_toposort_type` (classmethod) -- the *concrete* toposort type a creation helper
      should build for this strategy. This is a property of the strategy class (no instance data), so a
      helper can ask it *before* the operator exists.
    - :meth:`check_compatible` -- whether a given (already-built) operator's toposort/data are
      *acceptable*. This is more permissive and data-dependent (e.g. the plain sweeps accept any
      toposort, including a level-parallel one), so it is not the same as the build type.

    Building the toposort is an explicit, visible step (the helper builds ``required_toposort_type``
    from the domain, the caller can build it by hand, and it is rebuilt via
    ``TopologicalSortedAngleBase.reconstruct``) -- never hidden work inside operator construction.
    '''

    @abstractmethod
    def __call__(self, operator: "TransportOperator", b):
        ...

    @classmethod
    def required_toposort_type(cls) -> Type[TopologicalSortedAngleBase]:
        '''The concrete toposort type a creation helper should build for this strategy. Default: the plain sort.'''
        return TopologicalSortedAngle

    def check_compatible(self, operator: "TransportOperator"):
        '''Assert the operator's toposort/data are acceptable to this strategy. Default: nothing to check
        (any toposort works). Run from ``TransportOperator.__check_init__``, so it fires at construction
        and on every ``eqx.tree_at``.'''
        pass


class VmapSweep(InversionStrategy):
    '''Inverse sweep with all per-element matrices computed via vmap (high memory).'''

    def __call__(self, operator: "TransportOperator", b):
        return transport_inverse_sweep_complete_vmap(*operator._sweep_data(), b)


class BatchedSweep(InversionStrategy):
    '''Inverse sweep over batches of elements. ``batch_size=None`` reproduces the non-batched version.'''

    batch_size: int = None

    def __call__(self, operator: "TransportOperator", b):
        return transport_inverse_sweep_batched(*operator._sweep_data(), b, self.batch_size)


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
            op_i = eqx.tree_at(lambda o: o.topological_sorted_angle, single_op, toposort_i)
            return fn(op_i, v_i)
        return eqx.filter_vmap(one_angle, in_axes=(0, 0))(single_op.topological_sorted_angle, leading)

    def forward(self, single_op: "TransportOperator", x):
        return self._map_over_angles(single_op, x, lambda op, v: op.mv(v))

    def invert(self, single_op: "TransportOperator", b):
        return self._map_over_angles(single_op, b, lambda op, v: op._invert_internal(v))
