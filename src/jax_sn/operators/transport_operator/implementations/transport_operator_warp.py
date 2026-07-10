'''
Warp strategies and convenience constructors for the transport operator.

Warp provides two inverse-sweep families -- a sequential sweep and a level-based-parallelism sweep --
each with a single-angle kernel (an :class:`InversionStrategy`) and a *fused* multi-angle kernel (an
:class:`InversionAngleLifting`). The fused multi-angle inverse is not a vmap of the single-angle one;
it is a dedicated kernel tied to the warp data layout. There is no warp *forward* lift -- the forward
always uses the generic :class:`VmapForwardAngles` -- so these are inversion lifts only.

The warp forward is the batched forward (``BatchedForward``), so warp single-angle operators are just
``TransportOperator`` with a batched forward strategy + a warp inversion strategy -- no special
subclass. The level-parallel variant carries its level setup inside a
:class:`LeveledSweepData` toposort.
'''

from ..sweep_data import LeveledSweepData, ProvidesLevels
from jax_sn.solution_domain import SolutionDomainData

from ..transport_operator_strategies import InversionStrategy, InversionAngleLifting
from .linear_tetrahedron_warp_kernel import (
    single_angle_warp_sweep, multi_angle_warp_sweep,
    level_single_angle_warp_sweep, level_multi_angle_warp_sweep,
    NBASIS, NFACES,
)


def _assert_warp_compatible(solution_domain_data: SolutionDomainData):
    '''Validate that the solution-domain data matches what the warp tetrahedron kernels expect.'''
    assert solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape[-2:] == (3, 3), \
        f"For Warp kernel, expected vol_jacobians_inv shape (3,3), got {solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape}"
    assert solution_domain_data.n_basis == NBASIS, \
        f"For warp kernel, expected {NBASIS} basis functions, got {solution_domain_data.n_basis}"
    assert solution_domain_data.domain.etoe.shape[-1] == NFACES, \
        f"For warp kernel, expected {NFACES} faces per elements, got {solution_domain_data.domain.etoe.shape[-1]}"


# ---------------------------------------------------------------------------
# Inversion strategies (single angle)
# ---------------------------------------------------------------------------
class WarpSequentialSweep(InversionStrategy):
    '''Single-angle inverse via the sequential Warp sweep kernel (no batching).'''

    def invert(self, operator, b):
        return single_angle_warp_sweep(*operator._inversion_args(), b)

    def check_compatible(self, operator):
        _assert_warp_compatible(operator.solution_domain_data)

    @classmethod
    def vmappable(cls):
        return False   # opaque warp kernel: an outer vmap serialises

    @classmethod
    def default_inversion_lifting(cls):
        return WarpSequentialAngles()   # the fused multi-angle kernel, not a vmap of this one


class WarpLevelSweep(InversionStrategy):
    '''
    Single-angle inverse via the level-based-parallelism Warp sweep kernel. Requires sweep data
    providing the :class:`~jax_sn.operators.transport_operator.ProvidesLevels` capability (the level setup for both sweep
    directions). The base :meth:`~InversionStrategy.check_compatible` enforces that capability; this
    strategy only adds the warp-data-layout check. The minimal concrete builder is
    :class:`~jax_sn.operators.transport_operator.LeveledSweepData`.
    '''

    def invert(self, operator, b):
        return level_single_angle_warp_sweep(*operator._inversion_args(), b, operator.sweep_data.level_fwd)

    @classmethod
    def required_capabilities(cls):
        return (ProvidesLevels,)

    @classmethod
    def vmappable(cls):
        return False   # opaque warp kernel: an outer vmap serialises

    @classmethod
    def default_sweep_data_type(cls):
        return LeveledSweepData

    @classmethod
    def default_inversion_lifting(cls):
        return WarpLevelAngles()   # the fused multi-angle kernel, not a vmap of this one

    def check_compatible(self, operator):
        super().check_compatible(operator)  # enforces required_capabilities (ProvidesLevels)
        _assert_warp_compatible(operator.solution_domain_data)


# ---------------------------------------------------------------------------
# Inversion angle lifting (fused multi-angle warp kernels; forward stays VmapForwardAngles)
# ---------------------------------------------------------------------------
class WarpSequentialAngles(InversionAngleLifting):
    '''Multi-angle inverse lift: fused sequential Warp sweep.'''

    @classmethod
    def vmaps_inner(cls):
        return False   # fused kernel handles the angles itself

    def invert(self, single_op, b):
        return multi_angle_warp_sweep(*single_op._inversion_args(), b)

    def check_compatible(self, single_op):
        assert isinstance(single_op.inversion_strategy, WarpSequentialSweep), \
            f"WarpSequentialAngles needs a single-angle operator with a WarpSequentialSweep inverse, got {type(single_op.inversion_strategy).__name__}"


class WarpLevelAngles(InversionAngleLifting):
    '''Multi-angle inverse lift: fused level-based-parallelism Warp sweep.'''

    @classmethod
    def vmaps_inner(cls):
        return False   # fused kernel handles the angles itself

    def invert(self, single_op, b):
        return level_multi_angle_warp_sweep(*single_op._inversion_args(), b, single_op.sweep_data.level_fwd)

    def check_compatible(self, single_op):
        assert isinstance(single_op.inversion_strategy, WarpLevelSweep), \
            f"WarpLevelAngles needs a single-angle operator with a WarpLevelSweep inverse, got {type(single_op.inversion_strategy).__name__}"
