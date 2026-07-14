from jax_sn.operators.transport_operator import SingleAngleForwardStrategy, SingleAngleInversionStrategy, MultiAngleInversionStrategy, VmapForward, VmapSweep, VmapAngles
from jax_sn.operators.transport_operator import MultiAngleTransportSpec
from jax_sn.operators.operator_data import OperatorData, AngularData, DiscretisationData, NonDifferentiableData
from jax_sn.operators.mass_operator import MassStrategy, VmapMass
from jax_sn.operators.moment_operator import MomentStrategy, DiscreteStrategy, BatchedMoment, BatchedDiscrete
from jax_sn.operators.scatter_operator import GatherScatterOperator, GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan, MultiGroupOperator
from jax_sn.quadrature_set import QuadratureSetBase
from typing import Dict, Type
from jax_sn.solution_domain import SolutionDomain


def _resolve_multi_angle_inversion_strategy(inversion_strategy : SingleAngleInversionStrategy) -> MultiAngleInversionStrategy:
    '''The natural multi-angle inverse for a single-angle strategy: :class:`VmapAngles` over a vmappable
    kernel, else the strategy's fused native lift (``native_multi_angle``, e.g. warp). A non-vmappable
    strategy with no native lift must be paired explicitly (e.g. ``ScanAngles(strategy)``).'''
    if inversion_strategy.vmappable():
        return VmapAngles(inversion_strategy)
    native = inversion_strategy.native_multi_angle()
    if native is None:
        raise ValueError(
            f"{type(inversion_strategy).__name__} is not vmappable and declares no native multi-angle "
            f"lift; pass multi_angle_inversion_strategy explicitly (e.g. ScanAngles({type(inversion_strategy).__name__}())).")
    return native()


def create_multi_group_operator(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase,
                                forward_strategy : SingleAngleForwardStrategy = VmapForward(),
                                inversion_strategy : SingleAngleInversionStrategy = VmapSweep(),
                                multi_angle_inversion_strategy : MultiAngleInversionStrategy = None,
                                sweep_data_kwargs : Dict = {},
                                moment_strategy : MomentStrategy = BatchedMoment(),
                                discrete_strategy : DiscreteStrategy = BatchedDiscrete(),
                                mass_strategy : MassStrategy = VmapMass(),
                                gather_scatter_operator_cls : Type[GatherScatterOperator] = GatherScatterOperatorBroadcasted,
                                multi_group_operator_cls : Type[MultiGroupOperatorScan] = MultiGroupOperatorScan,
                                multi_group_operator_cls_kwargs : Dict = {}
                                ) -> MultiGroupOperator:
    r'''
    Create a MultiGroupOperator for the given solution domain and quadrature set. The transport operator
    is composed from strategy objects; the other sub-operators take custom types + kwargs.

    The toposort the transport operator needs is built from ``inversion_strategy.default_sweep_data_type()``
    -- so e.g. passing a level-based-parallelism sweep automatically builds a level-parallel toposort.
    Build parameters for that toposort (e.g. ``maximum_number_of_levels`` for the level-parallel one) go
    in ``sweep_data_kwargs``.

    Parameters
    ----------
    solution_domain : SolutionDomain
        The solution domain for which to create the multi group operator.
    quadrature_set : QuadratureSet
        The quadrature set for which to create the multi group operator.
    forward_strategy : SingleAngleForwardStrategy, optional
        How the (single-angle) transport operator applies the forward operator, by default :class:`VmapForward`.
    inversion_strategy : SingleAngleInversionStrategy, optional
        How the (single-angle) transport operator sweep-inverts, by default :class:`VmapSweep`. Used to
        auto-resolve ``multi_angle_inversion_strategy`` when that is ``None`` (see below).
    multi_angle_inversion_strategy : MultiAngleInversionStrategy, optional
        How to invert over the angle axis. ``None`` (default) resolves from ``inversion_strategy``:
        :class:`VmapAngles` (wrapping the strategy) for a vmappable kernel, or the strategy's fused
        native lift (``inversion_strategy.native_multi_angle()``, e.g. warp) for an opaque one -- so the
        natural multi-angle path is chosen without the caller having to know, and the sweep-data toposort
        is built from it (via ``default_sweep_data_type``). Pass one to override (e.g.
        ``ScanAngles(...)`` to serialise). A non-vmappable strategy with no native lift must be paired
        explicitly.
    sweep_data_kwargs : Dict, optional
        Build kwargs forwarded to ``default_sweep_data_type().from_domain_quadrature_set`` (e.g.
        ``{"maximum_number_of_levels": ...}`` for a level-parallel toposort), by default {}.
    mass_strategy : MassStrategy, optional
        The contraction strategy to use for the mass operator, by default :class:`VmapMass`
    gather_scatter_operator_cls : Type[GatherScatterOperator], optional
        The class to use for the gather scatter operator, by default :class:`GatherScatterOperatorBroadcasted`
    multi_group_operator_cls : Type[MultiGroupOperatorScan], optional
        The class to use for the multi group operator, by default :class:`MultiGroupOperatorScan`

    Returns
    -------
    MultiGroupOperator
        The created multi group operator.
    '''

    if multi_angle_inversion_strategy is None:
        multi_angle_inversion_strategy = _resolve_multi_angle_inversion_strategy(inversion_strategy)

    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    sweep_data_type = multi_angle_inversion_strategy.default_sweep_data_type()
    sweep_data     = sweep_data_type.from_domain_quadrature_set(solution_domain.domain, quadrature_set, **sweep_data_kwargs)

    angular_data             = AngularData.from_quadrature_set(quadrature_set, lmax)
    operator_data             = OperatorData(DiscretisationData(solution_domain.spatial_data, angular_data), NonDifferentiableData(sweep_data))
    transport_spec            = MultiAngleTransportSpec(forward_strategy, multi_angle_inversion_strategy)

    return multi_group_operator_cls(operator_data, transport_spec, n_energy_groups, moment_strategy, discrete_strategy, mass_strategy, gather_scatter_operator_cls, **multi_group_operator_cls_kwargs)
    
