from jax_sn.operators.transport_operator import ForwardStrategy, InversionStrategy, AngleLifting, VmapForward, VmapSweep, VmapAngles
from jax_sn.operators.transport_operator import TransportSpec, SolutionSetup, AngularSetup
from jax_sn.operators.mass_operator import MassStrategy, VmapMass
from jax_sn.operators.moment_operator import MomentStrategy, DiscreteStrategy, BatchedMoment, BatchedDiscrete
from jax_sn.operators.scatter_operator import GatherScatterOperator, GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan, MultiGroupOperator
from jax_sn.quadrature_set import QuadratureSetBase
from typing import Dict, Type
from jax_sn.solution_domain import SolutionDomain

def create_multi_group_operator(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase,
                                forward_strategy : ForwardStrategy = VmapForward(),
                                inversion_strategy : InversionStrategy = VmapSweep(),
                                angle_lifting : AngleLifting = None,
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
    forward_strategy : ForwardStrategy, optional
        How the (single-angle) transport operator applies the forward operator, by default :class:`VmapForward`.
    inversion_strategy : InversionStrategy, optional
        How the (single-angle) transport operator sweep-inverts, by default :class:`VmapSweep`. Also
        determines the toposort type that gets built (via ``default_sweep_data_type``).
    angle_lifting : AngleLifting, optional
        How the single-angle operator is lifted over the angle axis. ``None`` (default) resolves to
        ``inversion_strategy.default_angle_lifting()`` -- the generic :class:`VmapAngles` for an
        ordinary inversion, or the matching fused lift for an opaque-kernel inversion (warp), so the
        natural multi-angle path is chosen without the caller having to know. Pass one to override.
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

    if angle_lifting is None:
        angle_lifting = inversion_strategy.default_angle_lifting()

    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    sweep_data_type = inversion_strategy.default_sweep_data_type()
    sweep_data     = sweep_data_type.from_domain_quadrature_set(solution_domain.domain, quadrature_set, **sweep_data_kwargs)

    angular_setup             = AngularSetup.from_quadrature_set(quadrature_set, lmax)
    setup                     = SolutionSetup(solution_domain.solution_domain_data, angular_setup, sweep_data)
    transport_spec            = TransportSpec(forward_strategy, inversion_strategy, angle_lifting, energy_group=0)

    return multi_group_operator_cls(setup, transport_spec, n_energy_groups, moment_strategy, discrete_strategy, mass_strategy, gather_scatter_operator_cls, **multi_group_operator_cls_kwargs)
    
