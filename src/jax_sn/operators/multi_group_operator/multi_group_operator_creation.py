from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles
from jax_sn.operators.transport_operator import ForwardStrategy, InversionStrategy, AngleLifting, VmapForward, VmapSweep, VmapAngles
from jax_sn.operators.mass_operator import MassOperator, MassOperatorVmap
from jax_sn.operators.scatter_operator import GatherScatterOperator, GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan, MultiGroupOperator
from jax_sn.quadrature_set import QuadratureSet, QuadratureSetBase
from typing import Dict, Type
from jax_sn.solution_domain import SolutionDomain
import jax.numpy as jnp

def create_multi_group_operator(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase,
                                forward_strategy : ForwardStrategy = VmapForward(),
                                inversion_strategy : InversionStrategy = VmapSweep(),
                                angle_lifting : AngleLifting = VmapAngles(),
                                topological_sorted_angle_kwargs : Dict = {},
                                mass_operator_cls : Type[MassOperator] = MassOperatorVmap,
                                mass_operator_cls_kwargs : Dict = {},
                                gather_scatter_operator_cls : Type[GatherScatterOperator] = GatherScatterOperatorBroadcasted,
                                gather_scatter_operator_cls_kwargs : Dict = {},
                                multi_group_operator_cls : Type[MultiGroupOperatorScan] = MultiGroupOperatorScan,
                                multi_group_operator_cls_kwargs : Dict = {}
                                ) -> MultiGroupOperator:
    r'''
    Create a MultiGroupOperator for the given solution domain and quadrature set. The transport operator
    is composed from strategy objects; the other sub-operators take custom types + kwargs.

    The toposort the transport operator needs is built from ``inversion_strategy.required_toposort_type()``
    -- so e.g. passing a level-based-parallelism sweep automatically builds a level-parallel toposort.
    Build parameters for that toposort (e.g. ``maximum_number_of_levels`` for the level-parallel one) go
    in ``topological_sorted_angle_kwargs``.

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
        determines the toposort type that gets built (via ``required_toposort_type``).
    angle_lifting : AngleLifting, optional
        How the single-angle operator is lifted over the angle axis, by default :class:`VmapAngles`.
    topological_sorted_angle_kwargs : Dict, optional
        Build kwargs forwarded to ``required_toposort_type().from_domain_quadrature_set`` (e.g.
        ``{"maximum_number_of_levels": ...}`` for a level-parallel toposort), by default {}.
    mass_operator_cls : Type[MassOperator], optional
        The class to use for the mass operator, by default :class:`MassOperatorVmap`
    mass_operator_cls_kwargs : Dict, optional
        The kwargs to use for the mass operator, by default {}
    gather_scatter_operator_cls : Type[GatherScatterOperator], optional
        The class to use for the gather scatter operator, by default :class:`GatherScatterOperatorBroadcasted`
    gather_scatter_operator_cls_kwargs : Dict, optional
        The kwargs to use for the gather scatter operator, by default {}
    multi_group_operator_cls : Type[MultiGroupOperatorScan], optional
        The class to use for the multi group operator, by default :class:`MultiGroupOperatorScan`

    Returns
    -------
    MultiGroupOperator
        The created multi group operator.
    '''

    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    topological_sorted_angle_type = inversion_strategy.required_toposort_type()
    topological_sorted_angles     = topological_sorted_angle_type.from_domain_quadrature_set(solution_domain.domain, quadrature_set, **topological_sorted_angle_kwargs)

    single_angle_operator         = TransportOperator(solution_domain.solution_domain_data, topological_sorted_angles, 0, forward_strategy, inversion_strategy)
    transport_operator            = TransportOperatorMultipleAngles(single_angle_operator, angle_lifting)


    mass_operator             = mass_operator_cls.from_transport_operator(transport_operator, **mass_operator_cls_kwargs)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)            
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = gather_scatter_operator_cls(cross_sections_indexed=solution_domain.domain.cross_sections_indexed, **gather_scatter_operator_cls_kwargs)
    
    return multi_group_operator_cls(moment_operator, discrete_operator, transport_operator, mass_operator, gather_scatter_operator, n_energy_groups, **multi_group_operator_cls_kwargs)
    
