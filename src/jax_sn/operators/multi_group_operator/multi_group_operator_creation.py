from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, TransportOperatorVmap, TransportOperatorMultipleAnglesVmap
from jax_sn.operators.mass_operator import MassOperator, MassOperatorVmap
from jax_sn.domain import TopologicalSortedAngle
from jax_sn.operators.scatter_operator import GatherScatterOperator, GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan, MultiGroupOperator
from jax_sn.quadrature_set import QuadratureSet, QuadratureSetBase
from typing import Dict, Type
from jax_sn.solution_domain import SolutionDomain
import jax.numpy as jnp

def create_multi_group_operator(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase,
                                single_angle_transport_operator_cls : Type[TransportOperator] = TransportOperatorVmap,
                                single_angle_transport_operator_cls_kwargs : Dict = {},
                                multi_angle_transport_operator_cls : Type[TransportOperatorMultipleAngles] = TransportOperatorMultipleAnglesVmap,
                                multi_angle_transport_operator_cls_kwargs : Dict = {},
                                mass_operator_cls : Type[MassOperator] = MassOperatorVmap,
                                mass_operator_cls_kwargs : Dict = {},
                                gather_scatter_operator_cls : Type[GatherScatterOperator] = GatherScatterOperatorBroadcasted,
                                gather_scatter_operator_cls_kwargs : Dict = {},
                                multi_group_operator_cls : Type[MultiGroupOperatorScan] = MultiGroupOperatorScan,
                                multi_group_operator_cls_kwargs : Dict = {}
                                ) -> MultiGroupOperator:
    r'''
    Create a MultiGroupOperator for the given solution domain and quadrature set. Can provide custom types for each of the sub-operators, including possible kwargs.

    Parameters
    ----------
    solution_domain : SolutionDomain
        The solution domain for which to create the multi group operator.
    quadrature_set : QuadratureSet
        The quadrature set for which to create the multi group operator.
    single_angle_transport_operator_cls : Type[TransportOperator], optional
        The class to use for the single angle transport operator, by default :class:`TransportOperatorVmap`
    single_angle_transport_operator_cls_kwargs : Dict, optional
        The kwargs to use for the single angle transport operator, by default {}
    multi_angle_transport_operator_cls : Type[TransportOperatorMultipleAngles], optional
        The class to use for the multi angle transport operator, by default :class:`TransportOperatorMultipleAnglesVmap`
    multi_angle_transport_operator_cls_kwargs : Dict, optional
        The kwargs to use for the multi angle transport operator, by default {}
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

    topological_sorted_angles = TopologicalSortedAngle.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)

    transport_operator        = multi_angle_transport_operator_cls(single_angle_transport_operator_cls(solution_domain.solution_domain_data, topological_sorted_angles, 0, **single_angle_transport_operator_cls_kwargs), **multi_angle_transport_operator_cls_kwargs)


    mass_operator             = mass_operator_cls.from_transport_operator(transport_operator, **mass_operator_cls_kwargs)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)            
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = gather_scatter_operator_cls(cross_sections_indexed=solution_domain.domain.cross_sections_indexed, **gather_scatter_operator_cls_kwargs)
    
    return multi_group_operator_cls(moment_operator, discrete_operator, transport_operator, mass_operator, gather_scatter_operator, n_energy_groups, **multi_group_operator_cls_kwargs)
    
