from jax_sn.operators.operator_data import OperatorData, AngularData, DiscretisationData, NonDifferentiableData
from jax_sn.operators.multi_group_operator import MultiGroupOperator, MultiGroupStrategyBase
from jax_sn.quadrature_set import QuadratureSetBase
from typing import Dict, Protocol
from jax_sn.solution_domain import SolutionDomain
import jax.numpy as jnp

class DefinesDefaultSweepData(Protocol):
    @classmethod
    def default_sweep_data(cls):
        ...

def create_operator_data(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase, strategy : DefinesDefaultSweepData, sweep_data_kwargs : Dict = {}) -> OperatorData:
    '''
    Creates the total operator data struct for the given solution domain and quadrature set, using the default sweep data type of the given strategy.
    This can also be a single group operator or something else.
    '''    
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max
    sweep_data_type           = strategy.default_sweep_data()
    sweep_data                = sweep_data_type.from_domain_quadrature_set(solution_domain.domain, quadrature_set, **sweep_data_kwargs)
    angular_data              = AngularData.from_quadrature_set(quadrature_set, lmax)
    operator_data             = OperatorData(DiscretisationData(solution_domain.spatial_data, angular_data), NonDifferentiableData(sweep_data))
    return operator_data

def create_multi_group_operator(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase, multi_group_strategy : MultiGroupStrategyBase, sweep_data_kwargs : Dict = {}) -> MultiGroupOperator:
    return MultiGroupOperator(create_operator_data(solution_domain, quadrature_set, multi_group_strategy, sweep_data_kwargs), multi_group_strategy, transpose_bool = False)


