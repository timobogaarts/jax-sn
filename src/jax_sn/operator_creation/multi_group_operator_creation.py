'''
The original :class:`SolutionDomain`-based factories. These are now thin adapters over
:mod:`jax_sn.operator_creation.builders`: they recover the :class:`Builder` of the solution domain
they are given and hand off to its build methods, so there is a single derivation path shared with
the parametrisation machinery.
'''

from jax_sn.operators.operator_data import OperatorData
from jax_sn.operators.multi_group_operator import MultiGroupOperator, MultiGroupStrategyBase
from .builders import Builder
from jax_sn.operators.sweep_policy import Resort
from jax_sn.quadrature_set import QuadratureSetBase
from typing import Dict, Protocol
from jax_sn.solution_domain import SolutionDomain

class DefinesDefaultSweepData(Protocol):
    @classmethod
    def default_sweep_data(cls):
        ...

def create_operator_data(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase, strategy : DefinesDefaultSweepData, sweep_data_kwargs : Dict = {}) -> OperatorData:
    '''
    Creates the total operator data struct for the given solution domain and quadrature set, using the default sweep data type of the given strategy.
    This can also be a single group operator or something else.
    '''
    builder = Builder.from_solution_domain(solution_domain, quadrature_set)
    return builder.operator_data(strategy.default_sweep_data(), Resort(**sweep_data_kwargs))

def create_multi_group_operator(solution_domain : SolutionDomain, quadrature_set : QuadratureSetBase, multi_group_strategy : MultiGroupStrategyBase, sweep_data_kwargs : Dict = {}) -> MultiGroupOperator:
    builder = Builder.from_solution_domain(solution_domain, quadrature_set)
    return builder.multi_group_operator(multi_group_strategy, Resort(**sweep_data_kwargs))
