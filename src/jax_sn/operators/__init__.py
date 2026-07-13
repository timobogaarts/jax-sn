'''
This submodule defines the operators in the SN discretisation.

Each operator is defined by data (domain, quadrature, sweep ordering, fem matrices) and implementation 
strategies. 

The data is bundled in ```OperatorData```, split into differentiable and non-differentiable data.

Strategies are built either from combining operators or from a fused strategy. 

The operators are all Lineax operators, which are built from a combination of data and strategy.
'''

from . import transport_operator, moment_operator, mass_operator, scatter_operator, single_group_operator, multi_group_operator
__all__ = ['transport_operator', 'moment_operator', 'mass_operator', 'scatter_operator', 'single_group_operator', 'multi_group_operator']