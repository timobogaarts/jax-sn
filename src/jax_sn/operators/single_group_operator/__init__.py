'''
This sub-module defines operators on a single group.

This includes several sub-operators which are used in other places (i.e. lifting to multigroup / rhs):

1. The total single group operator (used for lower tri solves):
        I - M T_g^{-1} H D S_{gg}
2. The chain operator (used for gathering the scattering source):
        M T_g^{-1} H D
3. The source operator (used for right-hand-side construction):
        M T_g^{-1} H

Each operator carries a strategy of its matching type (full / chain / source); the composed strategies
factor the product into the mass / discrete / moment / transport sub-operators.
'''
from .single_group_operators_base import (
    SingleGroupStrategyBase,
    SingleGroupStrategy,
    SingleGroupChainStrategy,
    SingleGroupSourceStrategy,
    SingleGroupOperator,
    SingleGroupChainOperator,
    SingleGroupSourceOperator,
    single_group_moment_structure,
    single_group_angle_structure,
)
from .composed_single_group_strategies import (
    ComposedSingleGroupStrategy,
    ComposedSingleGroupChainStrategy,
    ComposedSingleGroupSourceStrategy,
)

__all__ = [
    'SingleGroupStrategyBase', 'SingleGroupStrategy', 'SingleGroupChainStrategy', 'SingleGroupSourceStrategy',
    'SingleGroupOperator', 'SingleGroupChainOperator', 'SingleGroupSourceOperator',
    'ComposedSingleGroupStrategy', 'ComposedSingleGroupChainStrategy', 'ComposedSingleGroupSourceStrategy',
    'single_group_moment_structure', 'single_group_angle_structure',
]
