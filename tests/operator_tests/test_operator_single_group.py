

import pytest
from .operator_util import BaseOperatorProperties

class TestSingleGroupOperator(BaseOperatorProperties):
    '''Algebraic properties of the assembled single-group operator (group 0 of the BB problem). It has no
    strategy variant to swap, so operator == reference and the consistency/grad/jvp checks pass trivially
    against its own frozen/live values (the scatter-operator pattern); the adjoint identity and transpose
    idempotence do the real work.'''

    @pytest.fixture
    def operator(self, bb_sgop):
        return bb_sgop[0]

    @pytest.fixture
    def build_reference(self, bb_sgop):
        return lambda: bb_sgop[0]
