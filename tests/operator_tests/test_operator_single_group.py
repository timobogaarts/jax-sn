import pytest
from .operator_util import BaseOperatorProperties

class TestSingleGroupOperator(BaseOperatorProperties):
    '''Has no variants except itself; tests only regression against frozen data.'''

    @pytest.fixture
    def operator(self, bb_sgop):
        return bb_sgop[0]

    @pytest.fixture
    def build_reference(self, bb_sgop):
        return lambda: bb_sgop[0]
