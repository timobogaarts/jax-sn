
import equinox as eqx
import pytest
import jax.numpy as jnp
import numpy as onp
import lineax as lx

from jax_sn.operators.multi_group_operator import MultiGroupScanStrategy, MultiGroupVmapStrategy
from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver
from .operator_util import BaseOperatorProperties
from conftest import DIM_DEGREES


def _relift(mgop, strategy):
    '''Swap the multigroup lift, keeping the shared chain + gather-scatter sub-strategies and data.'''
    return eqx.tree_at(lambda op: op.strategy, mgop, strategy)


def _scan(mgop, checkpoint):
    s = mgop.strategy
    return _relift(mgop, MultiGroupScanStrategy(s.single_group_chain_strategy,
                                                s.single_group_gather_scatter_strategy, checkpoint=checkpoint))


def _vmap(mgop):
    s = mgop.strategy
    return _relift(mgop, MultiGroupVmapStrategy(s.single_group_chain_strategy,
                                                s.single_group_gather_scatter_strategy))


# name -> builder(mgop) for each lift variant. `scan` (no checkpoint) is the pivot the reference is built
# from, so it is trivially self-consistent; the other two must reproduce it.
MG_VARIANTS = {
    "scan":      lambda mgop: _scan(mgop, checkpoint=False),
    "scan_ckpt": lambda mgop: _scan(mgop, checkpoint=True),
    "vmap":      _vmap,
}


def _combos():
    return [pytest.param((dim, degree), builder, id=f"dim{dim}-deg{degree}-{name}")
            for (dim, degree) in DIM_DEGREES
            for name, builder in MG_VARIANTS.items()]


@pytest.mark.parametrize("bb_mgop_dim_degree, build_variant", _combos(), indirect=["bb_mgop_dim_degree"])
class TestMultiGroupOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, build_variant):
        return build_variant(bb_mgop[0])

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: bb_mgop[0]


