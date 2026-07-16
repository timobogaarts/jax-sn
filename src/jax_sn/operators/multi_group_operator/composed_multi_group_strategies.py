r'''
Multi-group strategies

Just composed for now; fused kernel is supported but not implemented.
'''

from abc import abstractmethod
from typing import Type

import jax
import jax.numpy as jnp
import equinox as eqx

from jax_sn.operators.operator_data import OperatorData

from jax_sn.operators.scatter_operator import GatherScatterData, SingleGroupGatherScatterStrategy
from jax_sn.operators.single_group_operator import SingleGroupData
from .multi_group_operator_base import MultiGroupStrategyBase
from ..single_group_operator.single_group_operators_base import SingleGroupChainStrategy


class ComposedMultiGroupStrategy(MultiGroupStrategyBase):
    r'''
    Uses a chain and a gather-scatter that fully define the action of a single group, mapping in some way
    over all the groups.    
    '''
    single_group_chain_strategy          : SingleGroupChainStrategy
    single_group_gather_scatter_strategy : SingleGroupGatherScatterStrategy

    @abstractmethod
    def _lift_over_groups(self, body, n_energy_groups: int):
        '''Stack ``body(g)`` over ``g in range(n_energy_groups)`` into a leading group axis.'''
        ...

    def forward(self, operator_data: OperatorData, x):
        chain   = self.single_group_chain_strategy
        gather  = self.single_group_gather_scatter_strategy
        gs_data = GatherScatterData(operator_data.domain.cross_sections_indexed)

        # A_g x = x_g - C_g U_g[x]; U_g[x] gathers the scatter source for g over all incoming groups of x.
        def body(g):
            return x[g] - chain.forward(SingleGroupData(operator_data, g), gather.incoming_gather(gs_data, x, g))

        return self._lift_over_groups(body, operator_data.n_energy_groups)

    def transpose(self, operator_data: OperatorData, x):
        chain   = self.single_group_chain_strategy
        gather  = self.single_group_gather_scatter_strategy
        gs_data = GatherScatterData(operator_data.domain.cross_sections_indexed)

        # zeta_g = C_g^T x_g is per-group independent; the transpose-gather over all zeta then couples groups.
        zeta = self._lift_over_groups(
            lambda g: chain.transpose(SingleGroupData(operator_data, g), x[g]), operator_data.n_energy_groups)

        return self._lift_over_groups(
            lambda g: x[g] - gather.outgoing_gather(gs_data, zeta, g), operator_data.n_energy_groups)
    
    def default_sweep_data(self):
        return self.single_group_chain_strategy.default_sweep_data()


class MultiGroupScanStrategy(ComposedMultiGroupStrategy):
    '''Lift the per-group body by a serialising scan. Optionally checkpoint for reduced memory usage'''
    checkpoint  : bool = eqx.field(static=True, default=False)

    def _lift_over_groups(self, body, n_energy_groups: int):
        lifted_body = jax.checkpoint(body) if self.checkpoint else body
        return jax.lax.scan(lambda _, g: (None, lifted_body(g)), None, jnp.arange(n_energy_groups))[1]

class MultiGroupVmapStrategy(ComposedMultiGroupStrategy):
    '''Lift by batching the entire vmap. Naturally most parallelizable but significantly increases memory (use on small problems only)'''

    def _lift_over_groups(self, body, n_energy_groups: int):
        return jax.vmap(body)(jnp.arange(n_energy_groups))


