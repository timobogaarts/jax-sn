from ..operators.multi_group_operator.composed_multi_group_strategies import MultiGroupScanStrategy
from ..operators.single_group_operator.composed_single_group_strategies import ComposedSingleGroupChainStrategy

from ..operators.mass_operator import LaxMass
from ..operators.moment_operator import EinsumFluxMoment
from ..operators.transport_operator import VmapAngles, VmapAnglesForward, BatchedSweep, VmapSweep, VmapPallasSweep, BatchedPallasSweep, VmapForward, BatchedForward
from ..operators.scatter_operator import IndexedScatter, BroadcastedSingleGroupGather
from ..operators.transport_operator.implementations.transport_operator_warp import WarpLevelAngles
from ..operators.mass_operator import WarpMass


def _chain(transport_strategy, transport_inversion_strategy, mass_strategy):
    '''A ComposedSingleGroupChainStrategy with the einsum moment/discrete and indexed within-group scatter.'''
    return ComposedSingleGroupChainStrategy(
        transport_strategy           = transport_strategy,
        transport_inversion_strategy = transport_inversion_strategy,
        moment_strategy              = EinsumFluxMoment(),
        discrete_strategy            = EinsumFluxMoment(),
        mass_strategy                = mass_strategy,
        scatter_strategy             = IndexedScatter(),
    )


def _scan(chain, checkpoint):
    '''Lift a single-group chain over the energy-group axis with a (checkpointed) scan and the
    broadcasted per-group gather-scatter.'''
    return MultiGroupScanStrategy(chain, BroadcastedSingleGroupGather(), checkpoint=checkpoint)


class DefaultMultiGroupStrategy:

    @classmethod
    def default_multi_group_strategy(cls, batch_size : int | None = None, checkpoint = True):
        if batch_size is not None:
            chain = _chain(VmapAnglesForward(BatchedForward(batch_size)), VmapAngles(BatchedSweep(batch_size)), LaxMass())
        else:
            chain = _chain(VmapAnglesForward(VmapForward()), VmapAngles(VmapSweep()), LaxMass())
        return _scan(chain, checkpoint)

    @classmethod
    def default_multi_group_pallas_strategy(cls, batch_size : int | None = None, checkpoint : bool = True):
        if batch_size is not None:
            chain = _chain(VmapAnglesForward(BatchedForward(batch_size)), VmapAngles(BatchedPallasSweep(batch_size)), LaxMass())
        else:
            chain = _chain(VmapAnglesForward(VmapForward()), VmapAngles(VmapPallasSweep()), LaxMass())
        return _scan(chain, checkpoint)

    @classmethod
    def default_multi_group_warp_strategy(cls, batch_size : int | None = None, checkpoint : bool = True):
        chain = _chain(VmapAnglesForward(BatchedForward(batch_size)), WarpLevelAngles(), WarpMass())
        return _scan(chain, checkpoint)
