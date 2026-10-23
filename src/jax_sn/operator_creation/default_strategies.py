from ..operators.multi_group_operator.composed_multi_group_strategies import MultiGroupChainGatherScanStrategy
from ..operators.single_group_operator.composed_single_group_strategies import ComposedSingleGroupChainStrategy

from ..operators.mass_operator import LaxMass
from ..operators.moment_operator import EinsumFluxMoment
from ..operators.transport_operator import VmapAnglesInversion, VmapAnglesForward, BatchedSweep, VmapSweep, VmapPallasSweep, BatchedPallasSweep, VmapForward, BatchedForward
from ..operators.scatter_operator import BroadcastedSingleGroupGather


def _chain(transport_strategy, transport_inversion_strategy, mass_strategy):
    '''A ComposedSingleGroupChainStrategy with the einsum moment/discrete strategies.'''
    return ComposedSingleGroupChainStrategy(
        transport_strategy           = transport_strategy,
        transport_inversion_strategy = transport_inversion_strategy,
        moment_strategy              = EinsumFluxMoment(),
        discrete_strategy            = EinsumFluxMoment(),
        mass_strategy                = mass_strategy,
    )


def _scan(chain, checkpoint):
    '''Lift a single-group chain over the energy-group axis with a (checkpointed) scan and the
    broadcasted per-group gather-scatter.'''
    return MultiGroupChainGatherScanStrategy(chain, BroadcastedSingleGroupGather(), checkpoint=checkpoint)


class DefaultMultiGroupStrategy:

    @classmethod
    def default_multi_group_strategy(cls, batch_size : int | None = None, checkpoint = True):
        if batch_size is not None:
            chain = _chain(VmapAnglesForward(BatchedForward(batch_size)), VmapAnglesInversion(BatchedSweep(batch_size)), LaxMass())
        else:
            chain = _chain(VmapAnglesForward(VmapForward()), VmapAnglesInversion(VmapSweep()), LaxMass())
        return _scan(chain, checkpoint)

    @classmethod
    def default_multi_group_pallas_strategy(cls, batch_size : int | None = None, checkpoint : bool = True):
        if batch_size is not None:
            chain = _chain(VmapAnglesForward(BatchedForward(batch_size)), VmapAnglesInversion(BatchedPallasSweep(batch_size)), LaxMass())
        else:
            chain = _chain(VmapAnglesForward(VmapForward()), VmapAnglesInversion(VmapPallasSweep()), LaxMass())
        return _scan(chain, checkpoint)

    @classmethod
    def default_multi_group_warp_strategy(cls, batch_size : int | None = None, checkpoint : bool = True,
                                          maximum_number_of_levels : int | None = None):
        '''
        ``maximum_number_of_levels`` pins the width the leveled sweep pads its level arrays to; leave it
        ``None`` unless the sweep is rebuilt under a trace (see
        :attr:`~jax_sn.operators.transport_operator.implementations.transport_operator_warp.WarpLevelSweep.maximum_number_of_levels`).
        '''
        # Imported here: warp is an optional extra, so the module must import without it.
        from ..operators.transport_operator.implementations.transport_operator_warp import WarpLevelAngles
        from ..operators.mass_operator.mass_operator_warp import WarpMass
        chain = _chain(VmapAnglesForward(BatchedForward(batch_size)),
                       WarpLevelAngles(maximum_number_of_levels), WarpMass())
        return _scan(chain, checkpoint)
