'''Data a transport operator hands to its forward / inversion strategies.'''

import jax.numpy as jnp
import equinox as eqx

from jax_sn.solution_domain import SpatialData
from jax_sn.domain.cross_sections import CrossSectionDataSingleGroup
from jax_sn.operators.base_linear_operator import TransposableOperatorData
from .sweep_data_implementations import SweepData


class TransportData(TransposableOperatorData):
    r'''
    Data a transport operator hands to its forward / inversion strategies.
    '''

    spatial_data : SpatialData
    '''Geometry, FEM matrices, geometric factors and indexed cross sections. [differentiable]'''

    energy_group : int
    '''Energy group index. Selects the cross sections; see :attr:`cross_sections`.'''

    angle        : jnp.ndarray
    r'''Transport direction(s) :math:`\Omega`. Shape ``[n_dim]``, or ``[n_angles, n_dim]`` when batched over angles.'''

    sweep_data   : SweepData
    '''Per-angle topological ordering. Ignored by forward strategies.'''

    @property
    def cross_sections(self) -> CrossSectionDataSingleGroup:
        '''Cross sections mapped to full element array for this energy group. Shape [n_elements].'''
        return self.spatial_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group)

    @property
    def forward_args(self):
        '''Positional args of a forward kernel (unpacked with ``*data.forward_args``).'''
        return (*self.spatial_data, self.cross_sections, self.angle)

    @property
    def inversion_args(self):
        '''Positional args of an inverse-sweep kernel (unpacked with ``*data.inversion_args``).'''
        return (*self.forward_args, self.sweep_data)

    @property
    def transposed(self) -> "TransportData":
        '''Data of the transposed operator: the reversed direction. The ordering's reverse is the
        partner of negating the angle. See :ref:`here <transport_operator_transposition>`.'''
        return eqx.tree_at(lambda a : [a.angle, a.sweep_data], self, (-self.angle, self.sweep_data.reverse))

    def single_angle(self, idx) -> "TransportData":
        '''Slice the angle and the sweep ordering in lockstep down to angle ``idx``.'''
        return eqx.tree_at(lambda a : [a.angle, a.sweep_data], self, (self.angle[idx], self.sweep_data[idx]))
