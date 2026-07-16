from ..operator_data import OperatorData
from ..transport_operator import (
    TransportData, TransportOperatorMultiAngle,
    MultiAngleForwardStrategy, MultiAngleInversionStrategy,
)
from ..moment_operator import FluxMomentStrategy, MomentOperator, DiscreteOperator, MomentData, DiscreteData
from ..mass_operator import MassStrategy, MassOperator, MassData
from ..scatter_operator import ScatterStrategy, ScatterOperator, ScatterData
from .single_group_operators_base import (
    TransposableStrategy, SingleGroupStrategy, SingleGroupChainStrategy, SingleGroupSourceStrategy, SingleGroupData,
    single_group_moment_structure,
)
from typing import Tuple
'''
Helpers for building operators from single_group_data. This might be used in other places as well.
'''
def _build_moment_operator(single_group_data: SingleGroupData, strategy: FluxMomentStrategy) -> MomentOperator:    
    return MomentOperator(MomentData(single_group_data.operator_data.angular_data.wk_ylm), single_group_data.operator_data.elements_basis_shape, strategy)

def _build_discrete_operator(single_group_data: SingleGroupData, strategy: FluxMomentStrategy) -> DiscreteOperator:
    '''The moment-to-discrete operator :math:`\\mathbf{D}` from ``operator_data``'s harmonics.'''
    return DiscreteOperator(DiscreteData(single_group_data.operator_data.angular_data.ylm), single_group_data.operator_data.elements_basis_shape, strategy)


def _build_mass_operator(single_group_data: SingleGroupData, strategy: MassStrategy) -> MassOperator:
    '''The mass operator :math:`\\mathbf{H}` from ``operator_data``'s FEM matrices + geometric factors.'''
    spatial_data = single_group_data.operator_data.spatial_data
    return MassOperator(MassData(spatial_data.fem_matrices, spatial_data.precomputed_geometric_data),
                        (single_group_data.operator_data.n_angles,), strategy)


def _build_inscatter_operator(single_group_data: SingleGroupData, energy_group: int, strategy: ScatterStrategy) -> ScatterOperator:
    r'''The within-group scatter :math:`\mathbf{S}_{gg}`, from ``operator_data``'s cross sections.'''
    return ScatterOperator(
        ScatterData(energy_group, energy_group, single_group_data.operator_data.domain.cross_sections_indexed),
        strategy,
        single_group_moment_structure(single_group_data),
    )

def _build_transport_operator(single_group_data: SingleGroupData, forward_strategy: MultiAngleForwardStrategy,
                              inversion_strategy: MultiAngleInversionStrategy) -> TransportOperatorMultiAngle:
    '''The transport operator :math:`\\mathbf{T}_g` for ``energy_group``, from ``operator_data`` + the two strategies.'''
    return TransportOperatorMultiAngle(TransportData(single_group_data.operator_data.spatial_data, single_group_data.energy_group, single_group_data.operator_data.angular_data.quadrature_set.angles, single_group_data.operator_data.sweep_data),
                                       forward_strategy, inversion_strategy)

class ComposedSingleGroupBase(TransposableStrategy):
    '''
    Composed of multiple transport operators.
    '''
    transport_strategy           : MultiAngleForwardStrategy
    transport_inversion_strategy : MultiAngleInversionStrategy
    moment_strategy              : FluxMomentStrategy
    discrete_strategy            : FluxMomentStrategy
    mass_strategy                : MassStrategy
    scatter_strategy             : ScatterStrategy
    
    def all_operators(self, single_group_data : SingleGroupData) -> Tuple[TransportOperatorMultiAngle, MomentOperator, DiscreteOperator, MassOperator, ScatterOperator]:
        '''
        Construct the entire chain of operators. This is a free construction in any case (after JIT): just assigning a type.
        '''
        transport_operator = _build_transport_operator(single_group_data, self.transport_strategy, self.transport_inversion_strategy)
        moment_operator    = _build_moment_operator(single_group_data, self.moment_strategy)
        discrete_operator  = _build_discrete_operator(single_group_data, self.discrete_strategy)
        mass_operator      = _build_mass_operator(single_group_data, self.mass_strategy)
        scatter_operator   = _build_inscatter_operator(single_group_data, single_group_data.energy_group, self.scatter_strategy)
        return transport_operator, moment_operator, discrete_operator, mass_operator, scatter_operator
    
    def default_sweep_data(self):
        return self.transport_inversion_strategy.default_sweep_data_type()

class ComposedSingleGroupChainStrategy(ComposedSingleGroupBase, SingleGroupChainStrategy):

    def forward(self, data : SingleGroupData, x):    
        transport, moment, discrete, mass, _ = self.all_operators(data)
        return moment.mv(transport.inv(discrete.mv(mass.mv(x))))

    def transpose(self, data : SingleGroupData, x):
        transport, moment, discrete, mass, _ = self.all_operators(data)
        return mass.transpose().mv(
                discrete.transpose().mv(
                transport.transpose().inv(
                    moment.transpose().mv(x))))
    
class ComposedSingleGroupSourceStrategy(ComposedSingleGroupBase, SingleGroupSourceStrategy):

    def forward(self, data : SingleGroupData, x):    
        transport, moment, _, mass, _ = self.all_operators(data)
        return moment.mv(transport.inv(mass.mv(x)))        

    def transpose(self, data : SingleGroupData, x):
        transport, moment, _, mass, _ = self.all_operators(data)
        return mass.transpose().mv(transport.transpose().inv(moment.transpose().mv(x)))
    
class ComposedSingleGroupStrategy(ComposedSingleGroupBase, SingleGroupStrategy):
    

    def forward(self, data : SingleGroupData, x):
        transport, moment, discrete, mass, inscatter = self.all_operators(data)        
        return x - moment.mv(transport.inv(discrete.mv(mass.mv(inscatter.mv(x)))))
    
    def transpose(self, data : SingleGroupData, x):
        transport, moment, discrete, mass, inscatter = self.all_operators(data)            
        return x - inscatter.transpose().mv(
            mass.transpose().mv(
                discrete.transpose().mv(
                    transport.transpose().inv(
                        moment.transpose().mv(x)))))
