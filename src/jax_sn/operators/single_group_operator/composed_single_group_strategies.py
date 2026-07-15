from ..operator_data import OperatorData
from ..transport_operator import MultiAngleTransportSpec, TransportOperatorMultiAngle
from ..moment_operator import FluxMomentStrategy, MomentOperator, DiscreteOperator, MomentData, DiscreteData
from ..mass_operator import MassStrategy, MassOperator, MassData
from ..scatter_operator import ScatterStrategy, ScatterOperator, ScatterData
from .single_group_operators_base import (
    SingleGroupStrategyBase, SingleGroupStrategy, SingleGroupChainStrategy, SingleGroupSourceStrategy,
    single_group_moment_structure,
)


class ComposedSingleGroupBase(SingleGroupStrategyBase):
    '''
    Composed of multiple transport operators.
    '''
    transport_spec    : MultiAngleTransportSpec
    moment_strategy   : FluxMomentStrategy
    discrete_strategy : FluxMomentStrategy
    mass_strategy     : MassStrategy
    scatter_strategy  : ScatterStrategy

    def moment_operator(self, operator_data: OperatorData) -> MomentOperator:
        return MomentOperator(MomentData(operator_data.angular_data.wk_ylm), operator_data.elements_basis_shape, self.moment_strategy)

    def discrete_operator(self, operator_data: OperatorData) -> DiscreteOperator:
        return DiscreteOperator(DiscreteData(operator_data.angular_data.ylm), operator_data.elements_basis_shape, self.discrete_strategy)

    def mass_operator(self, operator_data: OperatorData) -> MassOperator:
        spatial_data = operator_data.spatial_data
        return MassOperator(MassData(spatial_data.fem_matrices, spatial_data.precomputed_geometric_data),
                            (operator_data.n_angles,), self.mass_strategy)

    def inscatter_operator(self, operator_data: OperatorData, energy_group: int) -> ScatterOperator:
        r'''The within-group scatter :math:`\mathbf{S}_{gg}`, from ``operator_data``'s cross sections.'''
        return ScatterOperator(
            ScatterData(energy_group, energy_group, operator_data.domain.cross_sections_indexed),
            self.scatter_strategy,
            single_group_moment_structure(operator_data),
        )

    def transport_operator(self, operator_data: OperatorData, energy_group: int) -> TransportOperatorMultiAngle:
        return self.transport_spec.build(operator_data, energy_group)


class ComposedSingleGroupChainStrategy(SingleGroupChainStrategy, ComposedSingleGroupBase):

    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):    
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(operator_data)
        return moment_operator.mv(transport_operator.inv(discrete_operator.mv(mass_operator.mv(x))))

    def mv_transpose(self, operator_data : OperatorData, energy_group: int, x):
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(operator_data)        
        return mass_operator.transpose().mv(
            discrete_operator.transpose().mv(
                transport_operator.transpose().inv(
                    moment_operator.transpose().mv(x))))
    
class ComposedSingleGroupSourceStrategy(SingleGroupSourceStrategy, ComposedSingleGroupBase):

    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):    
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        mass_operator      = self.mass_operator(operator_data)
        return moment_operator.mv(transport_operator.inv(mass_operator.mv(x)))

    def mv_transpose(self, operator_data : OperatorData, energy_group: int, x):
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        mass_operator      = self.mass_operator(operator_data)        
        return mass_operator.transpose().mv(
            transport_operator.transpose().inv(
                moment_operator.transpose().mv(x)))
    
class ComposedSingleGroupStrategy(SingleGroupStrategy, ComposedSingleGroupBase):

    def chain_strategy(self) -> ComposedSingleGroupChainStrategy:
        return ComposedSingleGroupChainStrategy(self.transport_spec, self.moment_strategy, self.discrete_strategy, self.mass_strategy, self.scatter_strategy)

    def source_strategy(self) -> ComposedSingleGroupSourceStrategy:
        return ComposedSingleGroupSourceStrategy(self.transport_spec, self.moment_strategy, self.discrete_strategy, self.mass_strategy, self.scatter_strategy)

    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):
        inscatter_operator = self.inscatter_operator(operator_data, energy_group)
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(operator_data)
        return x - moment_operator.mv(transport_operator.inv(discrete_operator.mv(mass_operator.mv(inscatter_operator.mv(x)))))
    
    def mv_transpose(self, operator_data : OperatorData, energy_group: int, x):
        inscatter_operator = self.inscatter_operator(operator_data, energy_group)
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(operator_data)        
        return x - inscatter_operator.transpose().mv(
            mass_operator.transpose().mv(
                discrete_operator.transpose().mv(
                    transport_operator.transpose().inv(
                        moment_operator.transpose().mv(x)))))
