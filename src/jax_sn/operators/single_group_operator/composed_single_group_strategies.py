from ..operator_data import OperatorData
from ..transport_operator import MultiAngleTransportSpec, TransportOperatorMultiAngle
from ..moment_operator import MomentStrategy, DiscreteStrategy, MomentOperator, DiscreteOperator
from ..mass_operator import MassStrategy, MassOperator
from .single_group_strategies import SingleGroupStrategyBase, SingleGroupStrategy, SingleGroupChainStrategy, SingleGroupSourceStrategy, build_inscatter_operator


class ComposedSingleGroupBase(SingleGroupStrategyBase):
    '''Shared composition data + sub-operator builders for the composed single-group strategies. Not a
    strategy on its own (no ``mv_forward``); mixed into each concrete composed strategy.'''
    transport_spec    : MultiAngleTransportSpec
    moment_strategy   : MomentStrategy
    discrete_strategy : DiscreteStrategy
    mass_strategy     : MassStrategy

    def moment_operator(self, operator_data: OperatorData) -> MomentOperator:        
        return MomentOperator(operator_data.angular_data.wk_ylm, operator_data.elements_basis_shape, self.moment_strategy)

    def discrete_operator(self, operator_data: OperatorData) -> DiscreteOperator:        
        return DiscreteOperator(operator_data.angular_data.ylm, operator_data.elements_basis_shape, self.discrete_strategy)

    def mass_operator(self, transport_operator: TransportOperatorMultiAngle) -> MassOperator:        
        return MassOperator.from_transport_operator(transport_operator, self.mass_strategy)

    def transport_operator(self, operator_data: OperatorData, energy_group: int) -> TransportOperatorMultiAngle:
        return self.transport_spec.build(operator_data, energy_group)

class ComposedSingleGroupChainStrategy(SingleGroupChainStrategy, ComposedSingleGroupBase):

    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):    
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(transport_operator)
        return moment_operator.mv(transport_operator.inv(discrete_operator.mv(mass_operator.mv(x))))

    def mv_transpose(self, operator_data : OperatorData, energy_group: int, x):
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(transport_operator)        
        return mass_operator.transpose().mv(
            discrete_operator.transpose().mv(
                transport_operator.transpose().inv(
                    moment_operator.transpose().mv(x))))
    
class ComposedSingleGroupSourceStrategy(SingleGroupSourceStrategy, ComposedSingleGroupBase):

    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):    
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        mass_operator      = self.mass_operator(transport_operator)
        return moment_operator.mv(transport_operator.inv(mass_operator.mv(x)))

    def mv_transpose(self, operator_data : OperatorData, energy_group: int, x):
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        mass_operator      = self.mass_operator(transport_operator)        
        return mass_operator.transpose().mv(
            transport_operator.transpose().inv(
                moment_operator.transpose().mv(x)))
    
class ComposedSingleGroupStrategy(SingleGroupStrategy, ComposedSingleGroupBase):

    def chain_strategy(self) -> ComposedSingleGroupChainStrategy:
        return ComposedSingleGroupChainStrategy(self.transport_spec, self.moment_strategy, self.discrete_strategy, self.mass_strategy)

    def source_strategy(self) -> ComposedSingleGroupSourceStrategy:
        return ComposedSingleGroupSourceStrategy(self.transport_spec, self.moment_strategy, self.discrete_strategy, self.mass_strategy)

    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):
        inscatter_operator = build_inscatter_operator(operator_data, energy_group)
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(transport_operator)
        return x - moment_operator.mv(transport_operator.inv(discrete_operator.mv(mass_operator.mv(inscatter_operator.mv(x)))))
    
    def mv_transpose(self, operator_data : OperatorData, energy_group: int, x):
        inscatter_operator = build_inscatter_operator(operator_data, energy_group)        
        transport_operator = self.transport_operator(operator_data, energy_group)
        moment_operator    = self.moment_operator(operator_data)
        discrete_operator  = self.discrete_operator(operator_data)
        mass_operator      = self.mass_operator(transport_operator)        
        return x - inscatter_operator.transpose().mv(
            mass_operator.transpose().mv(
                discrete_operator.transpose().mv(
                    transport_operator.transpose().inv(
                        moment_operator.transpose().mv(x)))))
