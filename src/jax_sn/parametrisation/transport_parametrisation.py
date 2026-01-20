from .base_parametrisation import ParametrizedModule
import equinox as eqx

class TransportOperatorSigmaParametrised(ParametrizedModule):
    '''
    Parametrisation case for transport operators parameterised by the total cross section.
    '''
    @staticmethod
    def theta_base(operator ):
        return operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.total
    
    @staticmethod
    def reconstruct_operator(operator, theta):
        return eqx.tree_at(lambda op : op.solution_domain_data.domain.cross_sections_indexed.cross_section_data.total, operator, replace = theta)
    


