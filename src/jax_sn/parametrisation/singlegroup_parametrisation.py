from jax_sn.tools.parametrisation_tools import ParametrisedModule
import equinox as eqx
class SingleGroupOperatorSigmaAllParametrised(ParametrisedModule):
    '''
    Parametrisation case for total operators parameterised by the total and scattering cross section.
    '''
    @staticmethod
    def theta_base(operator ):
        return operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data
    
    @staticmethod
    def reconstruct_operator(operator, theta):
        return eqx.tree_at(lambda op : [op.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data, op.inscatter_operator.cross_sections_indexed.cross_section_data], operator, replace = [theta,theta])
    