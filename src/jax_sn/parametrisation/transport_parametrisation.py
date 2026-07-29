from jax_sn.tools.parametrisation_tools import ParametrisedModule
import equinox as eqx

class TransportOperatorSigmaParametrised(ParametrisedModule):
    '''
    Parametrisation case for transport operators parameterised by the total cross section.
    '''
    @staticmethod
    def theta_base(operator ):
        return operator.data.spatial_data.domain.cross_sections_indexed.cross_section_data.total

    @staticmethod
    def reconstruct_operator(operator, theta):
        return eqx.tree_at(lambda op : op.data.spatial_data.domain.cross_sections_indexed.cross_section_data.total, operator, replace = theta)
    


