from jax_sn.tools.parametrisation_tools import ParametrisedModule
import equinox as eqx
class SingleGroupOperatorSigmaAllParametrised(ParametrisedModule):
    '''
    Parametrisation case for total operators parameterised by the total and scattering cross section.
    '''
    @staticmethod
    def theta_base(operator ):
        return operator.setup.spatial_data.domain.cross_sections_indexed.cross_section_data

    @staticmethod
    def reconstruct_operator(operator, theta):
        # Single canonical location now: the within-group scatter operator derives its cross
        # sections from the same `setup`, so one tree_at updates both transport and scatter.
        return eqx.tree_at(lambda op : op.setup.spatial_data.domain.cross_sections_indexed.cross_section_data, operator, replace = theta)
    