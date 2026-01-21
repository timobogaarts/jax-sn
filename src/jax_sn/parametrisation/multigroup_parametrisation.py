
import equinox as eqx
import jax
import jax.numpy as jnp
from jax_sn.domain.cross_sections import CrossSectionData
from .base_parametrisation import ParametrizedModule
import lineax as lx


class VerticesXSTheta(eqx.Module):
    vertices : jnp.ndarray 
    cross_section_data: CrossSectionData


class MultigroupOperatorParametrisedVerticesXS(ParametrizedModule):
    '''
    Parametrisation case for total operators parameterised by the total and scattering cross section.
    '''
    @staticmethod
    def theta_base(operator ):
        return VerticesXSTheta(operator.transport_operator.solution_domain_data.domain.vertices, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
    
    @staticmethod
    def reconstruct_operator(operator, theta):
        return reconstruct_multigroup_operator_including_toposort(operator, theta)

@eqx.filter_jit
def reconstruct_multigroup_operator_including_toposort(operator, theta):
        import jax_sn

        # We have to replace:
        # 1. cross sections
        xs_idxed             = eqx.tree_at(lambda xs_idx : xs_idx.cross_section_data, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed, replace = theta.cross_section_data)                
        operator_xs_replaced =  eqx.tree_at(operator.cross_sections_indexed_loc, operator, replace_fn= lambda _ :  xs_idxed)

        # 2. domain itself
        domain                   = operator_xs_replaced.transport_operator.solution_domain_data.domain
        #       -  Face normals in the domain & vertices in the domain (although they don't do anything anymore i think...)
        face_normals             = jax_sn.domain.geometric_domain.compute_face_normal_vectors_from_3d_simplices(theta.vertices[domain.connectivity], domain.face_template)
        domain_replaced          = eqx.tree_at(lambda d : [d.vertices, d.face_normals], domain, replace = [theta.vertices, face_normals])        
        operator_xs_dom_replaced = eqx.tree_at(lambda op : op.transport_operator.transport_operator.solution_domain_data.domain, operator_xs_replaced, replace = domain_replaced)

        # 3. precomputed geometric data
        precomputed_geometric_data_comp = jax_sn.solution_domain.basix_fem._precompute_jacobians_simplex(domain_replaced)        
        operator_xs_dom_precomp_replaced =  eqx.tree_at(operator_xs_dom_replaced.precomputed_geometric_data_loc, operator_xs_dom_replaced, replace_fn= lambda _ : precomputed_geometric_data_comp)

        # 4. topological sorts [they are only used for inversion, so the gradient should not flow through them. In principle, we could completley remove it here, put it directly in the inverse (which doesn't get 'gradded' anyway, but this saves 
        # recomputation for multiple energy groups)]
        topological_sorts_replaced = jax.lax.stop_gradient(jax_sn.domain.TopologicalSortedAngle.from_domain_quadrature_set(domain_replaced, operator_xs_dom_precomp_replaced.moment_operator.quadrature_set))

        return eqx.tree_at(lambda op : op.transport_operator.transport_operator.topological_sorted_angle, operator_xs_dom_precomp_replaced, replace = topological_sorts_replaced)

