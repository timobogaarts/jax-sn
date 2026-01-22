
import equinox as eqx
import jax
import jax.numpy as jnp
from jax_sn.domain.cross_sections import CrossSectionData
from .base_parametrisation import ParametrisedModule, ParametrisedModuleObjective, ParametrisationValues, ParametrisedModuleTheta
from typing import Union
import lineax as lx
from jax_sn.operators.multigroup_operator import MultiGroupOperator
from jaxtyping import PyTree
from abc import abstractmethod

class VerticesXSTheta(eqx.Module):
    vertices : jnp.ndarray 
    cross_section_data: CrossSectionData


class MultigroupOperatorParametrisedVerticesXS(ParametrisedModule):
    '''
    Parametrisation case for total operators parameterised by the total and scattering cross section.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return VerticesXSTheta(operator.transport_operator.solution_domain_data.domain.vertices, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
    
    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : VerticesXSTheta) -> MultiGroupOperator:
        return reconstruct_multigroup_operator_including_toposort(operator, theta)



class MultigroupOperatorParametrisedVertices(ParametrisedModule):
    '''
    Parametrisation case for total operators parameterised by the total and scattering cross section.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.transport_operator.solution_domain_data.domain.vertices
    
    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : jnp.ndarray) -> MultiGroupOperator:
        vertices_xs_theta = VerticesXSTheta(theta, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
        return reconstruct_multigroup_operator_including_toposort(operator, vertices_xs_theta)
    


class MultigroupOperatorParametrisedObjective(ParametrisedModuleObjective):

    rhs : PyTree
    multi_group_solver : lx.AbstractLinearSolver

    def __check_init(self):
        assert isinstance(self.multi_group_solver, lx.AbstractLinearSolver), "multi_group_solver must be an instance of lx.AbstractLinearSolver"
                 
    def module_objective(self, module : eqx.Module) -> Union[float, PyTree]:
        rhs
        

    @abstractmethod
    def solution_values_to_objective(self, solution_values : PyTree) -> Union[float, PyTree]:
        ...



@eqx.filter_jit
def reconstruct_multigroup_operator_including_toposort(operator : MultiGroupOperator, theta : VerticesXSTheta) -> MultiGroupOperator:
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
