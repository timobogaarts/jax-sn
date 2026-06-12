
import equinox as eqx
import jax
import jax.numpy as jnp
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.tools.parametrisation_tools import ParametrisedModule, ParametrisedModuleObjective, ParametrisationValues, ParametrisedModuleTheta
from typing import Union
import lineax as lx
from jax_sn.operators.multi_group_operator import MultiGroupOperator
from jaxtyping import PyTree
from abc import abstractmethod
from lineax._misc import structure_equal
from jax_sn.operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
from typing import Tuple

class VerticesXSTheta(eqx.Module):
    vertices : jnp.ndarray 
    cross_section_data: CrossSectionData


class MultigroupOperatorParametrisedVerticesXS(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by both the vertices and the cross section data.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return VerticesXSTheta(operator.transport_operator.solution_domain_data.domain.vertices, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
    
    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : VerticesXSTheta) -> MultiGroupOperator:
        return reconstruct_multigroup_operator_including_toposort(operator, theta)
    

class MultigroupOperatorParametrisedXS(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by both the vertices and the cross section data.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data
    
    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : CrossSectionData) -> MultiGroupOperator:
        return reconstruct_multigroup_operator_xs(operator, theta)




class MultigroupOperatorParametrisedVertices(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by the vertices only.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.transport_operator.solution_domain_data.domain.vertices
    
    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : jnp.ndarray) -> MultiGroupOperator:
        vertices_xs_theta = VerticesXSTheta(theta, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
        return reconstruct_multigroup_operator_including_toposort(operator, vertices_xs_theta)
    
class MultigroupOperatorParametrisedVerticesNoToposort(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by the vertices only, without replacing the topological sort. This is only 
    useful in 1D or when computing sensitivities without changing the geometry.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.transport_operator.solution_domain_data.domain.vertices
    
    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : jnp.ndarray) -> MultiGroupOperator:
        vertices_xs_theta = VerticesXSTheta(theta, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
        return reconstruct_multigroup_operator_including_toposort(operator, vertices_xs_theta, False)
    
class MultigroupOperatorParametrisedObjective(ParametrisedModuleObjective):

    rhs : PyTree
    multi_group_solver : lx.AbstractLinearSolver

    def __check_init(self):
        assert isinstance(self.multi_group_solver, lx.AbstractLinearSolver), "multi_group_solver must be an instance of lx.AbstractLinearSolver"
        assert structure_equal(self.rhs, self.theta_parametrised_module.base_module.in_structure()), "rhs must have the same structure as the module's in_structure"

    @abstractmethod
    def objective_and_constraints(self, theta : ParametrisationValues) -> Tuple[Union[float, PyTree]]:
        ...
    
    def _to_solution(self, mgop_theta : MultiGroupOperator) -> PyTree:        
        '''
        Internal function to solve the multi-group operator for given :class:`MultiGroupOperator`.

        Parameters
        ----------
        mgop_theta : MultiGroupOperator
            The parametrised multi-group operator.
        Returns
        -------
        PyTree
            The solution PyTree.
        '''
        return _solve_multigroup_operator_rhs_scan(mgop_theta, self.rhs, self.multi_group_solver)
        
    def solution(self, theta : ParametrisationValues) -> PyTree:
        '''
        Obtain the solution for a given parametrisation values theta.
        '''
        mgop = self.to_module(theta)        
        return self._to_solution(mgop)



@eqx.filter_jit
def reconstruct_multigroup_operator_xs(operator : MultiGroupOperator, cross_section_data : CrossSectionData) -> MultiGroupOperator:
        xs_idxed             = eqx.tree_at(lambda xs_idx : xs_idx.cross_section_data, operator.transport_operator.solution_domain_data.domain.cross_sections_indexed, replace = cross_section_data)                
        operator_xs_replaced =  eqx.tree_at(operator.cross_sections_indexed_loc, operator, replace_fn= lambda _ :  xs_idxed)
        return operator_xs_replaced

    

@eqx.filter_jit
def reconstruct_multigroup_operator_including_toposort(operator : MultiGroupOperator, theta : VerticesXSTheta, replace_topological_sort : bool = True) -> MultiGroupOperator:
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
        # In some cases we don't want to replace the topological sort, e.g. when we are only changing cross-sections but not the geometry or when we are using a 1D geometry where the toposort is fixed anyway.
        if replace_topological_sort:
            # The toposort knows how to recompute itself for the deformed domain, reusing its own stored
            # angles and preserving its concrete type (a level-parallel toposort rebuilds its level setup
            # with the same padding). So this needs no operator/toposort-class special-casing.
            existing_toposort = operator.transport_operator.transport_operator.topological_sorted_angle
            topological_sorts_replaced = jax.lax.stop_gradient(existing_toposort.reconstruct(domain_replaced))
            return eqx.tree_at(lambda op : op.transport_operator.transport_operator.topological_sorted_angle, operator_xs_dom_precomp_replaced, replace = topological_sorts_replaced)
        else:
            return operator_xs_dom_precomp_replaced
