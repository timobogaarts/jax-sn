
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
        return VerticesXSTheta(operator.setup.solution_domain_data.domain.vertices, operator.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data)

    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : VerticesXSTheta) -> MultiGroupOperator:
        return reconstruct_multigroup_operator_including_toposort(operator, theta)


class MultigroupOperatorParametrisedXS(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by both the vertices and the cross section data.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data

    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : CrossSectionData) -> MultiGroupOperator:
        return reconstruct_multigroup_operator_xs(operator, theta)




class MultigroupOperatorParametrisedVertices(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by the vertices only.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.setup.solution_domain_data.domain.vertices

    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : jnp.ndarray) -> MultiGroupOperator:
        vertices_xs_theta = VerticesXSTheta(theta, operator.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
        return reconstruct_multigroup_operator_including_toposort(operator, vertices_xs_theta)

class MultigroupOperatorParametrisedVerticesNoToposort(ParametrisedModule):
    '''
    Parametrisation case for a MultiGroupOperator parameterised by the vertices only, without replacing the topological sort. This is only
    useful in 1D or when computing sensitivities without changing the geometry.
    '''
    @staticmethod
    def theta_base(operator : MultiGroupOperator ):
        return operator.setup.solution_domain_data.domain.vertices

    @staticmethod
    def reconstruct_operator(operator : MultiGroupOperator, theta : jnp.ndarray) -> MultiGroupOperator:
        vertices_xs_theta = VerticesXSTheta(theta, operator.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data)
        return reconstruct_multigroup_operator_including_toposort(operator, vertices_xs_theta, False)

class MultigroupOperatorParametrisedObjective(ParametrisedModuleObjective):

    rhs : PyTree
    multi_group_solver : lx.AbstractLinearSolver

    def __check_init__(self):
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
        # Single canonical location: transport + gather-scatter both derive their cross sections from `setup`.
        return eqx.tree_at(lambda op : op.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data, operator, replace = cross_section_data)


@eqx.filter_jit
def reconstruct_multigroup_operator_including_toposort(operator : MultiGroupOperator, theta : VerticesXSTheta, replace_topological_sort : bool = True) -> MultiGroupOperator:
        import jax_sn

        # Everything the sub-operators draw on lives in one canonical `setup`, so each replacement is a
        # single tree_at on a canonical location (no *_loc tuples, no per-operator special-casing).
        domain = operator.setup.solution_domain_data.domain

        # 1. + 2. cross sections, vertices, and the face normals derived from them -- in one domain rebuild.
        face_normals    = jax_sn.domain.geometric_domain.compute_face_normal_vectors_from_3d_simplices(theta.vertices[domain.connectivity], domain.face_template)
        domain_replaced = eqx.tree_at(
            lambda d : [d.vertices, d.face_normals, d.cross_sections_indexed.cross_section_data],
            domain,
            replace = [theta.vertices, face_normals, theta.cross_section_data])
        operator_dom = eqx.tree_at(lambda op : op.setup.solution_domain_data.domain, operator, replace = domain_replaced)

        # 3. precomputed geometric data
        precomputed_geometric_data_comp = jax_sn.solution_domain.basix_fem._precompute_jacobians_simplex(domain_replaced)
        operator_precomp = eqx.tree_at(lambda op : op.setup.solution_domain_data.precomputed_geometric_data, operator_dom, replace = precomputed_geometric_data_comp)

        # 4. sweep data [only used for inversion, so the gradient should not flow through it -- hence the
        # stop_gradient. In some cases we don't replace it: only changing cross-sections, or a 1D geometry
        # where the sweep is fixed anyway.]
        if replace_topological_sort:
            # The sweep data knows how to recompute itself for the deformed domain, reusing its own stored
            # angles and preserving its concrete type. So this needs no operator/sweep-class special-casing.
            sweep_data_replaced = jax.lax.stop_gradient(operator.setup.sweep_data.reconstruct(domain_replaced))
            return eqx.tree_at(lambda op : op.setup.sweep_data, operator_precomp, replace = sweep_data_replaced)
        else:
            return operator_precomp
