import equinox as eqx
from abc import ABC, abstractmethod
import lineax as lx
from typing import Union, Callable, Dict, Type, Tuple
from jaxtyping import PyTree
from dataclasses import dataclass
import jax
import jax.numpy as jnp

class ParametrisedModule(ABC):
    '''
    Abstract base class for parametrised modules.

    Implementations provide methods to extract a parametrisation theta from a module,
    and to reconstruct the module from the parametrisation theta.

    This initially was created to support a abstract parametrisation framework for performance
    analysis of different operators.
    
    '''

    @staticmethod
    @abstractmethod
    def theta_base(self, module):
        ...

    @staticmethod
    @abstractmethod
    def reconstruct_operator(self, module, theta):
        ...
    

class ParametrisationValues(eqx.Module):
    ...

class ParametrisedObjective(eqx.Module):
        
    @abstractmethod
    def objective_and_constraints(self, theta : ParametrisationValues) -> Tuple[Union[float, PyTree]]:
        ...
        
class ParametrisedModuleTheta(eqx.Module):
    '''
    Abstract base class for module problem parametrisations.

    A problem parametrisation defines how to go from a set of parametrisation values (theta) 
    and a base module (containing e.g. settings and non-parametrised data) to a parametrised module.

    This adds a layer of abstraction on top of the module parametrisation. This is useful for example
    when the parametrisation values do not directly correspond to the module parametrisation theta,
    but need to be transformed first (e.g. when the parametrisation values are in log-space, or
    when the parametrisation values correspond to a different representation of the same data).

    A concrete example is that for a :class:`jax_sn.operators.multi_group_operator.MultiGroupOperator`,
    the module parametrisation may be given by :class:`MultigroupOperatorParametrisedVertices`, (i.e. in terms 
    of the vertices) but the problem parametrisation may define a mapping from a set of optimisation parameters to those vertices.


    This provides a interface to go from the optimisation parameters to the module theta via the module parametrisation.
    '''
    base_module            : eqx.Module
    '''The base module containing non-parametrised data and settings (and initial data possibly)'''
    
    @abstractmethod
    def to_module(self, theta : ParametrisationValues) -> eqx.Module:
        ...        

class ParametrisedModuleObjective(ParametrisedObjective):
    '''
    Objective defined on a parametrised module via a parametrised module theta.
    This is useful to recycle functions that will be the same for different specific parametrisations but similar values.

    A concrete example is that for a :class:`jax_sn.operators.multi_group_operator.MultiGroupOperator`,
    although the parametrisation is different (handled by different :class:`ParametrisedModuleTheta` subclasses) 
    and the objective may be different (e.g. different ways to go from solution to loss), the standard linear solve 
    from a non-parametrised right-hand-side is always the same.


    '''
    
    theta_parametrised_module : ParametrisedModuleTheta
    '''The parametrised module defining the mapping from parametrisation values to the module.'''
        
    @abstractmethod
    def objective_and_constraints(self, theta : ParametrisationValues) -> Tuple[Union[float, PyTree]]:
        ...

    @eqx.filter_jit
    def grad_objective_and_constraints(self, theta : ParametrisationValues) -> Tuple[Union[float, PyTree]]:
        '''
        Get the gradient of the objective and constraints with respect to the parametrisation values.

        Uses JAX reverse mode autodiff.

        Parameters:
        -------------
            theta: The parametrisation values.
        
        Returns:
        -------------
            The gradient of the objective and constraints.
        '''
        return jax.jacrev(self.objective_and_constraints, argnums=0)(theta)

    def to_module(self, theta : ParametrisationValues) -> eqx.Module:
        '''
        Get the parametrised module corresponding to the given parametrisation values.

        Parameters:
        -------------
            theta: The parametrisation values.

        Returns:
        -------------
            The parametrised module.
        '''
        return self.theta_parametrised_module.to_module(theta)



@dataclass
class ParametrisedScipyInterface:

    parametrised_objective : ParametrisedObjective
    _cache_theta           : ParametrisationValues = None
    _cache_result          : tuple = None

    def __init__(self, parametrised_objective : ParametrisedObjective):
        self.parametrised_objective = parametrised_objective
        self._cache_theta = None
        self._cache_result = None

    def evaluate_all(self, theta):
        theta_i = jnp.array(theta)
        if self._cache_theta is None or not jnp.allclose(theta_i, self._cache_theta):                                    
            self._cache_theta = jnp.copy(theta_i)            
            f_values, df_dtheta = Compute_Full(theta_i, self.base_parameters, self.static_opt_settings,
                                                self.array_opt_settings, self.global_settings, self.matrix_settings, self.f_obj, self.df_dtheta, self.df_dx)
            self._cache_result = (f_values[0], df_dtheta[0], f_values[1:], df_dtheta[1:])
            

        return self._cache_result

    def objective(self, x):
        obj, *_ = self.evaluate_all(x)
        return float(obj)

    def grad_objective(self, x):
        _, grad_obj, _, _ = self.evaluate_all(x)
        return jnp.array(grad_obj).astype(float)

    def constraints(self, x):
        _, _, cons, _ = self.evaluate_all(x)
        return [float(c) for c in cons]

    def jac_constraints(self, x):
        _, _, _, jac_cons = self.evaluate_all(x)
        return [jnp.array(j).astype(float) for j in jac_cons]