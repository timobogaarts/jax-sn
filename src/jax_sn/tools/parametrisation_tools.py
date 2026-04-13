import equinox as eqx
from abc import ABC, abstractmethod
import lineax as lx
from typing import Union, Callable, Dict, Type, Tuple
from jaxtyping import PyTree
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as onp

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
    f_theta_parametrisation_values : Callable[[jnp.ndarray], ParametrisationValues] 

    _cache_theta_val : jnp.ndarray = None
    _cache_val       : jnp.ndarray = None

    _cache_theta_jac : jnp.ndarray = None
    _cache_jac       : jnp.ndarray = None
    
    
    def __init__(self, parametrised_objective : ParametrisedObjective, f_theta_parametrisation_values : Tuple[Callable[[jnp.ndarray], ParametrisationValues], Callable[[ParametrisationValues], jnp.ndarray]]):
        self.parametrised_objective = parametrised_objective
        self.f_theta_parametrisation_values = f_theta_parametrisation_values
        self._cache_theta_val = None
        self._cache_result = None

    def evaluate_values(self, theta):
        theta = jnp.asarray(theta)

        print("Evaluating values for theta:", theta[0:10])        
        if self._cache_theta_val is None or not onp.array_equal(theta, self._cache_theta_val):
            print("\t cache miss, calculating")
            self._cache_theta_val = onp.array(theta).copy()
            pv = self.f_theta_parametrisation_values[0](theta)
            f_values = self.parametrised_objective.objective_and_constraints(pv)

            self._cache_val = onp.array(f_values).copy()
            print("\tValues: ", f_values)

        return self._cache_val
    
    def evaluate_jac(self, theta):
        theta = jnp.asarray(theta)
        print("Evaluating jac for theta:", theta[0:10])
        if self._cache_theta_jac is None or not onp.array_equal(theta, self._cache_theta_jac):
            print("\t cache miss, calculating")
            self._cache_theta_jac = onp.array(theta).copy()

            pv = self.f_theta_parametrisation_values[0](theta)
            df = self.parametrised_objective.grad_objective_and_constraints(pv)
            self._cache_jac = onp.array(self.f_theta_parametrisation_values[1](df)).copy()

        return self._cache_jac


    def objective(self, x):
        f, *_ = self.evaluate_values(x)
        return float(f)

    def constraints(self, x):
        _, *c = self.evaluate_values(x)
        return onp.asarray(c, float)

    def grad_objective(self, x):
        g, *_ = self.evaluate_jac(x)
        return onp.asarray(g, float)

    def jac_constraints(self, x):
        _, *J = self.evaluate_jac(x)
        return onp.asarray(J, float)

class DiagParametrisedScipyInterface:

    iteration_i : int = 0 
    n_in_iteration : int = 0
    save_dir : str = "iterations/"

    @classmethod 
    def from_scipy_interface(cls, scipy_interface : ParametrisedScipyInterface):
        return cls(scipy_interface.parametrised_objective, scipy_interface.f_theta_parametrisation_values)
    
    def evaluate_values(self, theta):
            theta = jnp.asarray(theta)
            
            print("Evaluating values for theta:", theta)        
            if self._cache_theta_val is None or not onp.array_equal(theta, self._cache_theta_val):
                
                jnp.save(self.save_dir + f"theta_{self.iteration_i}_{self.n_in_iteration}.npy", theta)
                self._cache_theta_val = onp.array(theta).copy()
                pv = self.f_theta_parametrisation_values[0](theta)
                f_values = self.parametrised_objective.objective_and_constraints(pv)
                self._cache_val = onp.array(f_values).copy()
                print("\tValues: ", f_values)
                jnp.save(self.save_dir + f"f_values_{self.iteration_i}_{self.n_in_iteration}.npy", f_values)
                self.n_in_iteration += 1
            return self._cache_val
        
    def evaluate_jac(self, theta):
        theta = jnp.asarray(theta)
        print("Evaluating jac for theta:", theta[0:10])
        if self._cache_theta_jac is None or not onp.array_equal(theta, self._cache_theta_jac):
            print("\t cache miss, calculating")
            jnp.save(self.save_dir + f"theta_{self.iteration_i}_{self.n_in_iteration}_jac.npy", theta)
            self._cache_theta_jac = onp.array(theta).copy()
            pv = self.f_theta_parametrisation_values[0](theta)
            df = self.parametrised_objective.grad_objective_and_constraints(pv)
            self._cache_jac = onp.array(self.f_theta_parametrisation_values[1](df)).copy()
            jnp.save(self.save_dir + f"jac_{self.iteration_i}_{self.n_in_iteration}.npy", self._cache_jac)

        return self._cache_jac   
    


def ks_agg(x, alpha=50.0):
    n = x.shape[0]
    m = jnp.max(x)
    return m + jnp.log(jnp.sum(jnp.exp(alpha * (x - m)))) / alpha - jnp.log(n) / alpha
