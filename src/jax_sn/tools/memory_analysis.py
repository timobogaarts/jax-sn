import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import numpy as onp 
import lineax as lx



def compiled_to_dict_mb(compiled):
    return {
        "peak" : compiled.memory_analysis().peak_memory_in_bytes / 1024**2,
        "argument" : compiled.memory_analysis().argument_size_in_bytes / 1024**2,
        "output" : compiled.memory_analysis().output_size_in_bytes / 1024**2,
        "temp"  : compiled.memory_analysis().temp_size_in_bytes / 1024**2,
    }

def _filter_jit_f_to_memory(f, *args):
    lowered     = f.lower(*args)  # warm up    
    compiled    = lowered.compile()
    dict_memory = compiled_to_dict_mb(compiled.compiled)
    return dict_memory

def array_bytes(x):
    if hasattr(x, "dtype") and hasattr(x, "shape"):
        return onp.prod(x.shape) * x.dtype.itemsize
    else:
        return 0

def pytree_nbytes(pytree):
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    total = 0
    for leaf in leaves:
        if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
            total += array_bytes(leaf)
    return total

#============================================================================================================
#                                             Gradient MV
#============================================================================================================

def _theta_op_apply(operator, theta, x_in, _construct_theta_op, method_name):
    return jnp.sum(getattr(_construct_theta_op(operator, theta), method_name)(x_in))

def _make_theta_op(method):
    '''
    Creates functions to compute particular theta-parameterised operator actions and their gradients.
    Uses a particular method of the linear operator, applies it to some input and sums the output to get a scalar.

    Parameters:
    ----------
    method : str
        The method of the operator to use, e.g. "mv" or "inv".    

    Returns:
    -------
    _theta_op_eqx : function
        JIT-compiled function that computes the summed output of the operator action.
    _theta_op_eqx_valgrad : function
        JIT-compiled function that computes the value and gradient of the summed output with respect to theta.

    '''
    @eqx.filter_jit
    def _theta_op_eqx(operator, theta, x_in, _construct_theta_op):
        return jnp.sum(getattr(_construct_theta_op(operator, theta), method)(x_in))
    
    _theta_op_eqx_valgrad = eqx.filter_jit(jax.value_and_grad(_theta_op_eqx, argnums = 1))

    return _theta_op_eqx, _theta_op_eqx_valgrad

def make_memory_parametrised_grad(method_name):

    theta_op_eqx, theta_op_eqx_valgrad = _make_theta_op(method_name)

    def memory_param_grad(operator, parametrisation_case):
        x_in = jnp.ones_like(operator.in_structure())
        theta_base = parametrisation_case.theta_base(operator)
        args = (operator, theta_base, x_in, parametrisation_case.reconstruct_operator)
        parametric_op      =  _filter_jit_f_to_memory(theta_op_eqx, *args)    
        parametric_op_grad =  _filter_jit_f_to_memory(theta_op_eqx_valgrad, *args)
        return parametric_op, parametric_op_grad
    return memory_param_grad



# ============================================================================================================
#                                             Timing utilities
# ============================================================================================================
from typing import Type, Tuple, Dict
def custom_timeit(func, number=100, ignore_n : int = 0):
    import time
    times = []
    for _ in range(number):
        start = time.time()
        func().block_until_ready()
        end = time.time()
        times.append(end - start)

    times = times[ignore_n:]  # ignore first n runs
    return onp.mean(times), onp.std(times)

from abc import ABC, abstractmethod


class OperatorAnalysis(ABC):

    @staticmethod
    def analyze_memory(operator : lx.AbstractLinearOperator, method : str):
        '''
        Analyzes memory usage for a operator method (mv or inv).
        First it initializes the operator with the provided arguments and keyword arguments.

        It uses _make_op to create a function that applies the specified method of the operator some input (just ones, for memory it does not matter).        

        This function is then lowered and compiled and its memory usage is analyzed using _filter_jit_f_to_memory.

        Parameters:
        ----------
        operator : lx.AbstractLinearOperator
            The linear operator to analyze.
        method : str
            The method of the operator to use, e.g. "mv" or "inv".
        
        Returns:
        -------
        mem_mv : Dict[str, float]
            Memory usage statistics for the operator method [in MB].
        '''        
        x_in = jnp.ones_like(operator.in_structure()) 
        

        @eqx.filter_jit
        def method_f(operator, x_in):
            return getattr(operator, method)(x_in)
                
        mem_mv  = _filter_jit_f_to_memory(method_f, operator, x_in)

        return mem_mv
    

    @staticmethod
    def analyze_timing(operator : lx.AbstractLinearOperator, method : str, n_samples : int = 100, ignore_n : int = 1):
        '''
        Analyzes timing for a operator method.

        It uses _make_op to create a function that applies the specified method of the operator some input (just ones, for timing it does not matter).        

        The function is then timed using custom_timeit, and the result is returned.

        Parameters:
        ----------
        operator : lx.AbstractLinearOperator
            The linear operator to analyze.
        method : str
            The method of the operator to use, e.g. "mv" or "inv".
        n_samples : int, optional
            The number of samples to use for timing, by default 100.
        ignore_n : int, optional
            The number of initial runs to ignore for timing (e.g. for compilation caching), by default 1.

        Returns:
        -------
        duration_mv : float
            The average time taken for the operator method.
        '''
        
        x_in = jnp.ones_like(operator.in_structure()) 

        @eqx.filter_jit
        def method_f(operator, x_in):
            return getattr(operator, method)(x_in)
                
        def run_method():
            return method_f(operator, x_in).block_until_ready()
                        
        duration_mv = custom_timeit( run_method, number = n_samples , ignore_n=ignore_n)[0]                
        return duration_mv

class OperatorThetaParametrised(ABC):
    @staticmethod
    @abstractmethod
    def theta_base(operator ):
        ...

    @staticmethod
    @abstractmethod
    def reconstruct_operator(operator, theta):
        ...

    @classmethod
    def analyze_grad_memory(cls, operator : lx.AbstractLinearOperator, method : str):
        '''
        Analyzes memory usage for the gradient of a parameterised operator method (mv or inv).
        First it initializes the operator with the provided arguments and keyword arguments.

        It uses _make_theta_op to create a function that applies the specified method of the operator some input (just ones, for memory it does not matter) and sums the output.
        This gives a scalar output, allowing the use of jax.value_and_grad to compute the gradient with respect to the parameter theta.

        This function is then lowered and compiled and its memory usage is analyzed using _filter_jit_f_to_memory. This is finally printed.

        Parameters:
        ----------
        operator : lx.AbstractLinearOperator
            The linear operator to analyze.
        method : str
            The method of the operator to use, e.g. "mv" or "inv".
        
        Returns:
        -------
        mem_mv : Dict[str, float]
            Memory usage statistics for the operator method [in MB].
        mem_grad_mv : Dict[str, float]
            Memory usage statistics for the gradient of the operator method [in MB].
        '''        
        mem_mv, mem_grad_mv  = make_memory_parametrised_grad(method)(operator, cls)

        return mem_mv, mem_grad_mv
        

    @classmethod
    def analyze_grad_timing(cls, operator : lx.AbstractLinearOperator, method : str, n_samples : int = 100, ignore_n : int = 1):        
        '''
        Analyzes timing for the gradient of a parameterised operator method.

        It uses _make_theta_op to create a function that applies the specified method of the operator some input (just ones, for timing it does not matter) and sums the output.
        This gives a scalar output, allowing the use of jax.value_and_grad to compute the gradient with respect to the parameter theta.

        The function and its gradient are then timed using custom_timeit, and the results are returned.

        Parameters:
        ----------
        operator : lx.AbstractLinearOperator
            The linear operator to analyze.
        method : str
            The method of the operator to use, e.g. "mv" or "inv".
        n_samples : int, optional
            The number of samples to use for timing, by default 100.
        ignore_n : int, optional
            The number of initial runs to ignore for timing (e.g. for compilation caching), by default 1.

        Returns:
        -------
        duration_mv : float
            The average time taken for the operator method.
        duration_grad_mv : float
            The average time taken for the gradient of the operator method.
        '''
        
        x_in = jnp.ones_like(operator.in_structure()) 

        args_method = (operator, cls.theta_base(operator), x_in, cls.reconstruct_operator)

        method_f, grad_method_f = _make_theta_op(method)
        
        def run_method():
            return method_f(*args_method).block_until_ready()
                
        duration_mv = custom_timeit( run_method, number = n_samples , ignore_n=ignore_n)[0]        
        def run_grad_method():
            return grad_method_f(*args_method)[1].block_until_ready()   

        duration_grad_mv = custom_timeit( run_grad_method, number=n_samples, ignore_n=ignore_n)[0]
        
        return duration_mv, duration_grad_mv

#============================================================================================================
#                                             Transport operator specific
#============================================================================================================

class TransportOperatorSigmaParametrised(OperatorThetaParametrised):
    @staticmethod
    def theta_base(operator ):
        return operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.total
    
    @staticmethod
    def reconstruct_operator(operator, theta):
        return eqx.tree_at(lambda op : op.solution_domain_data.domain.cross_sections_indexed.cross_section_data.total, operator, replace = theta)
    


