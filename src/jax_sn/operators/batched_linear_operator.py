
import jax 
import lineax as lx 
import jax.numpy as jnp
from .base_linear_operator import AbstractLinearOperator_sn


class BatchedLinearOperator(AbstractLinearOperator_sn):
    r'''
    A linear operator that applies a given linear operator in a batched manner over the leading dimensions.
    This is a class that simply wraps a base linear operator to get correct in/out shapes for batched applications.
    This does currently *not* implement the batching logic; this is assumed to be handled by the base operator itself.
    

    The base operator should have input shape that matches the trailing dimensions of the desired input shape.

    Furthermore, it is assumed that the base operator can handle arbitrary batching. This means that the base operator's `mv` method should be able to process inputs with additional leading dimensions.

    This class simply exists to be able to compose linear operators of tensor-product like structure:

    E.g., if we have a moment operator that maps as:

    .. math::
        M[psi_{k}] = \sum_{k} w_{k} Y_{l}^{m}(Omega_{k}) psi_{k}

    We also want to map this to a psi_k which is defined on a spatial grid, i.e., psi_{i,j,k}.

    .. math::
        M[psi_{i,j,k}] = \sum_{k} w_{k} Y_{l}^{m}(Omega_{k}) psi_{i,j,k}
    
    This class then allows us to compose it as follows:

    .. code-block:: python
        T        = TransportOperator(...)
        xin      = T.in_structure()  
        M_batched = BatchedLinearOperator(M, desired_in_shape=T.out_structure())

        MT = M_batched @ T

        MT.mv(xin)  # applies M in a batched manner over the leading dimensions of xin        

    '''
    base_operator : lx.AbstractLinearOperator
    in_shape         : jax.ShapeDtypeStruct
    out_shape        : jax.ShapeDtypeStruct

    def __init__(self, base_operator : lx.AbstractLinearOperator, desired_in_shape : jax.ShapeDtypeStruct):
        self.base_operator = base_operator
        assert base_operator.in_structure().dtype == desired_in_shape.dtype, f"Dtype of desired_in_shape {desired_in_shape.dtype} must match base operator dtype {base_operator.in_structure().dtype}."
        len_shape_diff = len(desired_in_shape.shape) - len(base_operator.in_structure().shape)
        assert base_operator.in_structure().shape == desired_in_shape.shape[len_shape_diff:], f"Trailing dimensions of desired_in_shape {desired_in_shape.shape[len_shape_diff:]} must match base operator in_structure shape {base_operator.in_structure().shape}."                  
        self.in_shape = desired_in_shape
        self.out_shape = jax.ShapeDtypeStruct(shape=desired_in_shape.shape[:len_shape_diff] + base_operator.out_structure().shape, dtype=base_operator.out_structure().dtype)
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.base_operator.mv(x)
    def as_matrix(self) -> jnp.ndarray:
        raise NotImplementedError("BatchedLinearOperator does not support as_matrix.")  
    def transpose(self):
        return BatchedLinearOperator(self.base_operator.transpose(), self.out_shape)#, self.in_shape)
    def in_structure(self):
        return self.in_shape
    def out_structure(self):
        return self.out_shape
    
@lx.is_symmetric.register(BatchedLinearOperator)
def _(operator : BatchedLinearOperator):
    return lx.is_symmetric(operator.base_operator)

@lx.linearise.register(BatchedLinearOperator)
def _(operator : BatchedLinearOperator):
    return operator


