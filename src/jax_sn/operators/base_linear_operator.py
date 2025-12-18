import lineax as lx 


class AbstractLinearOperator_sn(lx.AbstractLinearOperator):
    r'''
    Base class for linear operators in jax-sn.
    Extends lineax AbstractLinearOperator to provide a common base class for all jax-sn linear operators.

    Some specific functionality for jax-sn linear operators can be added here in the future.
    
    '''
    
    def __repr__(self):        
        return super().__repr__() +' maps '+ str(self.in_structure().shape) + ' -> ' + str(self.out_structure().shape)
    

    def batch(self, desired_input_shape):
        from .batched_linear_operator import BatchedLinearOperator
        return BatchedLinearOperator(self, desired_input_shape)