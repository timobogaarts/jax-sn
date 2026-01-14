import lineax as lx 


class AbstractLinearOperator_sn(lx.AbstractLinearOperator):
    r'''
    Base class for linear operators in jax-sn.
    Extends lineax AbstractLinearOperator to provide a common base class for all jax-sn linear operators.

    Some specific functionality for jax-sn linear operators can be added here in the future.

    Since the operators can have very large sizes, we prefer to throw by default when as_matrix is called.
    Since we only use real operators in jax-sn, the conj method is implemented to return self.    
    '''
    
    def __repr__(self):        
        return super().__repr__() #+' maps '+ str(self.in_structure().shape) + ' -> ' + str(self.out_structure().shape)
    
    def as_matrix(self):    
        '''
        In jax-snj, as_matrix is not implemented for linear operators due to potentially extremely large size. Prefer throwing to silently materializing.
        '''
        assert False, "In jax-sn, as_matrix is not implemented for linear operators due to potentially extremely large size. Prefer throwing to silently materializing."
        return jnp.array([[]])
    
    def conj(self):
        '''
        All operators in jax-sn are real, so the conjugate is the operator itself.
        '''
        return self  #real opera
