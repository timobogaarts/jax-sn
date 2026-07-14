from abc import abstractmethod
from typing import NamedTuple

import lineax as lx 
import equinox as eqx

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
        In jax-sn, as_matrix is not implemented for linear operators due to potentially extremely large size. Prefer throwing to silently materializing.
        '''
        raise NotImplementedError("In jax-sn, as_matrix is not implemented for linear operators due to potentially extremely large size. Prefer throwing to silently materializing.")
    
    def conj(self):
        '''
        All operators in jax-sn are real, so the conjugate is the operator itself.
        '''
        return self  #real opera



@lx.is_symmetric.register(AbstractLinearOperator_sn)
def _(operator):
    return False


@lx.is_positive_semidefinite.register(AbstractLinearOperator_sn)
def _(operator):
    return False


@lx.is_negative_semidefinite.register(AbstractLinearOperator_sn)
def _(operator):
    return False


@lx.linearise.register(AbstractLinearOperator_sn)
def _(operator):
    '''All jax-sn operators are already linear, so linearise is the identity. Required on the base:
    lineax's default linearise raises, and the JVP path of ``lx.linear_solve`` recurses into
    ``linearise`` on the primal operator, so autodiff through a solve needs this registered.'''
    return operator

class Strategy(eqx.Module):

    @abstractmethod
    def forward(self, data, x):
        ...

    @abstractmethod
    def transpose(self, data, x):
        ...
    
class SymmetricStrategy(Strategy):
    '''A strategy which is symmetric, i.e. forward and transpose are the same.'''
    def transpose(self, data, x):
        return self.forward(data, x)


class DataStrategyLinearOperator(AbstractLinearOperator_sn):
    r'''
    A linear operator which is defined by a data object and a strategy object.
    The data object contains the information needed to construct the operator, and the strategy object defines how to apply the operator.
    '''
    data           :  eqx.AbstractVar[NamedTuple]    
    strategy       :  eqx.AbstractVar[Strategy]
    
    @abstractmethod
    def mv(self, x):
        ...

class SymmetricDataStrategyLinearOperator(DataStrategyLinearOperator):        

    @eqx.filter_jit
    def mv(self, x):
        return self.strategy.forward(self.data, x)
        
    def transpose(self):
        return self
    
    @abstractmethod
    def _structure(self):
        ...

    def in_structure(self):
        return self._structure()

    def out_structure(self):
        return self._structure()
    
class TransposeBoolDataStrategyLinearOperator(DataStrategyLinearOperator):
    
    transpose_bool :  eqx.AbstractVar[bool]

    @eqx.filter_jit
    def mv(self, x):
        if self.transpose_bool:
            return self.strategy.transpose(self.data, x)
        else:
            return self.strategy.forward(self.data, x)

    def transpose(self):
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)
        
    @abstractmethod
    def _in_structure(self):
        ...

    @abstractmethod
    def _out_structure(self):
        ...

    def in_structure(self):
        return self._out_structure() if self.transpose_bool else self._in_structure()

    def out_structure(self):
        return self._in_structure() if self.transpose_bool else self._out_structure()