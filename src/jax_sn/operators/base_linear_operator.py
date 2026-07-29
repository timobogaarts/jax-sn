'''
This module defines specialized base classes for linear operators in jax-sn, using the lineax operators as a base.

The strategy + data is used here. Data is a pytree that contains all data the strategy needs to apply
the operator.

This cleanly separates choice (i.e. batch size) from the rest of the data. All operators are then of
similar nature, with some specialized classes for common patterns (e.g. transpose boolean vs data
transpose).

Why not inheriting and implementing a forward?
----------------------------------------------
This works well enough for single operators, but once it starts composing, it gets messy: if you want
to have implementations of a different kind, you need to have all the data for such an operator in the
structure, leading to duplicated data (not one source of truth).

Consider:

.. code-block:: python

    class Bop:
        fem_data : FEMData

        @abstractmethod
        def mv(self, x):
            ...

    class Cop:
        fem_data : FEMData

        @abstractmethod
        def mv(self, x):
            ...

    class Aop:
        bop : Bop
        cop : Cop

        def mv(self, x):
            return self.bop.mv(self.cop.mv(x))

The data for Bop and Cop is now duplicated twice! Instead we have:

.. code-block:: python

    class Bop:
        fem_data : FEMData
        strategy : Bstrategy

        def mv(self, x):
            return self.strategy.forward(self.fem_data, x)

    class Cop:
        fem_data : FEMData
        strategy : Cstrategy

        def mv(self, x):
            return self.strategy.forward(self.fem_data, x)

    class ComposedA(Astrategy):
        b_strategy : Bstrategy
        c_strategy : Cstrategy

        def forward(self, fem_data, x):
            return self.b_strategy.forward(fem_data, self.c_strategy.forward(fem_data, x))

    class Aop:
        fem_data   : FEMData
        a_strategy : Astrategy

        def mv(self, x):
            return self.a_strategy.forward(self.fem_data, x)

Construction is free (it's just composing, no computation), and the data is only stored once. The
strategies are stateless, so they can be shared between operators. This further gives us the
flexibility to have fused strategies, which make no reference to their suboperators (i.e. a
multi-angle fused sweep!)
'''

from abc import abstractmethod
from typing import Any
from typing_extensions import Self

import jax
import lineax as lx
import equinox as eqx

# Typing same as base lineax: an operator maps its in_structure ("_b") to its out_structure ("_a").
from jaxtyping import (
    Array,
    Inexact,
    PyTree,  # pyright: ignore
)

class AbstractLinearOperator_sn(lx.AbstractLinearOperator):
    r'''
    Base class for linear operators in jax-sn.
    Extends lineax AbstractLinearOperator to provide a common base class for all jax-sn linear operators.

    Some specific functionality for jax-sn linear operators can be added here in the future.

    Since the operators can have very large sizes, we prefer to throw by default when as_matrix is called.
    Since we only use real operators in jax-sn, the conj method is implemented to return self.    

    Abstract methods are forwarded from lineax.AbstractLinearOperator, and must be implemented by subclasses.
    '''
    
    def __repr__(self) -> str:
        return super().__repr__() #+' maps '+ str(self.in_structure().shape) + ' -> ' + str(self.out_structure().shape)

    def as_matrix(self) -> Inexact[Array, "a b"]:
        '''
        In jax-sn, as_matrix is not implemented for linear operators due to potentially extremely large size. Prefer throwing to silently materializing.
        '''
        raise NotImplementedError("In jax-sn, as_matrix is not implemented for linear operators due to potentially extremely large size. Prefer throwing to silently materializing.")

    def conj(self) -> Self:
        '''
        All operators in jax-sn are real, so the conjugate is the operator itself.
        '''
        return self

    @abstractmethod
    def mv(self, x : PyTree[Inexact[Array, " _b"]]) -> PyTree[Inexact[Array, " _a"]]:
        '''Apply the operator to a vector.'''
        ...

    @abstractmethod
    def transpose(self) -> "AbstractLinearOperator_sn":
        '''Return the transposed operator.'''
        ...

    @abstractmethod
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        '''Return the structure of the input vector.'''
        ...

    @abstractmethod
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        '''Return the structure of the output vector.'''
        ...

# Registering them. They don't do much

@lx.is_symmetric.register(AbstractLinearOperator_sn)
def _(operator : AbstractLinearOperator_sn) -> bool:
    return False


@lx.is_positive_semidefinite.register(AbstractLinearOperator_sn)
def _(operator : AbstractLinearOperator_sn) -> bool:
    return False


@lx.is_negative_semidefinite.register(AbstractLinearOperator_sn)
def _(operator : AbstractLinearOperator_sn) -> bool:
    return False

@lx.linearise.register(AbstractLinearOperator_sn)
def _(operator : AbstractLinearOperator_sn) -> AbstractLinearOperator_sn:
    return operator

class DirectlyInvertibleOperator(AbstractLinearOperator_sn):
    '''
    An operator that can be directly inverted. Uses lineax to define the JVP and VJP.
    '''

    @abstractmethod
    def _invert_internal(self, b : PyTree[Inexact[Array, " _a"]]) -> PyTree[Inexact[Array, " _b"]]:
        '''Apply the operator's own inverse to b. By definition always succeeds.'''
        ...

    def inv(self, b : PyTree[Inexact[Array, " _a"]]) -> PyTree[Inexact[Array, " _b"]]:
        r'''Solve :math:`\mathbf{A} x = b` for `x` using the operator's own direct inverse.'''
        return lx.linear_solve(self, b, DirectInverseSolver(), throw=False).value


class DirectInverseSolver(lx.AbstractLinearSolver["DirectlyInvertibleOperator"]):
    '''Complete dummy class for always-correct linear solves.'''

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]) -> DirectlyInvertibleOperator:
        '''Does nothing, but checks that the operator is a DirectlyInvertibleOperator.'''
        assert isinstance(operator, DirectlyInvertibleOperator), \
            f"DirectInverseSolver requires a DirectlyInvertibleOperator, but got {type(operator).__name__}."
        return operator

    def compute(
        self,
        state: DirectlyInvertibleOperator,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], lx.RESULTS, dict[str, Any]]:
        '''Always a succesful solve using the internal inversion'''
        return state._invert_internal(vector), lx.RESULTS.successful, {}

    def assume_full_rank(self) -> bool:
        '''Always full rank'''
        return True

    def transpose(
        self, state: DirectlyInvertibleOperator, options: dict[str, Any]
    ) -> tuple[DirectlyInvertibleOperator, dict[str, Any]]:
        '''Transpose the solver state or equally the operator'''
        return state.transpose(), options

    def conj(
        self, state: DirectlyInvertibleOperator, options: dict[str, Any]
    ) -> tuple[DirectlyInvertibleOperator, dict[str, Any]]:
        '''Conj defers to the operator.'''
        return state.conj(), options


class BaseOperatorData(eqx.Module):
    '''
    Base data class. For type checking only (all can just be eqx.Module themselves).
    '''


class TransposableOperatorData(BaseOperatorData):
    '''Data that carries the direction of its operator, and so knows its own transpose (e.g. the
    transport data, whose transpose negates the angle and reverses the sweep ordering together).'''

    @property
    @abstractmethod
    def transposed(self) -> Self:
        '''The data of the transposed operator.'''
        ...


class Strategy(eqx.Module):
    '''
    Applies a linear operator to a vector. Not transposable.
    '''

    @abstractmethod
    def forward(self, data : BaseOperatorData, x : PyTree[Inexact[Array, " _b"]]) -> PyTree[Inexact[Array, " _a"]]:
        ...


class TransposableStrategy(Strategy):
    '''A strategy that implements the transposed action itself, for operators that carry the direction
    as a flag rather than in their data.'''

    @abstractmethod
    def transpose(self, data : BaseOperatorData, x : PyTree[Inexact[Array, " _a"]]) -> PyTree[Inexact[Array, " _b"]]:
        ...

class SymmetricStrategy(TransposableStrategy):
    '''A strategy which is symmetric, i.e. forward and transpose are the same.'''
    def transpose(self, data : BaseOperatorData, x : PyTree[Inexact[Array, " _a"]]) -> PyTree[Inexact[Array, " _b"]]:
        return self.forward(data, x)


class DataStrategyLinearOperator(AbstractLinearOperator_sn):
    r'''
    A linear operator which is defined by a data object and a strategy object.
    The data object contains the information needed to construct the operator, and the strategy object defines how to apply the operator.
    '''
    data           :  eqx.AbstractVar[BaseOperatorData]
    strategy       :  eqx.AbstractVar[Strategy]

    @abstractmethod
    def mv(self, x : PyTree[Inexact[Array, " _b"]]) -> PyTree[Inexact[Array, " _a"]]:
        ...


class ForwardDataStrategyLinearOperator(DataStrategyLinearOperator):
    '''``mv`` is always the strategy's forward action, and in/out structures coincide. Subclasses
    supply the shared structure via ``_structure``, and say how transposition works.'''

    @eqx.filter_jit
    def mv(self, x : PyTree[Inexact[Array, " _b"]]) -> PyTree[Inexact[Array, " _a"]]:
        return self.strategy.forward(self.data, x)

    @abstractmethod
    def _structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        ...

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._structure()


class SymmetricDataStrategyLinearOperator(ForwardDataStrategyLinearOperator):
    '''Transposition is a no-op: the strategy's forward action is assumed to be the symmetric one.'''

    def transpose(self) -> Self:
        return self


class DataTransposeDataStrategyLinearOperator(ForwardDataStrategyLinearOperator):
    '''Transposition is a transform of the data.'''

    data           :  eqx.AbstractVar[TransposableOperatorData]

    def transpose(self) -> Self:
        return eqx.tree_at(lambda op: op.data, self, self.data.transposed)


class TransposeBoolDataStrategyLinearOperator(DataStrategyLinearOperator):
    '''Transposition is a flag: the data is direction-agnostic, so the strategy implements both
    directions and ``mv`` picks one. In/out structures swap with it.'''

    strategy       :  eqx.AbstractVar[TransposableStrategy]
    transpose_bool :  eqx.AbstractVar[bool]

    @eqx.filter_jit
    def mv(self, x : PyTree[Inexact[Array, " _b"]]) -> PyTree[Inexact[Array, " _a"]]:
        if self.transpose_bool:
            return self.strategy.transpose(self.data, x)
        else:
            return self.strategy.forward(self.data, x)

    def transpose(self) -> Self:
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)

    @abstractmethod
    def _in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        ...

    @abstractmethod
    def _out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        ...

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure() if self.transpose_bool else self._in_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure() if self.transpose_bool else self._out_structure()