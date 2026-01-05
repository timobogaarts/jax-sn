import jax
import jax.numpy as jnp
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.moment_operator import MomentOperator
from jax_sn.operators.mass_operator import MassOperator
from jax_sn.operators.scatter_operator import InScatterOperator
from jax_sn.operators.transport_operator import TransportOperatorMultipleAngles
import lineax as lx 

import equinox as eqx

class SingleGroupOperator(AbstractLinearOperator_sn):
    r'''
    Single energy group linear operator implementation base class.

    This implements the total, single-group linear operator:

    .. math::
        \mathbf{A}_g = \mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1} \mathbf{H} \mathbf{S}

    where :math:`\mathbf{T}` is the (multi-angle) transport operator, \mathbf{H} the batched (over angles) mass operator, :math:`\mathbf{M}` the batched (over element and basis function)
    moment operator,  :math:`\mathbf{S}` the batched (over element and basis) scattering operator.

    Mathematically, this operator maps:

    .. math::
        \mathbf{A}_g : \mathcal{M} \otimes F \rightarrow \mathcal{M} \otimes F

    '''

    moment_operator    : MomentOperator
    transport_operator : TransportOperatorMultipleAngles
    mass_operator      : MassOperator
    scatter_operator   : InScatterOperator

    def __check_init__(self):
        assert self.transport_operator.energy_group == self.scatter_operator.energy_group, f"Energy group mismatch between operators: transport: {self.transport_operator.energy_group}, scatter: {self.scatter_operator.energy_group}"
        assert isinstance(self.transport_operator, TransportOperatorMultipleAngles), f"transport_operator must be of type TransportOperatorMultipleAngles but is {type(self.transport_operator)}"
        assert isinstance(self.moment_operator, MomentOperator), f"moment_operator must be of type MomentOperator but is {type(self.moment_operator)}"
        assert isinstance(self.mass_operator, MassOperator), f"mass_operator must be of type MassOperator but is {type(self.mass_operator)}"
        assert isinstance(self.scatter_operator, InScatterOperator), f"scatter_operator must be of type InScatterOperator but is {type(self.scatter_operator)}"
        assert self.mass_operator.out_structure() == self.transport_operator.out_structure(), f"mass_operator out_structure {self.mass_operator.out_structure()} must match transport_operator [note it is inverted] out_structure {self.transport_operator.out_structure()}"
        assert self.scatter_operator.in_structure() == self.moment_operator.out_structure(), f"scatter_operator in_structure {self.scatter_operator.in_structure()} must match moment_operator out_structure {self.moment_operator.out_structure()}"
        assert self.transport_operator.in_structure() == self.moment_operator.in_structure(), f"transport_operator in_structure [note it is inverted] {self.transport_operator.in_structure()} must match moment_operator in_structure {self.moment_operator.in_structure()}"
        # TODO add more checks? shapes, etc.
         

    def mv(self, x):
        return _mv_single_group_operator(self.moment_operator,
                                        self.transport_operator,
                                        self.mass_operator,
                                        self.scatter_operator,
                                        x)
    
    def transpose(self):
        return SingleGroupOperatorTranspose(self.moment_operator,
                                            self.transport_operator,
                                            self.mass_operator,
                                            self.scatter_operator)  
    
    def conj(self):
        return self  # real operator

    def as_matrix(self):
        assert False, "Not desired for SingleGroupOperatorImp due to large size"        
        return jnp.array([[]])
    
    def in_structure(self):
        return jax.ShapeDtypeStruct((self.moment_operator.wk_yklm.shape[-1],) + self.transport_operator.out_structure().shape[1:] , self.moment_operator.wk_yklm.dtype)
    
    def out_structure(self):
        return jax.ShapeDtypeStruct((self.moment_operator.wk_yklm.shape[-1],) + self.transport_operator.out_structure().shape[1:] , self.moment_operator.wk_yklm.dtype)
    
    @property 
    def energy_group(self):
        return self.transport_operator.energy_group
    

    

@lx.is_symmetric.register(SingleGroupOperator)
def _(operator : SingleGroupOperator):
    return False

@lx.linearise.register(SingleGroupOperator)
def _(operator : SingleGroupOperator):
    return operator    
        
    

@eqx.filter_jit
def _mv_single_group_operator(moment_operator, transport_operator, mass_operator, scatter_operator, x):
    r'''
    Multiplies the single group operator with a vector x.

    Parameters
    ----------
    moment_operator : MomentOperator
        The moment operator :math:`\mathbf{M}`.
    transport_operator : TransportOperatorMultipleAngles
        The transport operator :math:`\mathbf{T}^{-1}`.
    mass_operator : MassOperator
        The mass operator :math:`\mathbf{H}`.
    scatter_operator : InScatterOperator
        The scattering operator :math:`\mathbf{S}`.
    x : jnp.ndarray
        The input vector.

    Returns
    -------
    jnp.ndarray
        The result of the multiplication.
    '''        
    return x - moment_operator.mv(transport_operator.inv(mass_operator.mv(scatter_operator.mv(x)))) 


@eqx.filter_jit
def _mv_single_group_operator_transpose(moment_operator, transport_operator, mass_operator, scatter_operator, x):
    r'''
    Multiplies the transpose of the single group operator with a vector x.

    Since the transposes are trivial (we only point to a different function), this does not add any overhead.

    Parameters
    ----------
    moment_operator : MomentOperator
        The moment operator :math:`\mathbf{M}`.
    transport_operator : TransportOperatorMultipleAngles
        The transport operator :math:`\mathbf{T}^{-1}`.
    mass_operator : MassOperator
        The mass operator :math:`\mathbf{H}`.
    scatter_operator : InScatterOperator
        The scattering operator :math:`\mathbf{S}`.
    x : jnp.ndarray
        The input vector.

    Returns
    -------
    jnp.ndarray
        The result of the multiplication.
    '''        
    return x - scatter_operator.transpose().mv(mass_operator.transpose().mv(transport_operator.transpose().inv(moment_operator.transpose().mv(x))))

class SingleGroupOperatorTranspose(SingleGroupOperator):
    r'''
    Transpose of the SingleGroupOperator. See SingleGroupOperator for details.
    '''
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return _mv_single_group_operator_transpose(self.moment_operator,
                                                   self.transport_operator,
                                                   self.mass_operator,
                                                   self.scatter_operator,
                                                   x)
    
    def transpose(self) -> "SingleGroupOperator":
        return SingleGroupOperator(self.moment_operator,
                                   self.transport_operator,
                                   self.mass_operator,
                                   self.scatter_operator)
    
    def in_structure(self):
        return SingleGroupOperator.out_structure(self)
            
    def out_structure(self):
        return SingleGroupOperator.in_structure(self)    

@lx.is_symmetric.register(SingleGroupOperatorTranspose)
def _(operator : SingleGroupOperatorTranspose):
    return False

@lx.linearise.register(SingleGroupOperatorTranspose)
def _(operator : SingleGroupOperatorTranspose):
    return operator    
        