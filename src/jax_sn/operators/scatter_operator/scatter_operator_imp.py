import jax
import jax.numpy as jnp

from .scatter_operator_util import _scatter_kernel_xs_indexed_discrete, _scatter_kernel_xs_indexed_discrete_transposed
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.moment_operator import DiscreteOperator, DiscreteOperatorTransposed
from jax_sn.domain.cross_sections import CrossSectionIndexed

from jax_sn.quadrature_set import QuadratureSet

import lineax as lx
from .scatter_operator_util import total_scatter_kernel_xs_indexed

class ScatterOperator(AbstractLinearOperator_sn):
    r'''
    Maps the flux moments to scattering for a specific energy group in and a specific energy group out, using indexed cross sections.

    Mathematical description
    ---------------------------
    We have:

    .. math::
        \mathbf{S}_{\tilde{g} g} = \overline{\mathbf{S}}_{\tilde{g} g}  \otimes \mathbf{I}_{\mathcal{B}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}

    with :math:`\tilde{g}` the outgoing energy group, :math:`g` the incoming energy group, :math:`\mathcal{F} = \mathcal{E}\otimes \mathcal{B}` the space of elements and basis functions, and :math:`\mathbf{S}_{\tilde{g} g}` the single-element scattering operator defined as:

    .. math:: 
        \overline{\mathbf{S}}_{\tilde{g} g} [\psi] 
         =\sigma_{\ell \tilde{g} g e} \psi_{g\ell e}    

    for an arbitrary :math:`\psi\in \mathcal{M} \otimes \mathcal{E}` (we use :math:`\ell` to denote :math:`(l,m)` moment index for brevity; for :math:`\sigma_{\ell}` this just equals :math:`\sigma_{l}` for all :math:`m`).

    Transposition
    ---------------------------
    Since this is a element wise multiplication, the transpose is itself.

    Technical details
    ---------------------------
    We use a :class:`jax.ShapeDtypeStruct` to represent the shape of the moments, elements and basis functions over which this operator is defined. This is used to define a consistent in and output structure.    
    Since we want to have the flexibility to have a different number of moments than is present in the cross sections, we explicitly provide it here.

    '''
    
    cross_sections_indexed : CrossSectionIndexed
    r'''The indexed cross sections to use.'''
    moments_elements_basis_shape   : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''
    energy_group_out       : int
    r'''The outgoing energy group index.'''
    energy_group_in        : int
    r'''The incoming energy group index.'''
    

    def __check_init__(self):
        assert self.moments_elements_basis_shape.ndim == 3, f"elements_basis_shape must be 2D [moments_n_elements, n_basis], but has shape {self.moments_elements_basis_shape.shape}."
        assert self.moments_elements_basis_shape.shape[1] == self.cross_sections_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.moments_elements_basis_shape.shape[0]} must match number of elements in cross_sections_indexed {self.cross_sections_indexed.indices.shape[0]}."


    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return total_scatter_kernel_xs_indexed(     self.energy_group_out,
                                                    self.energy_group_in,
                                                    self.cross_sections_indexed,
                                                    x)

    def transpose(self) -> "ScatterOperator":
        return self # This is a simple scaling operator, so its transpose is itself
    
    def to_energy_group(self, energy_group_out : int, energy_group_in : int) -> "ScatterOperator":
        return ScatterOperator(self.cross_sections_indexed,
                               self.moments_elements_basis_shape,
                               energy_group_in,
                               energy_group_out)
    
    def in_structure(self):
        return self.moments_elements_basis_shape
    def out_structure(self):
        return self.moments_elements_basis_shape
    



@lx.is_symmetric.register(ScatterOperator)
def _(operator):
    return False
    
@lx.linearise.register(ScatterOperator)
def _(operator):
    return operator