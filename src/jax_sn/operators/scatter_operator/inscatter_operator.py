import jax
import jax.numpy as jnp

from .scatter_operator_util import _scatter_kernel_xs_indexed_discrete, _scatter_kernel_xs_indexed_discrete_transposed
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.domain.cross_sections import CrossSectionIndexed

from jax_sn.quadrature_set import QuadratureSet

import lineax as lx
from .scatter_operator_util import total_scattering_kernel_xs_indexed_discrete, total_scattering_kernel_xs_indexed_discrete_transpose


class InScatterOperator(AbstractLinearOperator_sn):
    r'''
    InScatterOperator represents the in-scattering operator that maps angular flux moments to discrete angles using indexed cross sections for a specific energy group

    Implements (in this case :math:`g_{out}` is equal to :math:`g_{in} = g`):

    .. math::        
        s_{kei} = \sum_{lm} Y_{klm} \sigma_{l, g\leftarrow g}^{e} \psi_{lm,ei}^{ g} 
    where :math:`\sigma_{l, g\leftarrow g}^{e}` are the indexed scattering cross sections for element :math:`e`, 
    from energy group :math:`g` to :math:`g`, :math:`Y_{klm}` are the spherical harmonics evaluated at angle :math:`k`, and :math:`\psi_{lm, i}^{ g}` 
    are the angular flux moments for moment :math:`(l,m)` and basis function index :math:`i` in energy group :math:`g`.

    Note that it acts trivially on the basis function in each element.

    Mathematical description
    ---------------------------

    Mathematically, this operator maps as 

    .. math::
        \mathbf{S}_{gg} = \overline{\mathbf{S}}_{gg}  \otimes \mathbf{I}_{\mathcal{B}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{D} \otimes \mathcal{F}

    where :math:`\mathcal{F} = \mathcal{E}\otimes \mathcal{B}` is the space of elements and basis functions, and :math:`\mathbf{S}_b` is the single-element in-scattering operator defined as:

    .. math::
        \overline{\mathbf{S}}_{gg} [\psi_{g\ell e}] =
        s_{ke} = \sum_{\ell} Y_{k\ell} \sigma_{\ell gge} \psi_{g\ell e}

    (we use :math:`\ell` to denote :math:`(l,m)` moment index for brevity; for :math:`\sigma_{\ell}` this just equals :math:`\sigma_{l}` for all :math:`m`).

    Transposition
    ---------------------------

    Tranpose is thus:

    .. math::
        \mathbf{S}_{gg}^T = \overline{\mathbf{S}}_{gg}^T \otimes \mathbf{I}_{\mathcal{B}} : \ \  \mathcal{D} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}

    with

    .. math::
        \overline{\mathbf{S}}_{gg}^T [s_{ke}] = \psi_{g\ell e} = \sum_{k} Y_{k\ell} \sigma_{\ell gge} s_{ke}        

    Structure
    ---------------------------

    Note that this operator basically has a structure of two operators: one that simply scales, let's say :math:`\mathbf{\mathcal{S}}`:

    .. math::
        &\mathbf{\mathcal{S}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}\\
        &\mathbf{\mathcal{S}} = \overline{\mathcal{S}} \otimes \mathbf{I}_{\mathcal{B}}

    with

    .. math::
        \overline{\mathbf{\mathcal{S}}}[\psi_{g\ell e}] = \sigma_{\ell gg} \psi_{g\ell e}

    and a moment-to-discrete operator :math:`\mathbf{D}`:

    .. math::
        &\mathbf{D} : \mathcal{M} \otimes \mathcal{F} \to \mathcal{D} \otimes \mathcal{F}\\
        &\mathbf{D} = \overline{\mathbf{D}} \otimes \mathbf{I}_{\mathcal{F}}

    with

    .. math::
        \overline{\mathbf{D}}[\psi_{\ell i}] = s_{k} = \sum_{\ell} Y_{k\ell} \psi_{\ell i}

    They are combined in one operator here however.

    '''


    yklm                  : jnp.ndarray           # [n_angles, n_moments]
    cross_sections_indexed : CrossSectionIndexed
    energy_group          : int
    elements_basis_shape  : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    
    @classmethod 
    def from_quadrature_set_and_cross_sections(cls, quadrature_set : QuadratureSet, cross_sections : CrossSectionIndexed, energy_group : int, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct) -> "InScatterOperator":
        '''
        Constructs an InScatterOperator from a quadrature set and cross sections.
        Note this is merely a convenience function: the yklm matrix is explicitly calculated here.
        The default constructor can be used if the yklm matrix is already available.

        Parameters
        ----------
        quadrature_set : QuadratureSet
            The quadrature set to use.
        cross_sections : CrossSectionIndexed
            The cross sections to use.
        energy_group : int
            The energy group to use.
        l_max : int [0, ....]
            The maximum moment to use.  

        Returns
        -------
        InScatterOperator
            The constructed InScatterOperator.
        
        '''
        yklm = quadrature_set.ylm_real(l_max)  # [n_angles, n_moments]
        return cls(yklm, cross_sections, energy_group, elements_basis_shape)
    
    def __check_init__(self):
       # assert self.energy_group >=0 and self.energy_group < self.cross_sections_indexed.n_energy_groups, f"Energy group {self.energy_group} out of bounds for cross sections with {self.cross_sections_indexed.n_energy_groups} energy groups."
        assert self.yklm.ndim    == 2, f"yklm must be 2D [n_angles, n_moments], but has shape {self.yklm.shape}."
        assert self.elements_basis_shape.ndim == 2, f"elements_basis_shape must be 2D [n_elements, n_basis], but has shape {self.elements_basis_shape.shape}."
        assert self.elements_basis_shape.shape[0] == self.cross_sections_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.elements_basis_shape.shape[0]} must match number of elements in cross_sections_indexed {self.cross_sections_indexed.indices.shape[0]}."

    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return total_scattering_kernel_xs_indexed_discrete(
                                                    self.energy_group,  # g_out = self energy group
                                                    self.energy_group,  # g_in  = self energy group
                                                    self.cross_sections_indexed,
                                                    x,
                                                    self.yklm)
    def transpose(self) -> "InScatterOperatorTransposed":
        return InScatterOperatorTransposed(self.yklm, self.cross_sections_indexed, self.energy_group, self.elements_basis_shape)
    
    def to_energy_group(self, energy_group : int) -> "InScatterOperator":
        return InScatterOperator(self.yklm, self.cross_sections_indexed, energy_group, self.elements_basis_shape)
    
    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[1],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)

@lx.is_symmetric.register(InScatterOperator)
def _(operator):
    return False
    
@lx.linearise.register(InScatterOperator)
def _(operator):
    return operator

class InScatterOperatorTransposed(InScatterOperator):
    r'''
    Transpose of the InScatterOperator. See InScatterOperator for details.
    '''
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return total_scattering_kernel_xs_indexed_discrete_transpose(
                                                    self.energy_group,  # g_out = self energy group
                                                    self.energy_group,  # g_in  = self energy group
                                                    self.cross_sections_indexed,
                                                    x,
                                                    self.yklm)
    def transpose(self) -> "InScatterOperator":
        return InScatterOperator(self.yklm, self.cross_sections_indexed, self.energy_group, self.elements_basis_shape)
    
    def in_structure(self):
        return InScatterOperator.out_structure(self)
            
    def out_structure(self):
        return InScatterOperator.in_structure(self)
            

@lx.is_symmetric.register(InScatterOperatorTransposed)
def _(operator):
    return False
    
@lx.linearise.register(InScatterOperatorTransposed)
def _(operator):
    return operator

