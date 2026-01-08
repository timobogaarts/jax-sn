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
    InScatterOperator represents the in-scattering operator that maps angular flux moments to discrete angles using indexed cross sections.

    Implements:

    .. math::
        \mathbf{S} = \mathbf{S}_b  \otimes \mathbf{I}_{\mathcal{B}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{D} \otimes F

    where :math:`F = \mathcal{E}\otimes \mathcal{B}` is the space of elements and basis functions, and :math:`\mathbf{S}_b` is the single-element in-scattering operator defined as:

    .. math::
        \mathbf{S}_b [\psi_{lm, e}^{ g_{in}}] =
        s_{ke} = \sum_{lm} Y_{klm} \sigma_{l, g_{out}\leftarrow g_{in}}^{e} \psi_{lm,e}^{ g_{in}} =\sum_{lm} S_{klme}^{g_{out}\leftarrow g_{in}} \psi_{lm, e}^{ g_{in}}

    where :math:`\sigma_{l, g_{out}\leftarrow g_{in}}^{e}` are the indexed scattering cross sections for element :math:`e`, from energy group :math:`g_{in}` to :math:`g_{out}`, :math:`Y_{klm}` are the spherical harmonics evaluated at angle :math:`k`, and :math:`\psi_{lm, i}^{ g_{in}}` are the angular flux moments for moment :math:`(l,m)` and basis function index :math:`i` in energy group :math:`g_{in}`.
    (it acts trivially on the basis function in each element)


    Tranpose is thus:

    .. math::
        \mathbf{S}^T = \mathbf{S}_b^T \otimes \mathbf{I}_{\mathcal{B}} : \mathcal{D} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}

    .. math::
        \psi_{lm, ei}^{ g_{in}} = \sum_{k} Y_{klm} \sigma_{l, g_{out}\leftarrow g_{in}}^{e} s_{kei}


    Note that this operator basically has a structure of two operators: one that simply scales, let's say :math:`\mathbf{\mathcal{S}}`:

    .. math::
        &\mathbf{\mathcal{S}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}\\
        &\mathbf{\mathcal{S}} = \mathcal{S}_b \otimes \mathbf{I}_{\mathcal{B}}


    with
    .. math::
        \mathbf{\mathcal{S}}_b[\psi_{lm, ei}^{ g_{in}}] = \sigma_{l, g_{out}\leftarrow g_{in}}^{e} \psi_{lm, i}^{ g_{in}}

    '''


    yklm                  : jnp.ndarray           # [n_angles, n_moments]
    cross_section_indexed : CrossSectionIndexed
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
       # assert self.energy_group >=0 and self.energy_group < self.cross_section_indexed.n_energy_groups, f"Energy group {self.energy_group} out of bounds for cross sections with {self.cross_section_indexed.n_energy_groups} energy groups."
        assert self.yklm.ndim    == 2, f"yklm must be 2D [n_angles, n_moments], but has shape {self.yklm.shape}."
        assert self.elements_basis_shape.ndim == 2, f"elements_basis_shape must be 2D [n_elements, n_basis], but has shape {self.elements_basis_shape.shape}."
        assert self.elements_basis_shape.shape[0] == self.cross_section_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.elements_basis_shape.shape[0]} must match number of elements in cross_section_indexed {self.cross_section_indexed.indices.shape[0]}."

    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return total_scattering_kernel_xs_indexed_discrete(
                                                    self.energy_group,  # g_out = self energy group
                                                    self.energy_group,  # g_in  = self energy group
                                                    self.cross_section_indexed,
                                                    x,
                                                    self.yklm)
    def transpose(self) -> "InScatterOperatorTransposed":
        return InScatterOperatorTransposed(self.yklm, self.cross_section_indexed, self.energy_group, self.elements_basis_shape)
    
    def to_energy_group(self, energy_group : int) -> "InScatterOperator":
        return InScatterOperator(self.yklm, self.cross_section_indexed, energy_group, self.elements_basis_shape)
    
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
                                                    self.cross_section_indexed,
                                                    x,
                                                    self.yklm)
    def transpose(self) -> "InScatterOperator":
        return InScatterOperator(self.yklm, self.cross_section_indexed, self.energy_group, self.elements_basis_shape)
    
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

