import jax
import jax.numpy as jnp
from jax_sn.operators.moment_operator.moment_operator_imp import MomentOperator
from jax_sn.quadrature_set import QuadratureSet
from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
from jax_sn.domain.cross_sections import CrossSectionIndexed
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import lineax as lx
import equinox as eqx

from .scatter_operator_util import total_scattering_kernel_xs_indexed_multigroup_discrete, _total_scattering_kernel_xs_indexed_multigroup_backwards

class TotalGroupScatterOperator(eqx.Module):
    r'''
    Class grouping functions for scatter operators useful for multigroup total operators.

    These functions share the same data and general structure, so that is why we are grouping them. Furthermore, the functions could be specialized for specific scattering structures (e.g. downscatter only)

        
    Moments-to-discrete
    ---------------------------    
    Implements:     
    
    .. math::
        \mathbf{D}[\chi] & = \sum_{\ell} Y_{k\ell} \chi_{\ell e}        

    
    Transpose:
    ============================

    The transpose is simply:

    .. math::
        \mathbf{D}^T[s] & = \sum_{k} Y_{k\ell} s_{ke}


    Forward gathering and discretizing
    ------------------------------------------------------------

    .. math::
        \mathbf{D}\mathbf{S}_{g} = \mathbf{D}\circ \overline{\mathbf{S}}_g \otimes \mathbf{I}_B : \mathcal{G}\otimes\mathcal{M} \otimes \mathcal{F} \to  \mathcal{D} \otimes \mathcal{F}

    where :math:`\overline{\mathbf{S}}_g` is the single-element gathering scattering operator defined as:
    
    .. math::
        \overline{\mathbf{S}}_g[\psi] = s_{ke} = \sum_{\overline{g}}\sum_{\ell} Y_{k \ell } \sigma_{\ell g \overline{g}e} \psi_{\overline{g}\ell e}

    and :math:`\mathbf{D}` is as defined above.

    Backward scattering
    ------------------------------------------------------------
    In the backward (transpose) pass of the MultiGroupOperator, we need to apply the following:

    .. math::
        \overline{\mathbf{S}}_g^T \circ \mathbf{D}^T :  \mathcal{D} \otimes \mathcal{F}


    Transpose:
    ============================

    The transpose is simply:

    .. math::
        \mathbf{D}^T[s] & = \sum_{k} Y_{k\ell} s_{ke}

    Technical details
    ---------------------------
    This collection of functions provides parametrised (by :math:`g`) scattering functions with a simple interface and including checking. It does not inherit from AbstractLinearOperator_sn because it does not represent a single linear operator, but rather a collection of related operators.
    '''
    
    yklm                         : jnp.ndarray           # [n_angles, n_moments]
    '''Matrix of [n_angles, n_moments] containing :math:`Y_{l,m}(\Omega_k)` values for each angle :math:`k` and moment :math:`(l,m)` (collapsed into a single index).'''

    cross_sections_indexed        : CrossSectionIndexed
    '''The indexed cross sections to use.'''
    
    groups_elements_basis_shape  : jax.ShapeDtypeStruct  #[n_groups, n_elements, n_basis]
    '''The shape of the groups, elements and basis functions over which this operator is batched. Identified with :math:`\mathcal{G}\otimes \mathcal{F}` in the documentation above. However, note that the input and output are as [n_groups, n_moments, n_elements, n_basis] and [n_angles, n_elements, n_basis] respectively.'''
    
    @classmethod 
    def from_quadrature_set_and_cross_sections(cls, quadrature_set : QuadratureSet, cross_sections : CrossSectionIndexed, l_max : int, groups_elements_basis_shape : jax.ShapeDtypeStruct) -> "TotalGroupScatterOperator":
        '''
        Constructs a TotalScatterOperator from a quadrature set and cross sections.
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
        TotalScatterOperator
            The constructed TotalScatterOperator.
        
        '''
        yklm = quadrature_set.ylm_real(l_max)  # [n_angles, n_moments]
        return cls(yklm, cross_sections, groups_elements_basis_shape)
    
    def __check_init__(self):
        
        # assert self.energy_group >=0 and self.energy_group < self.cross_sections_indexed.n_energy_groups, f"Energy group {self.energy_group} out of bounds for cross sections with {self.cross_sections_indexed.n_energy_groups} energy groups."
        # assert self.energy_group < self.groups_elements_basis_shape.shape[0], f"Energy group {self.energy_group} out of bounds for groups_elements_basis_shape with {self.groups_elements_basis_shape.shape[0]} energy groups."
        assert self.yklm.ndim    == 2, f"yklm must be 2D [n_angles, n_moments], but has shape {self.yklm.shape}."
        assert self.groups_elements_basis_shape.ndim == 3, f"groups_elements_basis_shape must be 3D [n_elements, n_basis], but has shape {self.groups_elements_basis_shape.shape}."
        assert self.groups_elements_basis_shape.shape[1] == self.cross_sections_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.groups_elements_basis_shape.shape[1]} must match number of elements in cross_sections_indexed {self.cross_sections_indexed.indices.shape[0]}."
        self.cross_sections_indexed.cross_section_data.assert_n_energy_groups(self.groups_elements_basis_shape.shape[0]), f"Number of energy groups in cross_sections_indexed {self.cross_sections_indexed.n_energy_groups} must match number of groups in groups_elements_basis_shape {self.groups_elements_basis_shape.shape[0]}."

    def DS_g_mv(self, g_out : int, x : jnp.ndarray) -> jnp.ndarray:
        '''
        Computes the total scattering operation for outgoing energy group g_out:

        .. math::
            \mathbf{DS}_{g_{out}}[\psi] = s_{ke} = \sum_{\tilde{g}}\sum_{\ell} Y_{k \ell } \sigma_{\ell g_{out} \tilde{g}e} \psi_{\tilde{g}\ell e}
        '''
        assert x.shape == self.DS_g_in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.DS_g_in_structure().shape}."
        return total_scattering_kernel_xs_indexed_multigroup_discrete(g_out, self.cross_sections_indexed, x, self.yklm)
    
    def _gin_mv(self, g_in : int, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.S_gin_in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.S_gin_in_structure().shape}."
        return total_scattering_kernel_xs_indexed_multigroup_backwards
    
    def D_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.D_in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.D_in_structure().shape}."
        return _moment_to_discrete_mv(self.yklm, x)    

    def D_transpose_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.D_out_structure().shape, f"Input x shape {x.shape} does not match expected out_structure shape {self.D_out_structure().shape}."
        return _moment_to_discrete_transpose_mv(self.yklm, x)
    

    def transpose(self):
        raise ValueError("Transpose not implemented for TotalGroupScatterOperator because it is ill-defined")            
    
    def D_in_structure(self):
        '''
        Input structure is [n_moments, n_elements, n_basis]
        '''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[1],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    
    def D_out_structure(self):
        '''
        Output structure is [n_angles, n_elements, n_basis]
        '''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    
    def DS_g_in_structure(self):
        '''
        Input structure is [n_groups, n_moments, n_elements, n_basis]
        '''
        return jax.ShapeDtypeStruct(
            shape=(self.groups_elements_basis_shape.shape[0], ) + (self.yklm.shape[1],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    def DS_g_out_structure(self):
        '''
        Output structure is [n_angles, n_elements, n_basis]
        '''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    
    def S_gin_in_structure(self):
        '''
        Input structure is [n_moments, n_elements, n_basis]
        '''
        return jax.ShapeDtypeStruct(
            shape=(self.groups_elements_basis_shape.shape[0], ) + (self.yklm.shape[1],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    
    def S_gin_out_structure(self):
        '''
        Output structure is [n_moments, n_elements, n_basis]
        '''
        return self.S_gin_in_structure()
                
    def to_inscatter(self, energy_group : int):
        '''
        Convert to single group InScatterOperator for energy group.

        Parameters
        ----------
        energy_group : int
            Energy group index.
        Returns
        -------
        InScatterOperator
            The single group InScatterOperator for energy group.
        '''
        from jax_sn.operators.scatter_operator.inscatter_operator import InScatterOperator
        return InScatterOperator(self.yklm, self.cross_sections_indexed, energy_group, jax.ShapeDtypeStruct(self.groups_elements_basis_shape.shape[1:], dtype=self.groups_elements_basis_shape.dtype))

@lx.is_symmetric.register(TotalGroupScatterOperator)
def _(operator):
    return False
    
@lx.linearise.register(TotalGroupScatterOperator)
def _(operator):
    return operator
