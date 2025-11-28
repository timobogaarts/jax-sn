import basix
import jax.numpy as jnp
import numpy as onp
from abc import ABC, abstractmethod
from typing import Tuple
class Element(ABC):    
    
    @property
    @abstractmethod        
    def volume_quadrature(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Returns the quadrature points and weights for the element.
        '''
        ...
    @property
    @abstractmethod    
    def has_linear_jacobian(self) -> bool:
        '''
        Returns whether the element has a linear Jacobian
        '''
        ...

    @property
    @abstractmethod        
    def face_quadratures(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Returns the quadrature points and weights for the faces of the element.
        '''
        ...
    
    @abstractmethod    
    def tabulate_basis_functions_first_derivative(self, points : jnp.ndarray) -> jnp.ndarray:
        '''
        Returns the basis functions and their first derivatives at the given points.

        Parameters
        ----------
        points : jnp.ndarray [n_points, n_dim]
            The points at which to evaluate the basis functions and their derivatives.
        
        Returns
        -------
        jnp.ndarray [n_values, n_points, n_basis]
            The basis functions and their first derivatives at the given points.
            The n_values depends on the number of dimensions of the element.
            For example, for a triangle in 2D, n_derivatives = 3 (1 for the function value and 2 for the derivatives in x and y).

        '''
        ...

    @property
    @abstractmethod    
    def n_faces(self) -> int:
        '''
        Returns the number of faces of the element.
        '''
        ...
    

def compute_volume_matrices_linear_jacobian(element : Element):
    '''
    Computes the mass matrix, integration coefficient vector, and derivative matrix for an element with a linear Jacobian.

    Parameters
    ----------
    element : Element
        The finite element for which to compute the matrices.
    Returns
    -------
    mass_matrix : jnp.ndarray
        The mass matrix of shape [n_basis, n_basis].
    integration_coeff : jnp.ndarray
        The integration coefficient vector of shape [n_basis].
    derivative_matrix : jnp.ndarray
        The derivative matrix of shape [n_basis, n_basis].
    '''
    assert element.has_linear_jacobian, "Element must have a linear Jacobian!"
    points, weights   = element.volume_quadrature
    hi_dhi            = element.tabulate_basis_functions_first_derivative(points)  # [n_values, n_quadrature_points, n_basis]  
    hi                = hi_dhi[0]       # (n_quadrature_points, n_basis)             = value at [quad_point, basis_no]
    dhi_dr            = hi_dhi[1:]  # (n_values-1, n_quadrature_points, n_basis) = d value / dx at [quad_point, basis_no]
    mass_matrix       = _compute_mass_matrix(hi, weights)
    integration_coeff = _compute_integration_coefficient(hi, weights)
    derivative_matrix = _compute_derivative_matrix(hi, dhi_dr, weights)
    return mass_matrix, integration_coeff, derivative_matrix

def _compute_mass_matrix(hi, wi):
    '''
    Integrates the mass matrix given basis function evaluations h_i at quadrature points and quadrature weights w_i.

    i.e. A_{ij} = \\int h_i h_j dV = \\sum_q h_i(q) h_j(q) w(q)
    
    Parameters
    ----------
    h_i: jnp.ndarray
        Array of shape [n_quadrature_points, n_basis] containing basis function evaluations at quadrature points.
    w_i: jnp.ndarray
        Array of shape [n_quadrature_points] containing quadrature weights.    

    Returns
    -------
    mass_matrix: jnp.ndarray
        Array of shape [n_basis, n_basis] representing the mass matrix.
    '''
    hihj = jnp.einsum("qi,qj->qij", hi, hi)  #[n_quadrature_points, n_basis, n_basis]

    mass_matrix = jnp.einsum("qij,q->ij", hihj, wi)  #[n_basis, n_basis]
    return mass_matrix

def _compute_integration_coefficient(hi, wi):
    '''
    Integrates the coefficient vector given basis function evaluations h_i at quadrature points and quadrature weights w_i.

    i.e. I_i = \\int h_i dV = \\sum_q h_i(q) w(q)
    
    Parameters
    ----------
    h_i: jnp.ndarray
        Array of shape [n_quadrature_points, n_basis] containing basis function evaluations at quadrature points.
    w_i: jnp.ndarray
        Array of shape [n_quadrature_points] containing quadrature weights.    

    Returns
    -------
    coeff_vector: jnp.ndarray
        Array of shape [n_basis] representing the integrated coefficient vector.
    '''
    coeff_vector = jnp.einsum("qi,q->i", hi, wi)  #[n_basis]
    return coeff_vector

def _compute_derivative_matrix(hi, dhi_dr, wi):
    '''
    Integrates the derivative matrix given basis function evaluations h_i at quadrature points, their derivatives dhi_dr, and quadrature weights w_i.
    Note that dhi_dr is the derivative with respect to some coordinate r (which depends on the dimensionality of the element).

    i.e. B_{alpha, ij} = \\int dH_i/dr_alpha h_j dV = \\sum_q dh_i/dr_alpha(q) h_j(q) w(q)
    
    Parameters
    ----------
    hi: jnp.ndarray
        Array of shape [n_quadrature_points, n_basis] containing basis function evaluations at quadrature points.
    dhi_dr: jnp.ndarray
        Array of shape [n_dim, n_quadrature_points, n_basis] containing basis function derivatives at quadrature points.
    w_i: jnp.ndarray
        Array of shape [n_quadrature_points] containing quadrature weights.    

    Returns
    -------
    derivative_matrix: jnp.ndarray
        Array of shape [ndim, n_basis, n_basis] representing the derivative matrix.
    '''
    hidhj = - jnp.einsum("aqi,qj->qija", dhi_dr, hi)  #[n_quadrature_points, n_basis, n_basis]
    derivative_matrix = jnp.einsum("qija,q->aij", hidhj, wi)  #[n_basis, n_basis]
    return derivative_matrix

