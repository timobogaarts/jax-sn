import basix
import jax.numpy as jnp
import numpy as onp
from abc import ABC, abstractmethod
from typing import Tuple, List
from jax_sn.domain.geometric_domain import GeometricDomain
import jax 
from dataclasses import dataclass


#======================================================================================================================================================================
#                                                               Precomputed Data Classes
#=======================================================================================================================================================================    

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FEMMatrices:
    '''
    Convenience class to bundle all FEM matrices required for the solution of the transport equation on a domain
    This is static wrt the domain geometry since it is only a function of the element type and basis functions.s
    '''

    mass_matrix       : jnp.ndarray  #[n_basis, n_basis]
    r'''
    Mass matrix of the reference element:

    .. math:: 
        H_{ij}=  \int_R H_i H_j dV

    The mass matrix of a non-reference element with linear Jacobian can be obtained by scaling this matrix with the determinant of the Jacobian:

    .. math::
        H_{e,ij} = |J_e| H_{ij}
    '''
    
    integration_coeff : jnp.ndarray  #[n_basis]    
    r'''
    Integration coefficient vector of the reference element:
    
    .. math:: 
        I_{i}=  \int_R H_i dV

    The integration coefficient vector of a non-reference element with linear Jacobian can be obtained by scaling this vector with the determinant of the Jacobian:

    .. math::
        I_{e,i} = |J_e| I_{i}
    '''

    derivative_matrix : jnp.ndarray  #[n_dim, n_basis, n_basis]
    r'''
    Derivative matrix of the reference element:

    .. math:: 
        B_{\alpha, ij}=  \int_R \frac{\partial H_i}{\partial r_\alpha} H_j dV

    where :math:`r_\alpha` is the reference coordinate in dimension :math:`\alpha`.

    The derivative matrix of a non-reference element with linear Jacobian can be obtained by transforming this matrix with the inverse of the Jacobian:

    .. math::
        B_{e,\beta, ij} = \sum_{\alpha}  |J_e| (\mathbf{J}_e^{-1})_{\alpha, \beta} B_{\alpha, ij}

    such that the required finite element integral in real coordinates can be computed as:

    .. math::
        \int_e H_j \Omega \cdot \nabla H_i = \sum_{\beta} \Omega_{\beta} \sum_{\alpha}  |J_e| (\mathbf{J}_e^{-1})_{\alpha, \beta} B_{\alpha, ij}
    '''

    outflow_matrices  : jnp.ndarray  #[n_faces_per_element, n_basis, n_basis]
    r'''
    Outflow face matrices of the reference element:

    .. math::
        C_{\gamma ij}=  \int_{\gamma} H_i H_j dS

    The outflow face matrices of a non-reference element with linear Jacobian can be obtained by scaling this matrix with the determinant of the face Jacobian:

    .. math::
        \Omega \cdot n_{\gamma} \int_\gamma H_i H_j dA = |J_{e,\gamma}| C_{\gamma ij} \Omega \cdot n_{\gamma}        
    '''

    inflow_matrices   : jnp.ndarray  #[n_faces_per_element, n_permutations, n_faces_per_element, n_basis, n_basis]
    r'''
    Inflow face connection matrices of the reference element:

    .. math::
        D_{\gamma_1, p, \gamma_2, ij}=  \int_{\gamma_1} H_i \overline{H}_j dS

    where :p: indicates the permutation of the face :math:`\gamma_2` to match face :math:`\gamma_1`.

    The inflow face connection matrices of a non-reference element with linear Jacobian can be obtained by scaling this matrix with the determinant of the face Jacobian:

    .. math::
        \Omega \cdot n_{\gamma_1} \int_{\gamma_1} H_i \overline{H}_j dA = |J_{e,\gamma_1}| D_{\gamma_1, p, \gamma_2, ij} \Omega \cdot n_{\gamma_1}

    '''

    @classmethod 
    def from_element(cls, element : "Element"):
        return cls(*compute_volume_matrices_linear_jacobian(element), *compute_face_matrices_linear_jacobian(element))
    
    @property
    def n_basis(self) -> int:
        return self.mass_matrix.shape[0]
    
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PrecomputedGeometricData:
    '''
    Convenience class to bundle all precomputed geometric data required for the solution of the transport equation on a domain
    This is specific to a given domain geometry. Therefore, it is separate from FEMMatrices.
    '''

    vol_jacobians         : jnp.ndarray #[n_elements, n_dim, n_dim]
    r'''
    Volume Jacobians for all elements. Computed as:

    .. math::
        \mathbf{J}_{e,ij} = \frac{\partial x_i}{\partial r_j}

    where :math:`\mathbf{x}` are the real coordinates in the element :math:`e`, and :math:`\mathbf{r}` are the reference element coordinates.

    Shape [n_elements, n_dim, n_dim].
    '''

    vol_jacobians_inv     : jnp.ndarray #[n_elements, n_dim, n_dim]
    '''
    Inverse of the volume Jacobians for all elements. Shape [n_elements, n_dim, n_dim].
    '''

    vol_jacobians_det     : jnp.ndarray #[n_elements]
    '''
    Determinant of the volume Jacobians for all elements. Shape [n_elements].
    '''

    face_jacobians_det    : jnp.ndarray #[n_elements, n_faces_per_element]
    '''
    Determinant of the face Jacobians for all elements and faces. Shape [n_elements, n_faces_per_element].
    '''

    @property 
    def n_elements(self) -> int:
        return self.vol_jacobians.shape[0]


#======================================================================================================================================================================
#                                                               Base Element Class
#=======================================================================================================================================================================    

class Element(ABC):    
    '''
    Base class for finite elements in jax-sn. This provides functions for computing the finite element matrices, as well as for mapping between local and real coordinates.

    Methods to tabulate basis functions, get quadrature points and weights, and other element-specific operations are defined here as abstract methods to be implemented by concrete element classes.
    These do not necessarily need to be jittable, since they are typically only called during setup and precomputation. They can thus use non-jittable libraries such as basix.
    The only exception is the :ref:`precompute_geometric_data` method; this needs to be jittable and traceable to allow for derivative computations with respect to the geometry.
    '''
    
    @property
    @abstractmethod        
    def volume_quadrature(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Returns the quadrature points and weights for the element that are sufficiently accurate for the volume integrals
        (in other words, 2 times the maximum polynomial degree of the basis functions)
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

    @property 
    @abstractmethod
    def all_connection_face_quadratures(self) -> List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        '''
        Returns the quadrature points, and weights for all possible face connections of each face.

        The order needs to be as follows (row major):

        most significant index:
            face number of this element
        middle index: 
            permutation number of the face (as defined by its Lehmer code like in  _get_all_permutations in src/jax_sn/tools/permutation_tools.py)
        least significant index:
            face number of the connected element

        Purposefully not a stacked array since in principle the number of permutations can vary per face (e.g. for non-simplicial elements).        

        '''
        ...
    

    @property
    @abstractmethod    
    def n_faces(self) -> int:
        '''
        Returns the number of faces of the element.
        '''
        ...

    @property 
    @abstractmethod
    def n_basis(self) -> int:
        '''
        Returns the number of basis functions of the element.
        '''
        ...

    @property
    @abstractmethod
    def face_template(self) -> List[Tuple]:
        '''
        Returns the face template of the element.

        The face template is a list of tuples, where each tuple contains the indices of the vertices that make up a face.

        For example, for a triangle element, the face template would be [(0, 1), (1, 2), (2, 0)].
        '''
        ...
    
    @abstractmethod
    def n_vertices_per_face(self, face_idx : int) -> int:
        '''
        Returns the number of vertices per face of the element.

        Parameters  
        ----------
        face_idx : int
            The index of the face.
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


    @abstractmethod
    def compatible(self, geometric_domain : GeometricDomain) -> bool:
        '''
        Checks whether the element is compatible with the given geometric domain.

        This can include checks such as whether the number of faces of the element matches the number of faces in the geometric domain,
        whether the number of vertices per face matches, etc.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain to check compatibility with.
        '''
        ...


    @abstractmethod
    def compute_geometric_data(self, geometric_domain : GeometricDomain) -> PrecomputedGeometricData:
        '''
        Computes the Jacobians for the element given a geometric domain. Must be jittable/traceable:
        unlike :meth:`precompute_geometric_data`, this must NOT call :meth:`assert_compatible` (or
        otherwise touch ``geometric_domain`` in a non-traceable way), since it is also used to
        recompute geometric data under ``jit``/``grad`` when the domain vertices are perturbed (e.g.
        by a parametrisation reconstructing the operator after a geometry deformation).

        This is required to be dependent on the element: a different coordinate interpretation (i.e. a different
        standard element) in the type of finite element leads to a different reference-element-to-element mapping, and thus a different
        Jacobian.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain containing the mesh and adjacency information. Here, we only use geometric_domain.vertices and geometric_domain.connectivity.

        Returns
        -------

        PrecomputedGeometricData
            The precomputed geometric data for all elements [does not depend on direction or energy group so is the same for all angles and energy groups]
        '''
        ...

    @abstractmethod
    def local_coordinates(self, real_coordinates : jnp.ndarray, element_indices : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData) -> jnp.ndarray:
        '''
        Computes the local coordinates of the given points in the given elements.

        Parameters
        ----------
        points : jnp.ndarray [n_points, n_dim]
            The global coordinates of the points to compute local coordinates for.
        element_indices : jnp.ndarray [n_points]
            The indices of the elements containing the points.
        geometric_domain : GeometricDomain
            The geometric domain containing the mesh and adjacency information.
        Returns
        -------
        jnp.ndarray [n_points, n_dim]
            The local coordinates of the points in the given elements.
        '''
        ...

    @abstractmethod
    def real_coordinates(self, local_coordinates : jnp.ndarray, element_indices : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData):
        '''
        Computes the real coordinates of the given local coordinates in the given elements.

        Parameters
        ----------
        local_coordinates : jnp.ndarray [n_points, n_dim]
            The local coordinates of the points to compute real coordinates for.
        element_indices : jnp.ndarray [n_points]
            The indices of the elements containing the points.
        geometric_domain : GeometricDomain
            The geometric domain containing the mesh and adjacency information.
        Returns
        -------
        jnp.ndarray [n_points, n_dim]
            The real coordinates of the points in the given elements.            
        '''
        ...

    @abstractmethod        
    def volume_quadrature_order(self, n_deg : int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Returns the quadrature points and weights for the element that are sufficiently accurate for the volume integrals
        of polynomials up to degree n_deg.

        Parameters
        ----------
        n_deg : int
            The maximum polynomial degree to integrate exactly.
        
        Returns
        -------
        points : jnp.ndarray [n_quadrature_points, n_dim]
            The quadrature points.
        weights : jnp.ndarray [n_quadrature_points]
            The quadrature weights.
        '''
        ...


    def assert_compatible(self, geometric_domain : GeometricDomain):
        '''
        Asserts that the element is compatible with the given geometric domain.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain to check compatibility with.
        '''
        assert self.compatible(geometric_domain), "Element is not compatible with the given geometric domain."

    def precompute_geometric_data(self, geometric_domain : GeometricDomain) -> PrecomputedGeometricData:
        '''
        Asserts that ``geometric_domain`` is compatible with this element, then computes its geometric
        data. NOT jittable (the compatibility check is not traceable) -- this is meant for the initial,
        untraced setup of a :class:`~jax_sn.solution_domain.SolutionDomain`. If you need to recompute
        geometric data for an already-validated, possibly-traced domain (e.g. after perturbing
        vertices under ``jit``/``grad``), call :meth:`compute_geometric_data` directly instead.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain to precompute geometric data for.

        Returns
        -------
        PrecomputedGeometricData
            The precomputed geometric data for all elements.
        '''
        self.assert_compatible(geometric_domain)
        return self.compute_geometric_data(geometric_domain)


    def interpolate_from_local_coordinate(self, local_coordinates : jnp.ndarray, element_indices : jnp.ndarray, values_array : jnp.ndarray) :
        '''
        Interpolates the given values at the given local coordinates in the given elements.

        Parameters
        ----------
        local_coordinates : jnp.ndarray [n_points, n_dim]
            The local coordinates of the points to interpolate.
        element_indices : jnp.ndarray [n_points]
            The indices of the elements containing the points.
        values_array : jnp.ndarray [n_elements, n_basis]
            The values of each basis function for each element.

        Returns
        -------
        jnp.ndarray [n_points]
            The interpolated values at the given local coordinates in the given elements.
        '''
        hi = self.tabulate_basis_functions_first_derivative(local_coordinates)[0,:,:]  # [n_points, n_basis]
        element_values = values_array[element_indices]                                 # [n_points, n_basis]
        return jnp.einsum("pi,pi->p", hi, element_values)                             # [n_points]
    
    def interpolate_from_real_coordinate(self, real_coordinates : jnp.ndarray, element_indices : jnp.ndarray, geometric_domain : GeometricDomain, precomputed_geometric_data : PrecomputedGeometricData, values_array : jnp.ndarray) :
        '''
        Interpolates the given values at the given real coordinates in the given elements.

        Parameters
        ----------
        real_coordinates : jnp.ndarray [n_points, n_dim]
            The real coordinates of the points to interpolate.
        element_indices : jnp.ndarray [n_points]
            The indices of the elements containing the points.
        geometric_domain : GeometricDomain
            The geometric domain containing the mesh and adjacency information.
        precomputed_geometric_data : PrecomputedGeometricData
            The precomputed geometric data for the elements.
        values_array : jnp.ndarray [n_elements, n_basis]
            The values of each basis functions for each element.

        Returns
        -------
        jnp.ndarray [n_points]
            The interpolated values at the given real coordinates in the given elements.
        '''
        local_coordinates = self.local_coordinates(real_coordinates, element_indices, geometric_domain, precomputed_geometric_data)  # [n_points, n_dim]
        return self.interpolate_from_local_coordinate(local_coordinates, element_indices, values_array)                             # [n_points]


#======================================================================================================================================================================
#                                                               Finite Element Matrix Computations
#=======================================================================================================================================================================
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
    r'''
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
    r'''
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
    r'''
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

def _compute_face_outflow_matrices(element : Element) -> jnp.ndarray:
    r'''
    Computes the face outflow matrices for the given element.

    i.e. C_{ij}^{\gamma} = \\int_{\gamma} h_i h_j dA

    Parameters
    ----------
    element : Element
        The finite element for which to compute the face outflow matrices.
    Returns
    -------
    jnp.ndarray [n_faces, n_basis, n_basis]
        The face outflow matrices for each face of the element.
    
    '''    
    face_quadratures = element.face_quadratures    

    def compute_face_outflow_matrix(face_idx : int):
        p_i, w_i = face_quadratures[face_idx]                                  # [n_quadrature_points, n_dim], [n_quadrature_points]
        h_face = element.tabulate_basis_functions_first_derivative(p_i)[0,:,:] # [n_quadrature_points, points, n_basis]
        return _compute_mass_matrix(h_face, w_i)                               # [n_basis, n_basis]

    return jnp.stack([compute_face_outflow_matrix(face_idx) for face_idx in range(element.n_faces)], axis=0)  # [n_faces, n_basis, n_basis]


def _compute_face_inflow_matrices(element : Element) -> jnp.ndarray:
    r'''
    Computes the face inflow matrices for the given element.

    i.e. D_{ij}^{\gamma_1,p, \gamma_2} = \\int_{\gamma_1} h_i \\overline{h}_j dA

    Parameters
    ----------
    element : Element
        The finite element for which to compute the face inflow matrices.
    Returns
    -------
    jnp.ndarray [n_faces, n_basis, n_basis]
        The face inflow matrices for each face of the element.
    
    '''    
    face_quadratures = element.all_connection_face_quadratures

    def compute_face_inflow_matrix(face_idx : int):
        p_i_self, p_i_conn, wi = face_quadratures[face_idx]                                  # [n_quadrature_points, n_dim], [n_quadrature_points]
        h_face = element.tabulate_basis_functions_first_derivative(p_i_self)[0,:,:] # [n_quadrature_points, points, n_basis]
        h_face_conn = element.tabulate_basis_functions_first_derivative(p_i_conn)[0,:,:] # [n_quadrature_points, points, n_basis]
        hihj = jnp.einsum("qi,qj->qij", h_face, h_face_conn)  #[n_quadrature_points, n_basis, n_basis]        
        return jnp.einsum("qij,q->ij", hihj, wi)  #[n_basis, n_basis]        

    return jnp.stack([compute_face_inflow_matrix(face_idx) for face_idx in range(len(face_quadratures))], axis=0)  # [n_faces, n_basis, n_basis]

def compute_face_matrices_linear_jacobian(element : Element):
    '''
    Computes the face inflow and outflow matrices for an element with a linear Jacobian.

    Parameters
    ----------
    element : Element
        The finite element for which to compute the face matrices.
    Returns
    -------
    face_outflow_matrices : jnp.ndarray
        The face outflow matrices of shape [n_faces, n_basis, n_basis].
    face_inflow_matrices : jnp.ndarray
        The face inflow matrices of shape [n_faces, n_basis, n_basis].
    '''
    assert element.has_linear_jacobian, "Element must have a linear Jacobian!"
    
    face_outflow_matrices = _compute_face_outflow_matrices(element)
    face_inflow_matrices  = _compute_face_inflow_matrices(element)
    return face_outflow_matrices, face_inflow_matrices.reshape(element.n_faces, -1, element.n_faces, element.n_basis, element.n_basis)