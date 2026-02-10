Reduced Quadrature Set
=======================

In 1D or 2D problems, the resulting angular flux will not depend on the symmetric dimensions. In a 1D slab geometry, we will have 

.. math::
    \psi(\mathbf{r}, \Omega, E) = \psi(x, \Omega_x, E)
    
In a 2D slab geometry we similarly have 

.. math::
    \psi(\mathbf{r}, \Omega, E) = \psi(x, y, \Omega_x, \Omega_y, E)

For the transport operator, using a full 3D quadrature set is therefore significantly more expensive than necessary: the same :math:`\Omega_x` will result in the same resulting :math:`T^{-1}` application.
Therefore, we can use a reduced quadrature set where we only have one quadrature point per unique :math:`\Omega_x` (and :math:`\Omega_y` in 2D). 

This reduced quadrature set does have to provide a method to generate the :math:`w_k Y_{lm}(\Omega_k)` terms required for the moment calculation.


The normal quadrature will have the following moment calculation: 

.. math::
    \int_{S^2} Y_{lm}(\Omega) \psi( \Omega) d\Omega \approx \sum_k w_k Y_{lm}(\Omega_k) \psi(\Omega_k) \equiv \sum_k A_{lm, k} \psi(\Omega_k)

Let :math:`\{\Omega_k\}` be the full, 3D quadrature set. We define a set of orbits :math:`\mathcal{K}` which groups the quadrature points by their unique :math:`\Omega_x` (and :math:`\Omega_y` in 2D) values. 
Let :math:`\kappa ` index an orbit, with :math:`f` indexing a orbit member. Let :math`\tilde{\Omega}_{\kappa}` be a single representative from the orbit. The sum can be written as 

.. math:: 
    \sum_{\kappa}\sum_{f} w_{\kappa f} Y_{lm}(\Omega_{\kappa f}) \psi(\Omega_{\kappa f})

However, by the definition of an orbit, we have that :math:`\psi(\Omega_{\kappa f}) = \psi(\tilde{\Omega}_{\kappa})` for all :math:`f`. Therefore, we can write the sum as

.. math:: 
    \sum_{\kappa} \left( \sum_{f} w_{\kappa f} Y_{lm}(\Omega_{\kappa f}) \right) \psi(\tilde{\Omega}_{\kappa}) \equiv \sum_{\kappa} \tilde{A}_{lm, \kappa} \psi(\tilde{\Omega}_{\kappa})

Thus, if we modify the quadrature set to use :math:`\tilde{\Omega}_{\kappa}` as the quadrature points and :math:`\tilde{A}_{lm, \kappa}` as the moment calculation coefficients, we will get the same result as the full quadrature set.

The reverse expansion can be the same; that maps from the moments to the angular flux and is not affected by the reduction of the number of angles.



