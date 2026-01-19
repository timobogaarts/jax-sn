.. _spherical_harmonics:

Real Spherical Harmonics
==========================

The spherical harmonics used in this work are defined as

.. math::
    Y_{lm}(\theta, \phi)= \begin{cases} \sqrt{2} \mathcal{I}(Y_l^{|m|}(\theta,\phi)) & m < 0 \\ Y_l^0(\theta, \phi) & m =0 \\ \sqrt{2} \mathcal{R}(Y_l^m(\theta, \phi)) & m >0 \end{cases}
    
where :math:`Y_{lm}` are the complex spherical harmonics defined as per jax.scipy.special.sph_harm_y (including Condon-Shortley phase which is included in the :math:`P_l^m` definition):

.. math::
    Y_{l}^m(theta, phi) = \sqrt{\frac{2l+1}{4\pi} \cdot \frac{(l-m)!}{(l+m)!}} \cdot P_l^m(\cos(\theta)) \cdot e^{i m \phi}

If we assume that :math:`\int d\Omega = 4 \pi`, then the normalization is such that:

.. math::
    \int Y_{lm}(\theta, \phi) Y_{l'm'}(\theta, \phi) d\Omega = \delta_{ll'} \delta_{mm'}


Expansion and inverse
----------------------
Given a particular function :math:`f(\Omega)`, we can expand it in spherical harmonics as:

.. math::
    f(\Omega_k) = \sum_{lm} f_{lm} Y_{lm}(\Omega_k) \\
    f_{lm} = \int f(\Omega) Y_{lm}(\Omega) d\Omega \approx \sum_k w_k f(\Omega_k) Y_{lm}(\Omega_k)


Scalar flux 
----------------
The so-called scalar flux :math:`\phi` is defined as the zeroth moment coefficient as:

.. math::
    \phi = \sqrt{4\pi} \psi_{00} 