Multigroup discretisation
===========================

If we take the standard neutron transport equation:

.. math::
    \Omega \cdot \nabla \psi(\mathbf{r}, \Omega, E) + &\sigma(\mathbf{r}, \Omega) \psi(\mathbf{r}, \Omega, E) \\
     &=\iint \sigma_s(\mathbf{r}, \Omega' \to \Omega, E' \to E) \psi(\mathbf{r}, \Omega', E') dE' d\Omega'  + Q(\mathbf{r}, \Omega, E)

and integrate it over energy using the following assumptions:

- Cross sections are assumed to be constant and independent of the solution.
- Angular dependence of total cross section is ignored (arising from integrating over a angularly dependent angular flux).
- The infinite sum over scattering order :math:`\ell` is truncated to a finite maximum value :math:`L`. Even for highly anisotropic scattering (important in fusion due to the high energy neutrons), relatively low values are acceptable: the standard FENDL fusion nuclear data library provides :math:`L=6`.

it can be shown (see :ref:`here <appendix_multigroup_derivation>` for a detailed derivation) that the resulting equation is:

.. math::
    \left[\Omega \cdot \nabla  + \sigma_g(\mathbf{r})\right]\psi_g(\mathbf{r}, \Omega) = \sum_{\ell}\sum_{\tilde{g}}Y_{\ell}(\Omega) \sigma^s_{\ell  g  \tilde{g}}(\mathbf{r})\int_{S^2}Y_{\ell} (\tilde{\Omega})&\psi_{\tilde{g}}(\mathbf{r},\tilde{\Omega}) \,d \tilde{\Omega} \nonumber \\ &+ q_g(\mathbf{r}, \Omega) \ \ \ \forall g

where :math:`\psi_{g}(\mathbf{r}, \Omega) = \int_{E_g}^{E_{g-1}} \psi(\mathbf{r}, \Omega, E) dE` is the group-integrated angular flux, :math:`Q_g(\mathbf{r}, \Omega) = \int_{E_g}^{E_{g-1}} Q(\mathbf{r}, \Omega, E) dE` is the group-integrated external source, 
:math:`\sigma_{g}(\mathbf{r}, \Omega)` is the group-averaged total cross section,
:math:`Y_{\ell}(\Omega)` are the real spherical harmonics (:math:`\ell = (l,m)`), and :math:`\sigma^s_{\ell, g, g'    }(\mathbf{r})` are the Legendre moments of the group-averaged scattering cross section from :math:`g'` to :math:`g`.

By convention, we take the energy groups such that :math:`E_0` is the highest energy and :math:`E_{G}` is the lowest energy.

Thus, the multigroup transport equation is a discrete set of coupled spatial-angular PDEs, one for each energy group.
