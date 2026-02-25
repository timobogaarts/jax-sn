Multigroup discretisation
===========================

If we take the standard neutron transport equation:

.. math::
    \Omega \cdot \nabla \psi(\mathbf{r}, \Omega, E) + &\sigma(\mathbf{r}, \Omega) \psi(\mathbf{r}, \Omega, E) \\
     &=\iint \sigma_s(\mathbf{r}, \Omega' \to \Omega, E' \to E) \psi(\mathbf{r}, \Omega', E') dE' d\Omega'  + Q(\mathbf{r}, \Omega, E)

and integrate it over energy, it can be shown (see :ref:`here <appendix_multigroup_derivation>` for a detailed derivation) that the resulting equation is:

.. math::
    \Omega \cdot \nabla \psi_{g}(\mathbf{r}, \Omega) + &\sigma_{g}(\mathbf{r}, \Omega) \psi_{  g}(\mathbf{r}, \Omega) \\
                &= \sum_{g'} \sum_{lm} Y_{lm}(\Omega) \sigma^s_{lm, g, g'    }(\mathbf{r}) \int_{S^2} Y_{lm}(\Omega')  \psi_{g'}(\mathbf{r}, \Omega') d\Omega' + Q_{g}(\mathbf{r}, \Omega) \quad \forall g

where :math:`\psi_{g}(\mathbf{r}, \Omega) = \int_{E_g}^{E_{g-1}} \psi(\mathbf{r}, \Omega, E) dE` is the group-integrated angular flux, :math:`Q_g(\mathbf{r}, \Omega) = \int_{E_g}^{E_{g-1}} Q(\mathbf{r}, \Omega, E) dE` is the group-integrated external source, 
:math:`\sigma_{g}(\mathbf{r}, \Omega)` is the group-averaged total cross section,
:math:`Y_{lm}(\Omega)` are the real spherical harmonics, and :math:`\sigma^s_{lm, g, g'    }(\mathbf{r})` are the Legendre moments of the group-averaged scattering cross section from :math:`g'` to :math:`g`.

By convention, we take the energy groups such that :math:`E_0` is the highest energy and :math:`E_{G}` is the lowest energy.

Thus, the multigroup transport equation is a discrete set of coupled spatial-angular PDEs, one for each energy group.

As mentioned in the :ref:`linear_systems_differentation` section, we want to switch to an operator viewpoint as much as possible.

We can write the above equation in a more compact form as:

.. math::
    \mathcal{T}_g[\psi_g] = \sum_{g'} \mathcal{S}_{gg'}[\psi_{g'}] + Q_g \quad \forall g  

The :math:`\mathcal{T}_g = \Omega\cdot \nabla + \sigma_g` transport operator contains the streaming and total interaction terms, while the :math:`\mathcal{S}_{gg'} = \sum_{lm} Y_{lm}(\Omega) \sigma^s_{lm, g, g'    }(\mathbf{r}) \int_{S^2} Y_{lm}(\Omega') \cdot d\Omega'` scattering operator contains the scattering interactions from all groups :math:`g'` to group :math:`g`. 
This scattering operator provides the coupling between different energy groups. 

The multigroup transport equation structure is then as (note that all terms, and in fact operators themselves, still depend on :math:`\mathbf{r}` and :math:`\Omega`):



.. math::
    :label: eq:multigroup

    \begin{bmatrix} \mathcal{T}_0  - \mathcal{S}_{00} & -\mathcal{S}_{01} & \cdots & -\mathcal{S}_{0G} \\ -\mathcal{S}_{10} & \mathcal{T}_1  - \mathcal{S}_{11}& \cdots & -\mathcal{S}_{1G  } \\ \vdots & \vdots & \ddots & \vdots \\ -\mathcal{S}_{G0} & -\mathcal{S}_{G1} & \cdots & \mathcal{T}_G-  \mathcal{S}_{GG}\end{bmatrix}  \cdot \begin{bmatrix} \psi_0 \\ \psi_1 \\ \vdots \\ \psi_G\end{bmatrix} =  \begin{bmatrix} Q_0 \\ Q_1 \\ \vdots \\ Q_G\end{bmatrix}

Now, if the scattering operator only has downscatter, this implies that all :math:`S_{gg'}` with :math:`g' > g` are zero. This results in a lower-triangular operator structure, which can be solved by forward substitution (note the transpose can be solved by backward substitution!).




