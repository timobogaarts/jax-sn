Angular discretisation
--------------------------

We are now going to expand the :math:`\psi(\mathbf{r}, \Omega)` angular flux in terms of the legendre moments:

.. math::
    \psi_g(\mathbf{r}, \Omega) = \sum_{lm} Y_{lm}(\Omega) \phi_{glm}(\mathbf{r})

where :math:`Y_{lm}(\Omega)` are the real spherical harmonics and :math:`\phi_{glm}(\mathbf{r})` are the legendre moments of the angular flux, which could be computed by:  

.. math::
    \phi_{glm}(\mathbf{r}) = \int_{S^2} Y_{lm}(\Omega) \psi_g(\mathbf{r}, \Omega) d\Omega

We now note that by the definition of the scattering operator 

.. math::
    \mathcal{S}_{gg'}[\psi_g(\mathbf{r}, \Omega)] = \sum_{lm} Y_{lm}(\Omega) \sigma^s_{lm, g, g'    }(\mathbf{r}) \int_{S^2} Y_{lm}(\Omega')  \psi_{g'}(\mathbf{r}, \Omega') d\Omega'
    \equiv \sum_{lm} Y_{lm}(\Omega) \sigma^s_{lm, g, g'    }(\mathbf{r}) \phi_{g'lm}(\mathbf{r})

which is just an element-wise multiplication in the legendre moment space for :math:`\phi_{g'lm}(\mathbf{r})`!


The multigroup equations now look like 

.. math::
    \mathcal{T}_{g}[\sum_{lm} Y_{lm}(\Omega)\phi_{glm}(\mathbf{r})] = \sum_{g'} \mathcal{S}^{lm}_{gg'}[\phi_{g'lm}(\mathbf{r})] + Q_g
