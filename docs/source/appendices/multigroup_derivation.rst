Multigroup discretisation
===========================

THIS IS WRONG!

We wish to prove the multigroup formm of the neutron transport equation.
We will show this for the general case (where multiple unknowns per group are allowed) and then specialize it to the standard multigroup transport equation.

Proof 
------

We define energy group :math:`g` as the energy interval :math:`E \in [E_{g}, E_{g-1}]`, where :math:`E_0` is the highest energy and :math:`E_G` is the lowest energy (for a total of :math:`G` energy groups).

We now modify the neutron transport equation

.. math::
    \Omega \cdot \nabla \psi(\mathbf{r}, \Omega, E) + &\sigma(\mathbf{r}, \Omega) \psi(\mathbf{r}, \Omega, E) \\
     &=\iint \sigma^s(\mathbf{r}, \Omega' \to \Omega, E' \to E) \psi(\mathbf{r}, \Omega', E') dE' d\Omega'  + Q(\mathbf{r}, \Omega, E)

as follows. For each group, we define basis functions :math:`\chi_{gp}`. These basis functions have :math:`\sum_{p} \chi_{gp} = 1` for all :math:`E \in [E_{g}, E_{g-1}]`.
Multiply the transport equation by :math:`\chi_{gp}(E)` and integrate over :math:`E`.


Left hand side 
~~~~~~~~~~~~~~~~~~~
For the left-hand-side, we have 

.. math::
    \Omega \cdot \nabla \int_{g} \chi_{gp} \psi(\mathbf{r}, \Omega, E) dE + &\int_{g} \chi_{gp} \sigma(\mathbf{r}, \Omega, E) \psi(\mathbf{r}, \Omega, E) dE \\

By defining the quantities

.. math::
    &\psi_{gp}(\mathbf{r}, \Omega) = \int_{g} \chi_{gp} \psi dE\\
    &\sigma_{gp}(\mathbf{r}, \Omega) = \frac{\int_{g} \chi_{gp} \sigma \psi dE}{\psi_{gp}(\mathbf{r}, \Omega)}

this can be rewritten as 

.. math::
    \Omega \cdot \nabla \psi_{gp}(\mathbf{r}, \Omega) + \sigma_{gp}(\mathbf{r}, \Omega; \psi) \psi_{gp}(\mathbf{r}, \Omega)


Scattering term 
~~~~~~~~~~~~~~~~~~~~

For the right-hand-side scattering term, we recognize that, in most cases, scattering only depends on :math:`\Omega'\cdot \Omega` instead of both separately. Then, we can expand it into a Legendre series:

.. math::
    \sigma^s(\mathbf{r}, \Omega' \to \Omega, E' \to E) \equiv \sigma^s(\mathbf{r}, \Omega' \cdot \Omega, E' \to E) =\sum_{l=0}^{\infty} \frac{2l+1}{4\pi} P_l(\Omega \cdot \Omega') \sigma^s_{l}(\mathbf{r}, E' \to E)

where :math:`P_l` are the Legendre polynomials and :math:`\sigma^s_{l}(\mathbf{r}, E' \to E)` are the Legendre moments of the scattering cross section, computed by:

.. math::
    \sigma^s_{l}(\mathbf{r}, E' \to E) = 2\pi \int_{-1}^{1} P_l(\mu) \sigma^s(\mathbf{r}, \mu, E' \to E) d\mu

This significantly reduces the amount of data required to represent the scattering cross section. Furthermore, this is in practice a good approximation: truncating the sum to the first three terms is already
sufficient in many cases. By the spherical harmonic addition theorem and using the definition of the :ref:`spherical harmonics <spherical_harmonics>`, this can be rewritten as

.. math::
    \sigma^s(\mathbf{r}, \Omega' \to \Omega, E' \to E) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} Y_{lm}(\Omega) Y_{lm}(\Omega') \sigma^s_{l}(\mathbf{r}, E' \to E)


Then, we can fill this into the scattering term (we separate the :math:`E` integral into group integrals):

.. math::
    &\int_{g} \sum_{g'}\int_{g'} \int_{S^2} \sum_{l=0}^{\infty} \sum_{m=-l}^{l} Y_{lm}(\Omega) Y_{lm}(\Omega') \sigma^s_{l}(\mathbf{r}, E' \to E)\psi(\mathbf{r}, \Omega', E') d\Omega' dE' \chi_{gp}(E) dE \\
    &= \sum_{g'} \sum_{lm} Y_{lm}(\Omega) \int_g \int_{g'}\int_{S^2} \sigma^s_{l}(\mathbf{r}, E' \to E) Y_{lm}(\Omega') \psi(\mathbf{r}, \Omega', E')  \chi_{gp}(E) dE d\Omega' dE' \\
    
Under the integral, we are free to multiply with :math:`\sum_{g'p'} \chi_{g'p'}(E')=1`. Then, taking the sum out of the integral:

.. math::    
    \sum_{g'p'} \sum_{lm} Y_{lm}(\Omega) \int_g \int_{g'}\int_{S^2}\chi_{g'p'} (E')\sigma^s_{l}(\mathbf{r}, E' \to E) Y_{lm}(\Omega') \psi(\mathbf{r}, \Omega', E')  \chi_{gp}(E) dE d\Omega' dE' \\

Then, we define:

.. math::
    \sigma^s_{l, gp, g'p'}(\mathbf{r}; \psi) = \frac{\int_g \int_{g'} \sigma^s_{l}(\mathbf{r}, E' \to E) \chi_{gp}(E) \chi_{g'p'}(E') \psi(\mathbf{r}, \Omega', E') dE dE'}{\int_{g'} Y_{lm}\psi(\mathbf{r}, \Omega', E') dE'}

and we obtain:

.. math::
    \sum_{g'p'} \sum_{lm} Y_{lm}(\Omega) \sigma^s_{l, gp, g'p'}(\mathbf{r}; \psi) \int_{S^2} Y_{lm}(\Omega')  \psi_{g'p'}(\mathbf{r}, \Omega') d\Omega'

External Source term
~~~~~~~~~~~~~~~~~~~~~~~

Finally, for the external source term we can simply define:

.. math::
    Q_{gp}(\mathbf{r}, \Omega) = \int_{g} \chi_{gp} Q(\mathbf{r}, \Omega, E) dE


Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~~


This yields the multigroup transport equation, which is a *nonlinear* equation due to the dependence of the cross sections on the fluxes:


.. math::
    \Omega \cdot \nabla \psi_{gp}(\mathbf{r}, \Omega) + &\sigma_{gp}(\mathbf{r}, \Omega; \psi) \psi_{gp}(\mathbf{r}, \Omega) \\
     &= \sum_{g'} \sum_{lm} Y_{lm}(\Omega) \sigma^s_{l, gp, g'p'}(\mathbf{r}; \psi) \int_{S^2} Y_{lm}(\Omega')  \psi_{g'p'}(\mathbf{r}, \Omega') d\Omega' + Q_{gp}(\mathbf{r}, \Omega)

The way out is to use a closure relation. This closure relation approximates the dependence of the cross sections on the fluxes by a good guess. If this guess is exactly right, this consistently yields the exact solution.
In practice, one uses cross sections generated from a different code, possibly in a simplified scenario (e.g. infinite medium, homogeneous material compositions, etc.). If the guess is good, then the closure approximation is good, 
and the result of this equation will be good.

This yields the standard linear multigroup transport equation used in practice:


.. math::
    \Omega \cdot \nabla \psi_{gp}(\mathbf{r}, \Omega) + &\sigma_{gp}(\mathbf{r}, \Omega) \psi_{gp}(\mathbf{r}, \Omega) \\
     &= \sum_{g'} \sum_{lm} Y_{lm}(\Omega) \sigma^s_{l, gp, g'p'    }(\mathbf{r}) \int_{S^2} Y_{lm}(\Omega')  \psi_{g'p'}(\mathbf{r}, \Omega') d\Omega' + Q_{gp}(\mathbf{r}, \Omega)

Now, there is no dependence of the cross sections on the fluxes anymore. This can be self-consistenly solved by providing appropriate cross sections.

Simplification to single unknown per group
-------------------------------------------


If we assume that there is only a single unknown per group (i.e., :math:`p=1`), then we can drop the :math:`p` index everywhere. This yields the standard multigroup transport equation as used in many textbooks:

.. math::
    \Omega \cdot \nabla \psi_{gp}(\mathbf{r}, \Omega) + &\sigma_{g}(\mathbf{r}, \Omega) \psi_{  g}(\mathbf{r}, \Omega) \\
            &= \sum_{g'} \sum_{lm} Y_{lm}(\Omega) \sigma^s_{l, g, g'    }(\mathbf{r}) \int_{S^2} Y_{lm}(\Omega')  \psi_{g'}(\mathbf{r}, \Omega') d\Omega' + Q_{gp}(\mathbf{r}, \Omega)

with the multigroup cross sections defined as:

.. math::
    \sigma_{g}(\mathbf{r}, \Omega) &= \frac{\int_{g} \sigma(\mathbf{r}, \Omega, E) \psi(\mathbf{r}, \Omega, E) dE}{\int_{g} \psi(\mathbf{r}, \Omega, E) dE}\\
    \sigma^s_{l, g, g'    }(\mathbf{r}) &= \frac{\int_g \int_{g'} \sigma^s_{l}(\mathbf{r}, E' \to E) \psi(\mathbf{r}, \Omega', E') dE dE'}{\int_{g'} \psi(\mathbf{r}, \Omega', E') dE'}





Estimating multigroup cross sections
------------------------------------

In practice, multigroup cross sections can be obtained by continuous energy monte carlo methods in simplified geometry. The tallies required are simply the ones already shown:


.. math::
    \sigma_{gp}(\mathbf{r}, \Omega) = \frac{\int_{g} \chi_{gp} \sigma(\mathbf{r}, \Omega, E) \psi(\mathbf{r}, \Omega, E) dE}{\int_{g} \chi_{gp} \psi(\mathbf{r}, \Omega, E) dE}


