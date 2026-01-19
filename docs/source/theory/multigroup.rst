Multigroup discretisation
===========================

We wish to prove the multigroup formm of the neutron transport equation.
We will show this for the general case (where multiple unknowns per group are allowed) and then specialize it to the standard multigroup transport equation.

Proof 
------

We define energy group :math:`g` as the energy interval :math:`E \in [E_{g}, E_{g-1}]`, where :math:`E_0` is the highest energy and :math:`E_G` is the lowest energy (for a total of :math:`G` energy groups).
We now modify the neutron transport equation

.. math::
    \Omega \cdot \nabla \psi(\mathbf{r}, \Omega, E) + &\sigma(\mathbf{r}, \Omega) \psi(\mathbf{r}, \Omega, E) \\
     &=\iint \sigma^s(\mathbf{r}, \Omega' \to \Omega, E' \to E) \psi(\mathbf{r}, \Omega', E') dE' d\Omega'  + Q(\mathbf{r}, \Omega, E)

as follows. Multiply the transport equation by :math:`\chi_{gp}(E)` where :math:`\chi_{gp} = 0` if :math:`E \notin [E_{g}, E_{g-1}]` and :math:`p` is the index of the basis function in the energy group.
Then, integrate over :math:`E`.

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

Similarly, for the right-hand-side, we have

.. math::
    &\iint \left( \int_{g} \chi_{gp} \sigma^s  (\mathbf{r}, \Omega' \to \Omega, E' \to E) dE \right) \psi(\mathbf{r}, \Omega', E') dE' d\Omega'  + \int_{g} \chi_{gp} Q(\mathbf{r}, \Omega, E) dE \\
    &=\iint \sigma_{s, gp g'p'}(\mathbf{r}, \Omega' \to \Omega; \psi) \psi_{g'p'}(\mathbf{r}, \Omega') d\Omega' + Q_{gp}(\mathbf{r}, \Omega)
    
where we have defined 
