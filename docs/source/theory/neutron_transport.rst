Neutron transport 
===================

In neutron transport, we are interested in obtaining the distribution of neutrosn in phase space. A logical quantity is then the neutron density in phase space:

.. math::
    N(\mathbf{r}, \Omega, E) dV d\Omega dE = &\text{number of neutrons in } dV \text{ at } \mathbf{r}  \\
                                             &\text{ traveling in } d\Omega \text{ around } \Omega     \\
                                             &\text{ with energy in } dE \text{ around } E

However, when we want to calculate how many collisions (with e.g. a detector) occur, this is calculated as:


.. math::
    \text{collisions per second} = \iiint \sigma_c(\mathbf{r}, \Omega, E) v(E) N(\mathbf{r}, \Omega, E) dV d\Omega dE

where :math:`\sigma_c(\mathbf{r}, \Omega, E)` is the macroscopic cross section for the collision type of interest and :math:`v(E)` is the neutron speed at energy :math:`E`. The reason is that 
neutrons with a large speed, have a larger path length in a time interval and thus have a higher probability of colliding.

It is therefore often more convenient to work with the angular neutron flux, defined as:

.. math::
    \psi(\mathbf{r}, \Omega, E) = v(E) N(\mathbf{r}, \Omega, E).


Neutron transport equation
---------------------------
It can then be shown, that the behavior of neutrons in a medium is governed by the neutron transport equation (here in steady-state):

.. math::
    \Omega \cdot \nabla \psi(\mathbf{r}, \Omega, E) + &\sigma(\mathbf{r}, \Omega) \psi(\mathbf{r}, \Omega, E) \\
     &=\iint \sigma_s(\mathbf{r}, \Omega' \to \Omega, E' \to E) \psi(\mathbf{r}, \Omega', E') dE' d\Omega'  + Q(\mathbf{r}, \Omega, E)

where :math:`\sigma(\mathbf{r}, \Omega)` is the total macroscopic cross section, :math:`\sigma_s(\mathbf{r}, \Omega' \to \Omega, E' \to E)` is the differential scattering cross section from direction :math:`\Omega'` and energy :math:`E'` to direction :math:`\Omega` and energy :math:`E`, and :math:`Q(\mathbf{r}, \Omega, E)` is an external source term.

The first term on the left-hand side represents streaming (similar to :math:`\mathbf{v}\cdot \nabla f` in the Boltzmann equation), the second term represents losses due to interactions,
the first term on the right-hand side represents scattering from other energies and directions, and the second term on the right-hand side represents external sources (in the sense that they are not 
produced by any neutron interactions). Intuitively, the transport equation is a balance equation stating that the rate of change of neutrons in a given phase space volume is equal to the neutrons streaming in minus the neutrons streaming out,
plus the neutrons created by scattering and external sources minus the neutrons lost due to interactions.

This equation is 6-dimensional (3 spatial dimensions, 2 angular dimensions, 1 energy dimension) and integro-differential. It 
is often solved using Monte-Carlo methods, where the above equation is sampled. 

In deterministic methods, instead, the equation is fully discretised in all 6 dimensions. Since the neutron transport equation is linear, 
the discetised version will also be a linear system, which will need to be efficiently solved. Furthermore,
in the end, we are also interested in obtaining derivatives of a function solution of this equation with respect to some parameters (e.g., 
how the tritium breeding ratio in fusion blankets changes with respect to material composition or even 3D geometry). 

We want to keep this in mind when deriving a formulation of the neutron transport equation, and thus, before discretising, we will review some linear algebra.







