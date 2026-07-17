FW-CADIS
-------------
FW-CADIS is a variance reduction technique which uses deterministic importance functions to generate weight windows for 
stochastic simulations. For a particular fixed-source problem with source :math:`q`, it solves the forward equation to obtain :math:`\psi`. 

Then, it solves the adjoint neutron transport equation (same as forward but angle reversed and scattering energy reversed):

.. math::
    - \Omega \cdot \nabla \psi^{\dagger}(\mathbf{r}, \Omega, E) + &\sigma(\mathbf{r}, \Omega) \psi^{\dagger}(\mathbf{r}, \Omega, E) \\
     &=\iint \sigma_s(\mathbf{r}, \Omega' \to \Omega, E \to E') \psi^{\dagger}(\mathbf{r}, \Omega', E') dE' d\Omega'  + q^{\dagger}(\mathbf{r}, \Omega, E)

with a source 

.. math::
    q^\dagger = 1/\phi

The idea is that lower flux regions are weighted more in the importance calculation. 

Solving this adjoint equation works exactly the same as the forward equation, except that the transport operator is transpposed (remember this just changes the angle exactly like here) and the *total scatter operator* is transposed
(not the gathering operator!). 

.. math::
    \mathbf{I} - \mathbf{M}\mathbf{T}^{-1} \mathbf{H}\mathbf{D}\mathbf{S} \to \mathbf{I} - \mathbf{M}\mathbf{T}^{-1} \mathbf{H}\mathbf{D}\mathbf{S}^T

Ideally we want to re-use all the machinery from the forward solve. We can do this by after computing the relevant matrices for the :math:`\mathbf{D}, \mathbf{M}` operators (which take the normal angles), just reversing the 
angles (and topological inversion order) and swapping the scatter matrix out and incoming groups. Given this new data, the forward operator is exactly the same as the 
continuous adjoint. 

