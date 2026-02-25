Angular discretisation
--------------------------

We are now going to expand the :math:`\psi(\mathbf{r}, \Omega)` angular flux in terms of the Legendre moments:

.. math::
    \psi_g(\mathbf{r}, \Omega) = \sum_{lm} Y_{lm}(\Omega) \phi_{glm}(\mathbf{r})

where :math:`Y_{lm}(\Omega)` are the real spherical harmonics and :math:`\phi_{glm}(\mathbf{r})` are the Legendre moments of the angular flux, which could be computed by:  

.. math::
    \phi_{glm}(\mathbf{r}) = \int_{S^2} Y_{lm}(\Omega) \psi_g(\mathbf{r}, \Omega) d\Omega


To get a finite set of equations, we fill in the expansion into the multigroup equation

.. math::
    \mathcal{T}_g[\psi_g] = \sum_{g'} \mathcal{S}_{gg'}[\psi_{g'}] + Q_g \quad \forall g  

and then multiply both sides by :math:`Y_{lm}(\Omega)`, integrate over the unit sphere, and require that the equation holds for all :math:`l,m` (this is similar to a weak form in finite-element methods).
This results in the :math:`P_N` equations, which are a set of coupled PDEs for the Legendre moments of the angular flux. This procedure is 
described in detail :ref:`here <pn_methods>`. This yields the total linear system

.. math::
    \begin{bmatrix}
    \boldsymbol{\tau}_0 - \mathbf{S}_{00} & -\mathbf{S}_{01} & \cdots & -\mathbf{S}_{0G}\\
    -\mathbf{S}_{10} & \boldsymbol{\tau}_1 - \mathbf{S}_{11} & \cdots & -\mathbf{S}_{1G}\\
    \vdots & \vdots & \ddots & \vdots\\
    -\mathbf{S}_{G0} & -\mathbf{S}_{G1} & \cdots & \boldsymbol{\tau}_G - \mathbf{S}_{GG}
    \end{bmatrix} \cdot 
    \begin{bmatrix}
    \{\phi_{0,lm}\} \\ 
    \{\phi_{1,lm}\} \\ 
    \vdots \\ 
    \{\phi_{G,lm}\}
    \end{bmatrix} = 
    \begin{bmatrix}
    \{Q_{0,lm}\} \\ 
    \{Q_{1,lm}\} \\ 
    \vdots \\ 
    \{Q_{G,lm}\}
    \end{bmatrix}

with :math:`\boldsymbol{\tau}_g` a operator (coupling :math:`l,m` modes) mapping a set of spatially varying Legendre moments to another set of spatially varying Legendre moments. 
Its precise definition is given in the :ref:`appendix <pn_methods>`, but we won't use that here. It is the Legendre moments discretised version of :math:`\mathcal{T}_g = \Omega \cdot \nabla + \sigma_g`.
Furthermore :math:`\mathbf{S}_{gg'}` a diagonal operator (it does not couple different :math:`l,m` modes) that maps a set of spatially varying Legendre moments 
to another set of spatially varying Legendre moments by multiplying with the scattering cross section in the Legendre moment space:

.. math::
    \mathbf{S}_{gg'}[\{\phi_{g,lm}\}] = \{\sigma_{lm,gg'}(\mathbf{r}) \phi_{g'lm}(\mathbf{r}) \}

In principle, if we now discretize the spatial variable, we have a fully discrete set of equations in an operator structure as desired.


Conditioning
==================

Although this can work, we want to exploit the structure of the transport operator in order to arrive at a very well conditioned system (i.e. where we know that it is quite 
close to the identity matrix!). Well conditioned systems have the attractive property that the iterative solvers converge and that they converge quickly.

To do this, multiply both sides of the equation by a diagonal inverse of the transport operator of each group:

.. math::
    \begin{bmatrix}
    \mathbf{I} - \boldsymbol{\tau}_0^{-1}\mathbf{S}_{00} & -\boldsymbol{\tau}_0^{-1}\mathbf{S}_{01} & \cdots & -\boldsymbol{\tau}_0^{-1}\mathbf{S}_{0G}\\
    -\boldsymbol{\tau}_1^{-1}\mathbf{S}_{10} & \mathbf{I} - \boldsymbol{\tau}_1^{-1}\mathbf{S}_{11} & \cdots & -\boldsymbol{\tau}_1^{-1}\mathbf{S}_{1G}\\
    \vdots & \vdots & \ddots & \vdots\\
    -\boldsymbol{\tau}_G^{-1}\mathbf{S}_{G0} & -\boldsymbol{\tau}_G^{-1}\mathbf{S}_{G1} & \cdots & \mathbf{I} - \boldsymbol{\tau}_G^{-1}\mathbf{S}_{GG}
    \end{bmatrix} \cdot 
    \begin{bmatrix}
    \{\phi_{0,lm}\} \\ 
    \{\phi_{1,lm}\} \\ 
    \vdots \\ 
    \{\phi_{G,lm}\}
    \end{bmatrix} = 
    \begin{bmatrix}
    \boldsymbol{\tau}_0^{-1}\{Q_{0,lm}\} \\ 
    \boldsymbol{\tau}_1^{-1}\{Q_{1,lm}\} \\ 
    \vdots \\ 
    \boldsymbol{\tau}_G^{-1}\{Q_{G,lm}\}
    \end{bmatrix}

Note that as the scattering operator goes to 0, this is precisely the identity matrix! It might seem curious to do this, since 
the transport operator seems the most difficult operator and we are setting up the matrix product to already be the inverse. However, as we will show later, the transport operator can in fact
be inverted quite easily, and the most difficult part is the interaction between both operators. Nevertheless, the system as written without reconditioning is a valid system and can be solved; it is just 
not the system we are solving and this section only tries to motivate why we will be solving the conditioned system instead of the original system.

Now, very handwavey, we have that the spectral radius of the diagonal elements is

.. math::
    \rho(\boldsymbol{\tau}_g^{-1} \mathbf{S}_{gg}) = \rho((\Omega \cdot \nabla + \sigma_g)^{-1} \mathbf{S}_{gg'}) \approx \frac{1}{\sigma_g} \sigma_{gg,00} < 1 

where the last inequality holds by definition in non-multiplying systems (in multiplying systems, one should solve an eigenvalue problem, and not this fixed source problem).

So, in e.g. a lower triangular (downscattering system), we can solve by foward substitution and solve a particularly well-conditioned single-group system. 


Of course, now we have to show that the transport operator can be inverted easily.

Quadrature
===============

First, we can now :ref:`show <sn_methods>` that by invoking a so-called discerete ordinates quadrature, where we approximate an angular integral as 

.. math:: 
    \int_{S^2} f(\Omega) d\Omega \approx \sum_k w_k f(\Omega_k),

leads to the approximation of the inverse of the :math:`P_N` transport operator as the inversion of :math:`K` spatial PDEs for a set of :math:`K` discrete angles :math:`\Omega_k`.

In other words:

.. math::
    \boldsymbol{\tau}_g^{-1} \approx \mathbf{M} \circ \mathbf{T}_{g}^{-1} \circ \mathbf{D}

This uses the following operators (the notation is slightly different than in the derivation; here we opt for something that is easier to understand):

.. math::
    \mathbf{M} : \mathcal{D} \to \mathcal{M} \text{ with } \mathbf{M}[\{z_{k}(\mathbf{r})\}] = \sum_k w_k z_k(\mathbf{r}) Y_{\lambda\mu}(\Omega_k)

which is the discrete to moment (very simmilar to the quadrature!) operator, mapping functions in the discrete angle space :math:`\mathcal{D}` to functions in the Legendre moment space :math:`\mathcal{M}`.,  
and the moment to discrete operator:

.. math::
    \mathbf{D} : \mathcal{M} \to \mathcal{D} \text{ with } \mathbf{D}[\{z_{\lambda\mu}(\mathbf{r})\}] = \sum_{lm} Y_{lm}(\Omega_k) z_{lm}(\mathbf{r})
    
Physically, we map the source in terms of Legendre moments to a source in each of the discrete angles (:math:`\mathbf{D}`). Then, we solve the spatial PDE for each angle
(:math:`\mathbf{T}_g^{-1}`), and then we map the result back to the Legendre moment space (:math:`\mathbf{M}`).

The transport operator inverse :math:`\boldsymbol{\tau}_g^{-1}` has now been converted from :math:`(L+1)^2` (with :math:`L` the order of the Legendre expansion) coupled PDEs to :math:`K` uncoupled PDEs. 
This already a significant simplification (of course, as long as the quadrature approximation is good...). 
Still, a single operator application thus requires the solution of :math:`K` spatial PDEs, which does not seem to be more efficient than just using the non-conditioned matrix (which is just a simple application of the gradient!)
However, in the next section, we will show that the spatial discretisation can be chosen such that the solution of these PDEs is extremely efficient. This so-called 'sweep' algorithms will not solve a 
a large system but only a sequence of small systems. Then, it is sufficiently efficient that the conditioning is worth it.

Furthermore, a bonus is that for all for spherical harmonics larger than the maximum scattering moment :math:`L_{\max}` (which is usually :math:`<5`), we do not even have to store or compute 
the :math:`\phi_{glm}` (with :math:`l > L_{\max}`) moments. Their submatrices just look like identity matrices in the reconditioned system. They can thus be safely removed, 
massively reducing the memory requirements of the system.

Operator structure
====================

Using the discrete ordinates quadrature, we now have the following structure:

.. math::
    \scriptsize
    \begin{bmatrix}
    \mathbf{I} - \mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{00} & -\mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{01} & \cdots & -\mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{0G}\\
    -\mathbf{M} \circ \mathbf{T}_1^{-1} \circ \mathbf{D} \circ \mathbf{S}_{10} & \mathbf{I} -\mathbf{M} \circ \mathbf{T}_1^{-1} \circ \mathbf{D}  \circ\mathbf{S}_{11} & \cdots & -\mathbf{M}  \circ\mathbf{T}_1^{-1}\circ\mathbf{D}\circ\mathbf{S}_{1G}\\
    \vdots & \vdots & \ddots & \vdots\\
    -\mathbf{M}\circ\mathbf{T}_G^{-1}\circ\mathbf{D}\circ\mathbf{S}_{G0} & -\mathbf{M}\circ\mathbf{T}_G^{-1}\circ\mathbf{D}\circ\mathbf{S}_{G1} & \cdots & 	\mathbf{I}-\mathbf{M}\circ\mathbf{T}_G^{-1}\circ\mathbf{D}\circ\mathbf{S}_{GG}
    \end{bmatrix}
    \cdot 
    \begin{bmatrix}
    \{\phi_{0,lm}\} \\ 
    \{\phi_{1,lm}\} \\ 
    \vdots \\ 
    \{\phi_{G,lm}\}
    \end{bmatrix} = 
    \begin{bmatrix}
    \mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \{Q_{0,lm}\} \\ 
    \mathbf{M} \circ \mathbf{T}_1^{-1} \circ \mathbf{D} \circ \{Q_{1,lm}\} \\ 
    \vdots \\ 
    \mathbf{M} \circ \mathbf{T}_G^{-1} \circ \mathbf{D} \circ \{Q_{G,lm}\}
    \end{bmatrix}


It may seem that there are :math:`K` \times G^2` operator inversions required, but in fact, due to the linearity of the operators, we only need :math:`K` \times G`!
We can just collapse the product of the operators on the left-hand side as (let us just look at the first row for simplicity):

.. math::
    \scriptsize    
    &\mathbf{I} - \mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{00}[\{\phi_{0,lm}\}] -\mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{01}[\{\phi_{1,lm}\}] -\cdots -\mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{0G}[\{\phi_{G,lm}\}] \\
    &=  \mathbf{I} -  \mathbf{M} \circ \mathbf{T}_0^{-1} \circ \mathbf{D}  \left[\sum_{g'}\mathbf{S}_{0g'}[\{\phi_{g',lm}\}]\right]

and the same applies to all other rows. Thus, per operator application, we only need to apply the transport operator inverse once per group.

Conclusion
=====================

In this section we have shown how a spherical harmonics expansion can be used to discretise the angular variable, how multiplying by the transport operator inverse
leads to a well-conditioned system, and how this transport operator inverse can be approximated by a quadrature. 

This results in an operator which requires for an application (not a *solve*), to solve :math:`K` \times G` spatial PDEs. The conditioning ensures that iterative methods converge 
quite quickly (within :math:`\mathcal{O}(10)` iterations per group usually), but still this implies appling this operator relatively many times.

Therefore, the efficiency of the scheme depends critically on the efficiency on on the solution of the spatial PDEs. 
In the next section we will show that it is possible to obtain a very efficient scheme.








