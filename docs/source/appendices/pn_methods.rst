.. _pn_methods:

:math:`P_N` methods
=======================================================

We start from the multigroup equations with the angular flux expanded in terms of the legendre moments:

.. math::
    \mathcal{T}_{g}\left[\sum_{lm} Y_{lm}(\Omega)\phi_{glm}(\mathbf{r})\right] = \sum_{g'}\sum_{lm} Y_{lm}(\Omega)\sigma_{lm,gg'}\phi_{g'lm}(\mathbf{r}) + Q_g

Now, we rename the dummy indices :math:`lm\to \lambda\mu` and multiply both sides by :math:`Y_{lm}(\Omega)` and integrate over the unit sphere and require that this holds for all :math:`l,m`:

.. math::
    \int_{S^2} Y_{lm}(\Omega) \mathcal{T}_{g}\left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega)\phi_{g\lambda\mu}(\mathbf{r})\right] d\Omega = &\sum_{g'}\sum_{\lambda\mu} \int_{S^2} Y_{lm}(\Omega) Y_{\lambda\mu}(\Omega)\sigma_{\lambda\mu,gg'}\phi_{g'\lambda\mu}(\mathbf{r}) d\Omega \\
     &+ \int_{S^2} Y_{lm}(\Omega) Q_g d\Omega \quad \forall g, l,m

By the orthogonality of the real spherical harmonics, we have:

.. math::
    \int_{S^2} Y_{lm}(\Omega) \mathcal{T}_{g}\left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega)\phi_{g\lambda\mu}(\mathbf{r})\right] d\Omega = \sum_{g'} \sigma_{lm ,gg'}\phi_{g'lm}(\mathbf{r}) + Q_{glm}\quad \forall g, l,m

We see that only the transport operator couples the Legendre moments, as the scattering operator is just an element-wise multiplication in the Legendre moment space. We can write the transport operator as:

.. math::
    \int_{S^2} Y_{lm}(\Omega) \mathcal{T}_{g}\left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega)\phi_{g\lambda\mu}(\mathbf{r})\right] d\Omega = &\int_{S^2} Y_{lm}(\Omega) \Omega \cdot \nabla \left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega)\phi_{g\lambda\mu}(\mathbf{r})\right] d\Omega\\
    &+\int_{S^2} Y_{lm}(\Omega) \sigma^t_g(\mathbf{r}) \left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega)\phi_{g\lambda\mu}(\mathbf{r})\right] d\Omega

By linearity and orthogonality of the real spherical harmonics, this is equal to:

.. math::
    \int_{S^2} Y_{lm}(\Omega) Y_{\lambda\mu}(\Omega)  \Omega d\Omega \cdot \nabla \phi_{g\lambda\mu}(\mathbf{r}) + \sigma^t_g(\mathbf{r}) \phi_{glm}(\mathbf{r})

The right term is similar to the scattering term and does not couple the Legendre moments. 

The left term integral does not depend on the spatial variable, and can be precomputed. We denote this as:

.. math::
    \int_{S^2} Y_{lm}(\Omega) Y_{\lambda\mu}(\Omega)  \Omega d\Omega = \mathbf{C}_{lm, \lambda\mu}

These vectors are related to the `Gaunt Coefficients <https://en.wikipedia.org/wiki/3-j_symbol#Relation_to_spherical_harmonics;_Gaunt_coefficients>`__. 
It sufficies to say that they can be computed analytically and are non-zero only for :math:`\lambda = l \pm 1` and :math:`\mu = m, m \pm 1`.

Now, the total :math:`P_N` equations look like:

.. math::
    \sum_{\lambda\mu} \mathbf{C}_{lm, \lambda\mu} \cdot \nabla \phi_{g\lambda\mu}(\mathbf{r}) + \sigma^t_g(\mathbf{r}) \phi_{glm}(\mathbf{r}) = \sum_{g'} \sigma_{lm ,gg'}\phi_{g'lm}(\mathbf{r}) + Q_{glm}\quad \forall g, l,m

Let's now try to make sense of this structure for a bit.

Transport operator 
-------------------

Let's first look at the action of the left-hand side for a fixed :math:`g`.

.. math::
    \scriptsize
    \begin{bmatrix}
    C_{00,00} \cdot \nabla + \sigma^t_g(\mathbf{r}) & C_{00,1-1} \cdot \nabla & C_{00,10} \cdot \nabla & C_{00,11} \cdot \nabla & \cdots & C_{00,LL} \cdot \nabla\\
    C_{1-1,00} \cdot \nabla & C_{1-1,1-1} \cdot \nabla + \sigma^t_g(\mathbf{r}) & C_{1-1,10} \cdot \nabla & C_{1-1,11} \cdot \nabla & \cdots & C_{1-1,LL} \cdot \nabla\\
    C_{10,00} \cdot \nabla & C_{10,1-1} \cdot \nabla & C_{10,10} \cdot \nabla + \sigma^t_g(\mathbf{r}) & C_{10,11} \cdot \nabla & \cdots & C_{10,LL} \cdot \nabla\\
    C_{11,00} \cdot \nabla & C_{11,1-1} \cdot \nabla & C_{11,10} \cdot \nabla & C_{11,11} \cdot \nabla + \sigma^t_g(\mathbf{r}) & \cdots & C_{11,LL} \cdot \nabla\\
    \vdots & \vdots & \vdots & \vdots & \ddots & \vdots\\
    C_{LL,00} \cdot \nabla & C_{LL,1-1} \cdot \nabla & C_{LL,10} \cdot \nabla & C_{LL,11} \cdot \nabla & \cdots & C_{LL,LL} \cdot \nabla + \sigma^t_g(\mathbf{r})
    \end{bmatrix}\cdot  \begin{bmatrix}
    \phi_{g00}(\mathbf{r}) \\
    \phi_{g1-1}(\mathbf{r}) \\
    \phi_{g10}(\mathbf{r}) \\
    \phi_{g11}(\mathbf{r}) \\
    \vdots\\
    \phi_{gLL}(\mathbf{r})
    \end{bmatrix}

we can use the selection rule of the :math:`\mathbf{C}_{lm, \lambda\mu}` coefficients to simplify this structure to:

.. math::
   
   \scriptsize
   \begin{bmatrix}
   \sigma^t_g &
   \mathbf{C}_{00,1{-1}}\cdot\nabla &
   \mathbf{C}_{00,10}\cdot\nabla &
   \mathbf{C}_{00,11}\cdot\nabla &
   0 &
   0 &
   0 &
   0 &
   \cdots \\
   \mathbf{C}_{1{-1},00}\cdot\nabla &
   \sigma^t_g &
   0 &
   \mathbf{C}_{1{-1},11}\cdot\nabla &
   \mathbf{C}_{1{-1},2{-2}}\cdot\nabla &
   \mathbf{C}_{1{-1},2{-1}}\cdot\nabla &
   \mathbf{C}_{1{-1},20}\cdot\nabla &
   0 &
   \cdots \\
   \mathbf{C}_{10,00}\cdot\nabla &
   0 &
   \sigma^t_g &
   0 &
   \mathbf{C}_{10,2{-1}}\cdot\nabla &
   \mathbf{C}_{10,20}\cdot\nabla &
   \mathbf{C}_{10,21}\cdot\nabla &
   0 &
   \cdots \\
   \mathbf{C}_{11,00}\cdot\nabla &
   \mathbf{C}_{11,1{-1}}\cdot\nabla &
   0 &
   \sigma^t_g &
   0 &
   \mathbf{C}_{11,21}\cdot\nabla &
   \mathbf{C}_{11,22}\cdot\nabla &
   0 &
   \cdots \\
   0 &
   \mathbf{C}_{2{-2},1{-1}}\cdot\nabla &
   \mathbf{C}_{2{-2},10}\cdot\nabla &
   0 &
   \sigma^t_g &
   0 &
   \mathbf{C}_{2{-2},31}\cdot\nabla &
   0 &
   \cdots \\
   0 &
   \mathbf{C}_{2{-1},1{-1}}\cdot\nabla &
   \mathbf{C}_{2{-1},10}\cdot\nabla &
   \mathbf{C}_{2{-1},11}\cdot\nabla &
   0 &
   \sigma^t_g &
   0 &
   \mathbf{C}_{2{-1},32}\cdot\nabla &
   \cdots \\
   0 &
   0 &
   \mathbf{C}_{20,10}\cdot\nabla &
   \mathbf{C}_{20,11}\cdot\nabla &
   \mathbf{C}_{20,2{-1}}\cdot\nabla &
   0 &
   \sigma^t_g &
   \mathbf{C}_{20,33}\cdot\nabla &
   \cdots \\
   0 &
   0 &
   0 &
   \mathbf{C}_{21,11}\cdot\nabla &
   0 &
   \mathbf{C}_{21,2{-1}}\cdot\nabla &
   \mathbf{C}_{21,20}\cdot\nabla &
   \sigma^t_g &
   \cdots \\
   \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
   \end{bmatrix}
   \cdot
   \begin{bmatrix}
   \phi_{g00}\\
   \phi_{g1{-1}}\\
   \phi_{g10}\\
   \phi_{g11}\\
   \phi_{g2{-2}}\\
   \phi_{g2{-1}}\\
   \phi_{g20}\\
   \phi_{g21}\\
   \vdots
   \end{bmatrix}

Let's now simply define :math:`\boldsymbol{\tau}_g` as the left-hand-side operator, which operates on a set of spatially-varying Legendre moments :math:`\{\phi_{g,lm}\}` 
defined on a domain :math:`X \subseteq \mathbb{R}^3` and produces another set of spatially-varying Legendre moments defined on the same domain. In other words, we can write the transport operator as a linear operator between function spaces as:

.. math::
    \boldsymbol{\tau}_g : \mathcal{M} \times H^1(X) \to \mathcal{M} \times H^1(X)

where the action is defined as the above matrix-vector product (:math:`H^1(X)` is the standard Sobolev space of square-integrable functions with square-integrable first derivatives, or roughly speaking, the set of functions on the domain :math:`X` which have finite derivatives as required by the operator).

Scattering operator
--------------------

The scattering operator is much simpler, as it does not couple the Legendre moments. Let us look at a fixed :math:`g` and :math:`g'`.It is just an element-wise multiplication in the Legendre moment space:    

.. math::
    \begin{bmatrix}
    \sigma_{00,gg'} & 0 & 0 & 0 & \cdots & 0\\
    0 & \sigma_{1{-1},gg'} & 0 & 0 &\cdots & 0\\
    0 & 0 & \sigma_{10,gg'} & 0 & \cdots& 0\\
    0 & 0 & 0 & \sigma_{11,gg'} & \cdots& 0\\
    &\vdots &\vdots & \vdots& \ddots & \vdots \\
    0 & 0 & 0 & 0 & \cdots & \sigma_{LL,gg'}
    \end{bmatrix} \cdot \begin{bmatrix}
    \phi_{g'00}(\mathbf{r}) \\
    \phi_{g'1{-1}}(\mathbf{r}) \\
    \phi_{g'10}(\mathbf{r}) \\
    \phi_{g'11}(\mathbf{r}) \\
    \vdots \\ 
    \phi_{g'LL}(\mathbf{r})
     \end{bmatrix}

Let's now define this to be the operator :math:`\mathbf{S}_{gg'}`, that maps a set of spatially-varying Legendre moments :math:`\{\phi_{g',lm}\}` to another set of spatially-varying Legendre moments :math:`\{\sigma_{lm,gg'} \phi_{g',lm}\}`.

In other words, we can write the scattering operator as a linear operator between function spaces as:

.. math::
    \mathbf{S}_{gg'} : \mathcal{M} \times H^1(X) \to \mathcal{M} \times H^1(X)
    
where the action is defined as the above matrix-vector, or equivalently: as 

.. math::
    \mathbf{S}_{gg'} =  \mathbf{\Sigma}_{gg'} \otimes \mathbf{I}_{H^1(X)}

where :math:`\mathbf{\Sigma}_{gg'}` is the diagonal matrix of the scattering cross sections in the Legendre moment space, and :math:`\mathbf{I}_{H^1(X)}` is the identity operator on the function space :math:`H^1(X)`.

Total equation
------------------

We can start to see why we prefer an operator viewpoint; the number of indices and structure starts to become rather complex, and this was just for one single energy group!

The total equation then looks like

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

This is of course exactly the same as the multigroup equation  :eq:`eq:multigroup`. However, we now have explicit representations for the operators acting on a discrete set of 
spatially varying moments instead of on 5D functions.




