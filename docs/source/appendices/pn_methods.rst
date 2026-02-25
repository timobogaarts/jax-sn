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
It sufficies to say that they can be computed analytically and are non-zero only for :math:`\lambda = l \pm 1` and :math:`\mu = m, m \pm 1, -m , -m \pm 1`.

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
    \tiny
	\begin{bmatrix}
	\sigma^t_g & C_{00,1-1} \cdot \nabla & C_{00,10} \cdot \nabla & C_{00,11} \cdot \nabla & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \cdots \\
	C_{1-1,00} \cdot \nabla & \sigma^t_g & 0 & 0 & C_{1-1,2-2} \cdot \nabla & C_{1-1,2-1} \cdot \nabla & C_{1-1,20} \cdot \nabla & C_{1-1,21} \cdot \nabla & C_{1-1,22} \cdot \nabla & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \cdots \\
	C_{10,00} \cdot \nabla & 0 & \sigma^t_g & 0 & 0 & C_{10,2-1} \cdot \nabla & C_{10,20} \cdot \nabla & C_{10,21} \cdot \nabla & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \cdots \\
	C_{11,00} \cdot \nabla & 0 & 0 & \sigma^t_g & C_{11,2-2} \cdot \nabla & C_{11,2-1} \cdot \nabla & C_{11,20} \cdot \nabla & C_{11,21} \cdot \nabla & C_{11,22} \cdot \nabla & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \cdots \\
	0 & C_{2-2,1-1} \cdot \nabla & 0 & C_{2-2,11} \cdot \nabla & \sigma^t_g & 0 & 0 & 0 & 0 & C_{2-2,3-3} \cdot \nabla & C_{2-2,3-2} \cdot \nabla & C_{2-2,3-1} \cdot \nabla & 0 & C_{2-2,31} \cdot \nabla & C_{2-2,32} \cdot \nabla & C_{2-2,33} \cdot \nabla & \cdots \\
	0 & C_{2-1,1-1} \cdot \nabla & C_{2-1,10} \cdot \nabla & C_{2-1,11} \cdot \nabla & 0 & \sigma^t_g & 0 & 0 & 0 & 0 & C_{2-1,3-2} \cdot \nabla & C_{2-1,3-1} \cdot \nabla & C_{2-1,30} \cdot \nabla & C_{2-1,31} \cdot \nabla & C_{2-1,32} \cdot \nabla & 0 & \cdots \\
	0 & C_{20,1-1} \cdot \nabla & C_{20,10} \cdot \nabla & C_{20,11} \cdot \nabla & 0 & 0 & \sigma^t_g & 0 & 0 & 0 & 0 & C_{20,3-1} \cdot \nabla & C_{20,30} \cdot \nabla & C_{20,31} \cdot \nabla & 0 & 0 & \cdots \\
	0 & C_{21,1-1} \cdot \nabla & C_{21,10} \cdot \nabla & C_{21,11} \cdot \nabla & 0 & 0 & 0 & \sigma^t_g & 0 & 0 & C_{21,3-2} \cdot \nabla & C_{21,3-1} \cdot \nabla & C_{21,30} \cdot \nabla & C_{21,31} \cdot \nabla & C_{21,32} \cdot \nabla & 0 & \cdots \\
	0 & C_{22,1-1} \cdot \nabla & 0 & C_{22,11} \cdot \nabla & 0 & 0 & 0 & 0 & \sigma^t_g & C_{22,3-3} \cdot \nabla & C_{22,3-2} \cdot \nabla & C_{22,3-1} \cdot \nabla & 0 & C_{22,31} \cdot \nabla & C_{22,32} \cdot \nabla & C_{22,33} \cdot \nabla & \cdots \\
	0 & 0 & 0 & 0 & C_{3-3,2-2} \cdot \nabla & 0 & 0 & 0 & C_{3-3,22} \cdot \nabla & \sigma^t_g & 0 & 0 & 0 & 0 & 0 & 0 & \cdots \\
	0 & 0 & 0 & 0 & C_{3-2,2-2} \cdot \nabla & C_{3-2,2-1} \cdot \nabla & 0 & C_{3-2,21} \cdot \nabla & C_{3-2,22} \cdot \nabla & 0 & \sigma^t_g & 0 & 0 & 0 & 0 & 0 & \cdots \\
	0 & 0 & 0 & 0 & C_{3-1,2-2} \cdot \nabla & C_{3-1,2-1} \cdot \nabla & C_{3-1,20} \cdot \nabla & C_{3-1,21} \cdot \nabla & C_{3-1,22} \cdot \nabla & 0 & 0 & \sigma^t_g & 0 & 0 & 0 & 0 & \cdots \\
	0 & 0 & 0 & 0 & 0 & C_{30,2-1} \cdot \nabla & C_{30,20} \cdot \nabla & C_{30,21} \cdot \nabla & 0 & 0 & 0 & 0 & \sigma^t_g & 0 & 0 & 0 & \cdots \\
	0 & 0 & 0 & 0 & C_{31,2-2} \cdot \nabla & C_{31,2-1} \cdot \nabla & C_{31,20} \cdot \nabla & C_{31,21} \cdot \nabla & C_{31,22} \cdot \nabla & 0 & 0 & 0 & 0 & \sigma^t_g & 0 & 0 & \cdots \\
	0 & 0 & 0 & 0 & C_{32,2-2} \cdot \nabla & C_{32,2-1} \cdot \nabla & 0 & C_{32,21} \cdot \nabla & C_{32,22} \cdot \nabla & 0 & 0 & 0 & 0 & 0 & \sigma^t_g & 0 & \cdots \\
	0 & 0 & 0 & 0 & C_{33,2-2} \cdot \nabla & 0 & 0 & 0 & C_{33,22} \cdot \nabla & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^t_g & \cdots \\
	 \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots \\
	\end{bmatrix}
where we can clearly see the :math:`l \pm 1` and :math:`m, m \pm 1, -m , -m \pm 1` selection rules.

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



Computation of coefficients
----------------------------
This section can be skipped if you are only interested in the discrete-ordinates version.

We have 

.. math::
    \mathbf{C}_{lm, \lambda\mu} = \int_{S^2} Y_{lm}(\Omega) Y_{\lambda\mu}(\Omega)  \Omega d\Omega

From the Gaunt coefficients, we have 

.. math::
    \small
    \int Y_{l_1}^{m_1}(\Omega) Y_{l_2}^{ m_2}(\Omega) Y_{l_3}^{m_3}(\Omega) d\Omega = \sqrt{\frac{(2l_1 + 1)(2l_2 + 1)(2l_3 + 1)}{4\pi}} \begin{pmatrix}
    l_1 & l_2 & l_3 \\
    0 & 0 & 0
    \end{pmatrix} \begin{pmatrix}
    l_1 & l_2 & l_3 \\
    m_1 & m_2 & m_3
    \end{pmatrix}

where the terms in brackets are the Wigner 3-j symbols and we use :math:`Y_{l}^m` to denote the *complex* spherical harmonics. Explicitly, they are defined as:

.. math::    
    \small\begin{pmatrix}
    l_1 & l_2 & l_3 \\
    m_1 & m_2 & m_3
    \end{pmatrix} & \small = \delta(m_1 + m_2 + m_3) \sqrt{\frac{(l_1 + l_2 - l_3)!(l_1 - l_2 + l_3)!(-l_1 + l_2 + l_3)!}{(l_1 + l_2 + l_3 + 1)!}} \times \\
    &\small \sqrt{(l_1 + m_1)!(l_1 - m_1)!(l_2 + m_2)!(l_2 - m_2)!(l_3 + m_3)!(l_3 - m_3)!}\times \\
    &\small \sum_{k=K}^{N} \frac{(-1)^k}{k!(l_1 + l_2 - l_3 - k)!(l_1 - m_1 - k)!(l_2 + m_2 - k)!(l_3 - l_2 + m_1 + z)!(l_3 - l_1 - m_2 + k)!}

where :math:`K = \max(0, l_2 - l_3 - m_1, l_1 - l_3 + m_2)` and :math:`N = \min(l_1 + l_2 - l_3, l_1 - m_1, l_2 + m_2)`.

These Wigner 3-j symbol is zero unless the following selection rules are satisfied:

.. math::
    m_1 + m_2 + m_3 = 0 \\
    m_i \in {-l_i, -l_i + 1, \cdots, l_i - 1, l_i} \quad \forall i \in \{1,2,3\}\\
    |l_1 - l_2| \leq l_3 \leq l_1 + l_2

Now, we have to note that the real spherical harmonics are related to the complex spherical harmonics as (see :ref:`Spherical Harmonics <spherical_harmonics>` for more details):

.. math::
    Y_{lm}(\theta, \phi)= \begin{cases} \sqrt{2} \mathcal{I}(Y_l^{|m|}(\theta,\phi)) & m < 0 \\ Y_l^0(\theta, \phi) & m =0 \\ \sqrt{2} \mathcal{R}(Y_l^m(\theta, \phi)) & m >0 \end{cases}

or equivalently

.. math::
    Y_{lm}(\theta,\phi) = \begin{cases} \frac{-i}{\sqrt{2}}(Y_l^{|m|}(\theta,\phi) - Y_l^{-|m|}(\theta,\phi)) & m < 0 \\ Y_l^0(\theta, \phi) & m =0 \\ \frac{1}{\sqrt{2}}(Y_l^m(\theta, \phi) + Y_l^{-m}(\theta, \phi)) & m >0 \end{cases}

We want to compute the particular integral as defined above, which is in terms of real spherical harmonics. To use the Gaunt coefficients defined for complex coefficients, we have to transform. 
This is just a linear transformation:

.. math::
    Y_{lm} = \sum_{m'=-l}^l U^{(l)}_{mm'} Y_l^{m'}

where explicitly, we have 

.. math::
    U^{(l)}_{mm'} = \begin{cases} \frac{-i}{\sqrt{2}}(\delta_{m',|m|} - \delta_{m',-|m|}) & m < 0 \\ \delta_{m',0} & m =0 \\ \frac{1}{\sqrt{2}}(\delta_{m',m} + \delta_{m',-m}) & m >0 \end{cases}

where :math:`U^{(l)}_{mm'}` is the matrix that transforms the complex spherical harmonics.

Filling in the integral equation yields:

.. math::
    \int_{S^2} Y_{lm}(\Omega) Y_{\lambda\mu}(\Omega)  \Omega d\Omega = &\sum_{m'=-l}^l \sum_{\mu'=-\lambda}^\lambda U^{(l)}_{mm'} U^{(\lambda)}_{\mu\mu'} \int_{S^2} Y_l^{m'}(\Omega) Y_\lambda^{\mu'}(\Omega)  \Omega d\Omega\\
    
This is already quite close to the Gaunt coefficient definition. Now we just have to express the vector :math:`\Omega` in terms of spherical harmonics. This of course 
requires the introduction of a basis for the :math:`\Omega` components. We just use cartesian coordinates.

We can use :

.. math::
    \Omega_x = \sqrt{\frac{2\pi}{3}}(Y_1^{-1} - Y_1^1) \\
    \Omega_y = i\sqrt{\frac{2\pi}{3}}(Y_1^{-1} + Y_1^1) \\
    \Omega_z = \sqrt{\frac{4\pi}{3}}Y_1^0

or equivalently:

.. math::
    \Omega_q =  \sum_{\nu=-1}^1 \alpha_{q\nu} Y_1^\nu

where :math:`q \in \{x,y,z\}` and :math:`\alpha_{qm}` are the coefficients of the expansion.

Filling this into the integral equation yields:

.. math::
    C_{lm,\lambda\mu, q} &\sum_{m'=-l}^l \sum_{\mu'=-\lambda}^\lambda U^{(l)}_{mm'} U^{(\lambda)}_{\mu\mu'} \int_{S^2} Y_l^{m'}(\Omega) Y_\lambda^{\mu'}(\Omega)  \Omega_q d\Omega\\\ 
    &=\sum_{\nu=-1}^1\sum_{m'=-l}^l \sum_{\mu'=-\lambda}^\lambda U^{(l)}_{mm'} U^{(\lambda)}_{\mu\mu'}  \alpha_{q\nu} \int_{S^2} Y_l^{m'}(\Omega) Y_\lambda^{\mu'}(\Omega)  Y_1^{\nu}(\Omega) d\Omega\\

Using the 3j-symbols

.. math::
    C_{lm,\lambda\mu, q} = &\sum_{\nu=-1}^1\sum_{m'=-l}^l \sum_{\mu'=-\lambda}^\lambda U^{(l)}_{mm'} U^{(\lambda)}_{\mu\mu'}  \alpha_{q\nu} \sqrt{\frac{(2l + 1)(2\lambda + 1)(3)}{4\pi}} \begin{pmatrix}
    l & \lambda & 1 \\
    0 & 0 & 0
    \end{pmatrix} \begin{pmatrix}
    l & \lambda & 1 \\
    m' & \mu' & \nu
    \end{pmatrix}\\ 
    &= \sum_{\nu=-1}^1\sum_{m'=-l}^l \sum_{\mu'=-\lambda}^\lambda U^{(l)}_{mm'} U^{(\lambda)}_{\mu\mu'}  \alpha_{q\nu} \mathcal{G}_{l\lambda}^{m' \mu' \nu}

The 3j-symbols are zero unless :math:`l = \lambda \pm 1` (fill in :math:`l_3=1` for the selection rule above). That proves the assertion above that the
coefficients are zero unless :math:`\lambda = l \pm 1`.

Now, we note that the transformation matrix :math:`U^{(l)}_{mm'}` is also zero unless :math:`m' = m,-m` (in the case of m=0, this already picks out the correct value anyway). 
So the only thing contributing to :math:`C_{lm,\lambda\mu,q}` is the :math:`\mathcal{G}_{l\lambda}^{m' \mu' \nu}` term where :math:`m' = \pm m `.

The 3j-symbols are zero unless :math:`m' + \mu' + \nu = 0`. We know that :math:`\nu = -1,0,1`, so this means the nonzero 
entries of :math:`\mathcal{G}_{l\lambda}^{m'\mu'\nu}` are where :math:`\mu' =  - m ' +(-1,0,1) \equiv \pm m + (-1,0,1)`.

Again, :math:`U^{(\lambda)}_{\mu\mu'}` is zero unless :math:`\mu = \pm \mu'`. 
Since the only nonzero terms are :math:` \mu' = \pm  m + (-1,0,1)`, this means the nonzero entries of :math:`U^{(\lambda)}_{\mu\mu'}` are only for :math:`\mu = \pm m + (-1,0,1)`, as we mentioned before.


