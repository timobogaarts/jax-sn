.. _sn_methods:

:math:`S_N` approximation
=======================================================

From :ref:`the section on PN methods <pn_methods>`, we have the following representation of the transport operator:

.. math::
    \boldsymbol{\tau}_g [\phi_{g,\lambda\mu}] = \int Y_{lm}(\Omega) \left[\Omega \cdot \nabla + \sigma_g(\mathbf{r})\right] \sum_{\lambda\mu} Y_{\lambda\mu}(\Omega) \phi_{g,\lambda\mu}(\mathbf{r}) \ d\Omega


If we define the expansion operator (remind that :math:`\mathcal{M}` is the space of legendre moment coefficients, and :math:`H^1(X)` is the function space on the domain :math:`X` with integrable first derivatives):

.. math::
    &\mathbf{E} : \mathcal{M} \otimes H^1(X) \to S^2 \otimes H^1(X) \\ 
    &\mathbf{E}[x = \{z_{\lambda\mu}\}] = \sum_{\lambda\mu} Y_{\lambda\mu}(\Omega) z_{\lambda\mu}(\mathbf{r})

the projection operator:

.. math::
    &\mathbf{P} : S^2 \otimes H^1(X) \to \mathcal{M} \otimes H^1(X) \\
    &\mathbf{P}[x] = \int_{S^2} Y_{lm}(\Omega) x(\mathbf{r}, \Omega) d\Omega

and the usual transport operator (as before):

.. math::
    &\mathcal{T}_g : S^2 \otimes H^1(X) \to S^2 \otimes H^1(X) \\
    &\mathcal{T}_g[x] = \Omega \cdot \nabla x + \sigma_g(\mathbf{r})x


this yields the total operator

.. math::
    &\boldsymbol{\tau}_g : \mathcal{M} \otimes H^1(X) \to \mathcal{M} \otimes H^1(X) \\
    &\boldsymbol{\tau}_g = \mathbf{P} \circ \mathcal{T}_g \circ \mathbf{E}

Then, we naturally:

.. math::
    \boldsymbol{\tau}_g^{-1} = \mathbf{E}^{-1} \circ \mathcal{T}_g^{-1} \circ \mathbf{P}^{-1}

Since the spherical harmonics are a basis on the unit sphere, the expansion and projection operators are invertible. 
Furthermore, they are eachothers inverse: :math:`\mathbf{E}^{-1} = \mathbf{P}` and :math:`\mathbf{P}^{-1} = \mathbf{E}`.
This means that the transport operator is invertible and its inverse is given by:

.. math::
    \boldsymbol{\tau}_g^{-1}[{z_{\lambda\mu}}] &= \mathbf{P} \circ \mathcal{T}_g^{-1} \circ \mathbf{E}\\
    &=  \int Y_{lm}(\Omega) \mathcal{T}_g^{-1}\left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega) z_{\lambda\mu}(\mathbf{r})\right] d\Omega


As long as we take the limit :math:`L\to \infty`, this is an exact representation of the transport operator. Nothing has been approximated yet.

Quadrature
============

Now, the quadrature is introduced:

.. math::
    \int_{S^2} f(\Omega) d\Omega \approx \sum_k w_k f(\Omega_k)

for some discrete angles :math:`\Omega_k` and weights :math:`w_k`.

Applying this to the above expression for the inverse of the transport operator, we have:

.. math::
    \boldsymbol{\tau}_g^{-1}[{z_{\lambda\mu}}] \approx \sum_k w_k Y_{lm}(\Omega_k)\left( \mathcal{T}_g^{-1}\left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega) z_{\lambda\mu}(\mathbf{r})\right]\right)(\Omega_k)

where we evalute the result of the application of the operator :math:`\mathcal{T}_g^{-1}` at the discrete angles :math:`\Omega_k` (not multiplication).

Now, notice that since the transport operator only acts on the specific angle itself (there is nothing coupling different angles! Normally, the scattering operator couples different angles, but we are now only looking at the 
transport operator): 

.. math::
    \mathcal{T}_g[ \psi] (\Omega_k)= \Omega_k \cdot \nabla \psi(\mathbf{r}, \Omega_k) + \sigma_g(\mathbf{r}) \psi(\mathbf{r}, \Omega_k)

so its inverse only requires the inversion of the discrete angle version:

.. math::
    \mathcal{T}_{gk} [\psi(\mathbf{r}, \Omega_k)]  = \Omega_k \cdot \nabla \psi(\mathbf{r}, \Omega_k) + \sigma_g(\mathbf{r}) \psi(\mathbf{r}, \Omega_k)

or in other words:

.. math::
    \left( \mathcal{T}_g^{-1}\left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega) z_{\lambda\mu}(\mathbf{r})\right]\right)(\Omega_k) &= \mathcal{T}_{gk}^{-1} \left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega_k) z_{\lambda\mu}(\mathbf{r})\right]\\
    &= [\Omega_k \cdot \nabla + \sigma_g(\mathbf{r})]^{-1} \left[\sum_{\lambda\mu} Y_{\lambda\mu}(\Omega_k) z_{\lambda\mu}(\mathbf{r})\right]

Note that the application of the inverse of the transport operator really just means solving the spatial PDE for that specific right-hand-side.

Now, we can define the following operators and spaces. :math:`\mathcal{D}` is the space of coefficients corresponding the discrete angles.
First the discrete-to-moment operator:

.. math::
    &\mathbf{M} : \mathcal{D} \to \mathcal{M} \\
    &\mathbf{M}[\{z_{k}\}] = \sum_k w_k z_k Y_{\lambda\mu}(\Omega_k)

and the moment-to-discrete operator:

.. math::
    &\mathbf{D} : \mathcal{M} \to \mathcal{D} \\
    &\mathbf{D}[\{z_{\lambda\mu}\}] = \sum_{\lambda\mu} Y_{\lambda\mu}(\Omega_k) z_{\lambda\mu}

Note that these do only take a set of coefficients and return a different set. To take a set of spatial functions, we can create the tensor product with the identity operator on the spatial function space:

.. math::
    &\mathbf{M} \otimes \mathbf{I}_{H^1(X)} : \mathcal{D} \otimes H^1(X) \to \mathcal{M} \otimes H^1(X) \\
    &(\mathbf{M} \otimes \mathbf{I}_{H^1(X)})[\{z_{k}(\mathbf{r})\}] = \sum_k w_k z_k(\mathbf{r}) Y_{\lambda\mu}(\Omega_k)

and 

.. math::
    &\mathbf{D} \otimes \mathbf{I}_{H^1(X)} : \mathcal{M} \otimes H^1(X) \to \mathcal{D} \otimes H^1(X) \\
    &(\mathbf{D} \otimes \mathbf{I}_{H^1(X)})[\{z_{\lambda\mu}(\mathbf{r})\}] = \sum_{\lambda\mu} Y_{\lambda\mu}(\Omega_k) z_{\lambda\mu}(\mathbf{r})

This might seem all a bit formal and abstract, but later, we will exploit these tensor product structures (especially when the spatial discretisation is introduced; then it avoids 
carrying many indices around...)

Finally, 
we can write the total discrete transport operator as:

.. math::
    \mathbf{T}_{H^1(X),g}^{-1} : \mathcal{D} \otimes H^1(X) \to \mathcal{D} \otimes H^1(X) \\
    \mathbf{T}_{H^1(X),g}^{-1} = \text{diag}\left( [\Omega_k \cdot \nabla + \sigma_g(\mathbf{r})]^{-1} \right)    

Using this notation, we can obtain the following expression for the approximation of the inverse of the transport operator using quadrature:

.. math::
    \boldsymbol{\tau}_g^{-1} \approx \left[\mathbf{M} \otimes \mathbf{I}_{H^1(X)}\right] \circ \mathbf{T}_{H^1(X),g}^{-1} \circ \left[\mathbf{D} \otimes \mathbf{I}_{H^1(X)}\right]

To gain a bit of intuitition, we can explicitly write the matrix representation (once more showing that the operator notation is a lot more compact...)

.. math::

    \scriptsize
    \begin{bmatrix}
    w_0 Y_{00}(\Omega_0) & w_1 Y_{00}(\Omega_1) & \cdots & w_K Y_{00}(\Omega_K)\\
    w_0 Y_{1{-1}}(\Omega_0) & w_1 Y_{1{-1}}(\Omega_1) & \cdots & w_K Y_{1{-1}}(\Omega_K)\\
    w_0 Y_{10}(\Omega_0) & w_1 Y_{10}(\Omega_1) & \cdots & w_K Y_{10}(\Omega_K)\\
    w_0 Y_{11}(\Omega_0) & w_1 Y_{11}(\Omega_1) & \cdots & w_K Y_{11}(\Omega_K)\\
    \vdots & \vdots & \ddots & \vdots \\    
    \end{bmatrix}\begin{bmatrix}
    [\Omega_0 \cdot \nabla + \sigma_g(\mathbf{r})]^{-1} & 0 & 0 & 0 & \cdots & 0\\
    0 & [\Omega_1 \cdot \nabla + \sigma_g(\mathbf{r})]^{-1} & 0 & 0 &\cdots & 0\\
    0 & 0 & [\Omega_2 \cdot \nabla + \sigma_g(\mathbf{r})]^{-1} & 0 & \cdots& 0\\
    0 & 0 & 0 & [\Omega_3 \cdot \nabla + \sigma_g(\mathbf{r})]^{-1} & \cdots& 0\\
    &\vdots &\vdots & \vdots& \ddots & \vdots \\
    0 & 0 & 0 & 0 & \cdots & [\Omega_K \cdot \nabla + \sigma_g(\mathbf{r})  ]^{-1}
    \end{bmatrix}\begin{bmatrix}
    Y_{00}(\Omega_0) & Y_{1{-1}}(\Omega_0) & Y_{10}(\Omega_0) & Y_{11}(\Omega_0) & \cdots \\ 
    Y_{00}(\Omega_1) & Y_{1{-1}}(\Omega_1) & Y_{10}(\Omega_1) & Y_{11}(\Omega_1) & \cdots \\
    Y_{00}(\Omega_2) & Y_{1{-1}}(\Omega_2) & Y_{10}(\Omega_2) & Y_{11}(\Omega_2) & \cdots \\
    Y_{00}(\Omega_3) & Y_{1{-1}}(\Omega_3) & Y_{10}(\Omega_3) & Y_{11}(\Omega_3) & \cdots \\
    \vdots & \vdots & \vdots & \vdots & \ddots
    \end{bmatrix}\cdot
    \begin{bmatrix}
    \phi_{g00}(\mathbf{r})\\
    \phi_{g1{-1}}(\mathbf{r})\\
    \phi_{g10}(\mathbf{r})\\
    \phi_{g11}(\mathbf{r})\\
    \vdots
    \end{bmatrix}


We will now simply write :math:`\mathbf{M}\otimes \mathbf{I}_{H^1(X)}` as :math:`\mathbf{M}_{H^1(X)}` and :math:`\mathbf{D}\otimes \mathbf{I}_{H^1(X)}` as :math:`\mathbf{D}_{H^1(X)}`.
Then, the total transport operator is given by:

.. math::
    \boldsymbol{\tau}_g^{-1} \approx \mathbf{M}_{H^1(X)} \circ \mathbf{T}_{H^1(X), g}^{-1} \circ \mathbf{D}_{H^1(X)}



    