Downscattering
------------------

The fusion property of only-downscattering can be exploited to yield a more efficient solver. The only operator coupling different energy groups is the scattering operator.

Defining the group-to-group scattering operator and the group gather scatter operator as 

.. math::
    &\mathbf{S}_{g\tilde{g}}[a_{\ell e \dots} ] &&= \sigma^s_{\ell g \tilde{g} e} a_{\ell e \dots} \\ 
    &\mathbf{S}_{g}[a_{\tilde{g} \ell e \dots }] &&= \sum_{\tilde{g}}\mathbf{S}_{g\tilde{g}}[a_{\tilde{g} \ell e \dots}]

Then, the total system can also be written in a group matrix form:

.. math::
    \begin{bmatrix}
        \mathbf{I} - \mathbf{M}\mathbf{T}_1^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{11} & -\mathbf{M}\mathbf{T}_1^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{12}& \cdots & -\mathbf{M}\mathbf{T}_1^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{1G} \\
        -\mathbf{M}\mathbf{T}_2^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{21}& \mathbf{I} - \mathbf{M}\mathbf{T}_2^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{22} & \cdots & -\mathbf{M}\mathbf{T}_2^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{2G} \\
        \vdots & \vdots & \ddots & \vdots \\
        -\mathbf{M}\mathbf{T}_G^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{G1} & -\mathbf{M}\mathbf{T}_G^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{G2} & \cdots & \mathbf{I} - \mathbf{M}\mathbf{T}_G^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{GG}
    \end{bmatrix}
    \begin{bmatrix}
        \phi_{1\ell ei} \\ \phi_{2\ell ei} \\ \vdots \\ \phi_{G \ell ei}
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mathbf{M}\mathbf{T}_1^{-1}\mathbf{H}[q_{1kei}] \\ \mathbf{M}\mathbf{T}_2^{-1}\mathbf{H}[q_{2kei}] \\ \vdots \\ \mathbf{M}\mathbf{T}_G^{-1}\mathbf{H}[q_{Gkei}]
    \end{bmatrix}

If there's no upscattering, this matrix is lower triangular, which can be solved by forward substitution. 

.. container:: algorithm-box

   **Downscattering algorithm**

   Initialize:

   :math:`\phi_{g\ell ei} \gets 0`

   :loop:`For` :math:`g = 1, \dots, G`

      Compute group right-hand side:

      :math:`b_{\ell ei} =
      \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}
      \left[
      \mathbf{D}\mathbf{S}_g[\phi_{\tilde{g}\ell ei}]
      +
      [q_{gkei}]
      \right]`

      Iteratively solve:

      :math:`(
      \mathbf{I}
      -
      \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{gg}
      )[x_{\ell ei}]
      =
      b_{\ell ei}`

      Update:

      :math:`\phi_{g\ell ei} \gets x_{\ell ei}`

Also for the adjoint solve we  should be able to exploit the same property. Let us transpose the entire system, noting that 

.. math::
    \mathbf{S}^T_{g\tilde{g}} = \mathbf{S}_{g\tilde{g}}

since it just acts as elementwise multiplication for fixed :math:`g,\tilde{g}`. Transposing the block matrix yields the system of a given right-hand-side :math:`b^\dagger_{glei}`

.. math::
    \begin{bmatrix}
        \mathbf{I} - \mathbf{S}_{11}\mathbf{D}^T\mathbf{H}\mathbf{T}_1^{-T}\mathbf{M}^T & -\mathbf{S}_{21}\mathbf{D}^T\mathbf{H}\mathbf{T}_2^{-T}\mathbf{M}^T & \cdots & -\mathbf{S}_{G1}\mathbf{D}^T\mathbf{H}\mathbf{T}_G^{-T}\mathbf{M}^T \\
        -\mathbf{S}_{12}\mathbf{D}^T\mathbf{H}\mathbf{T}_1^{-T}\mathbf{M}^T & \mathbf{I} - \mathbf{S}_{22}\mathbf{D}^T\mathbf{H}\mathbf{T}_2^{-T}\mathbf{M}^T & \cdots & -\mathbf{S}_{G2}\mathbf{D}^T\mathbf{H}\mathbf{T}_G^{-T}\mathbf{M}^T \\
        \vdots & \vdots & \ddots & \vdots \\
        -\mathbf{S}_{1G}\mathbf{D}^T\mathbf{H}\mathbf{T}_1^{-T}\mathbf{M}^T & -\mathbf{S}_{2G}\mathbf{D}^T\mathbf{H}\mathbf{T}_2^{-T}\mathbf{M}^T & \cdots & \mathbf{I} - \mathbf{S}_{GG}\mathbf{D}^T\mathbf{H}\mathbf{T}_G^{-T}\mathbf{M}^T
    \end{bmatrix}
    \begin{bmatrix}
        \phi^\dagger_{1\ell ei} \\ \phi^\dagger_{2\ell ei} \\ \vdots \\ \phi^\dagger_{G \ell ei}
    \end{bmatrix}
    =
    \begin{bmatrix}
        b^\dagger_{1\ell ei} \\ b^\dagger_{2\ell ei} \\ \vdots \\ b^\dagger_{G \ell ei}
    \end{bmatrix}

which is upper triangular if all :math:`\mathbf{S}_{g\tilde{g}}` are zero for :math:`g < \tilde{g}`. This can be solved by backward substitution. 
One thing to note is that for a single row, *all* transport operators for all groups are present and we cannot solve it with just one sweep. Instead, we can define the 
transpose gathering operator:

.. math::
    \mathbf{S}^T_{g}[a_{\tilde{g}\ell e \dots }] = \sum_{\tilde{g}}\mathbf{S}_{\tilde{g} g}[a_{\tilde{g}\ell e \dots}]
and the transposed downscatter system can be solved group-by-group using an intermediate quantity :math:`\zeta`, as shown below.

.. container:: algorithm-box

   **Transposed Downscatter solution**

   Initialize:

   :math:`\phi^\dagger_{g\ell ei} \gets 0`

   Set intermediate:

   :math:`\zeta_{g\ell ei} \gets 0`

   :loop:`For` :math:`g = G, \dots, 1`

      Compute group right-hand side:

      :math:`b_{\ell ei} =
      \mathbf{S}_g^T[\zeta_{\tilde{g}\ell ei}]
      +
      b^\dagger_{g\ell ei}`

      Iteratively solve:

      :math:`(
      \mathbf{I}
      -
      \mathbf{S}_{gg}
      \mathbf{D}^T
      \mathbf{H}
      \mathbf{T}_g^{-T}
      \mathbf{M}^T
      )[x_{\ell ei}]
      =
      b_{\ell ei}`

      Update:

      :math:`\phi^\dagger_{g\ell ei} \gets x_{\ell ei}`

      Update intermediate:

      :math:`\zeta_{g\ell ei}
      \gets
      \mathbf{D}^T
      \mathbf{H}
      \mathbf{T}_g^{-T}
      \mathbf{M}^T
      [x_{\ell ei}]`


    