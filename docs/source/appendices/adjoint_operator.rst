.. _appendix_adjoint:

Adjoint Operator
==================

.. warning::

    This section is for the adjoint of the *continuous* neutron transport equation. This will **NOT** correspond to the transpose of the discrete operators! They serve a different purpose:
    The transpose of the standard operators is for derivative propagation; while the continuous adjoint is for coupling to variance reduction schemes such as CADIS.

The adjoint operator is defined as follows:

.. math::
    \int \psi^\dagger L[\psi] dP = \int \psi L^\dagger[\psi^\dagger] dP

Or equivalently:

.. math::
    \int \psi^\dagger q dP = \int \psi q^\dagger dP

for some source term :math:`q` and adjoint source :math:`q^\dagger`.

It can be shown that the adjoint operator is then 

.. math::
    L^\dagger[\psi^\dagger] = -\Omega \cdot \nabla \psi^\dagger + \sigma(\mathbf{r}, \Omega, E) \psi^\dagger - \int_{S^2} \int_0^\infty \sigma^s(\mathbf{r}, E \to E', \Omega \to \Omega') \psi^\dagger(\mathbf{r}, \Omega', E') dE' d\Omega'

such that the only modification are the negative sign in the streaming term and the replacement of the scattering term by its transpose (the incoming and outgoing angles and energies are swapped).

We will now derive the discrete form of the adjoint equation.

Multigroup form
---------------------------

Exactly the same as the forward equation, we have 

.. math::
    - \Omega \cdot \nabla \psi^\dagger_g +  \sigma_g \psi^\dagger_g - \sum_{g'}\sum_{lm} Y_{lm}(\Omega) \sigma^s_{l, g' \leftarrow g} \int_{S^2} Y_{lm}(\Omega') \psi^\dagger_{g'}(\mathbf{r}, \Omega') d\Omega' = q^\dagger_g \ \ \forall g

Note that we used that the scattering cross section is symmetric with respect to incoming and outgoing angles. In fact, in the multigroup from, only the streaming sign is changed
and the scattering cross section matrices are transposed.

We can then also write this in matrix form as 


.. math::
    :label: eq:multigroup_adjoint

    \begin{bmatrix} \overline{\mathcal{T}}_0  - \mathcal{S}_{00} & -\mathcal{S}_{10} & \cdots & -\mathcal{S}_{G0} \\ -\mathcal{S}_{01} & \overline{\mathcal{T}}_1  - \mathcal{S}_{11}& \cdots & -\mathcal{S}_{G1} \\ \vdots & \vdots & \ddots & \vdots \\ -\mathcal{S}_{0G} & -\mathcal{S}_{1G} & \cdots & \overline{\mathcal{T}}_G-  \mathcal{S}_{GG}\end{bmatrix}  \cdot \begin{bmatrix} \psi^\dagger_0 \\ \psi^\dagger_1 \\ \vdots \\ \psi^\dagger_G\end{bmatrix} =  \begin{bmatrix} Q^\dagger_0 \\ Q^\dagger_1 \\ \vdots \\ Q^\dagger_G\end{bmatrix}

where the overline denotes the sign change (here, it seems it's just a minus, but in the next section, the operator will be different!). The :math:`\mathcal{S}` is exactly
the same as in the forward case, but the indices are swapped  (compare with the equation :eq:`eq:multigroup` in the forward derivation )

Spherical Harmonics form
---------------------------

The spherical harmonics form of the adjoint equation is exactly the same as the forward equation, but with a sign change in the streaming term and a transpose of the scattering term. 

This yields the matrix equation as 

.. math::
    \begin{bmatrix}
    \overline{\boldsymbol{\tau}}_0 - \mathbf{S}_{00} & -\mathbf{S}_{10} & \cdots & -\mathbf{S}_{G0}\\
    -\mathbf{S}_{01} & \overline{\boldsymbol{\tau}}_1 - \mathbf{S}_{11} & \cdots & -\mathbf{S}_{G1}\\
    \vdots & \vdots & \ddots & \vdots\\
    -\mathbf{S}_{0G} & -\mathbf{S}_{1G} & \cdots & \overline{\boldsymbol{\tau}}_G - \mathbf{S}_{GG}
    \end{bmatrix} \cdot
    \begin{bmatrix}
    \{\phi^\dagger_{0,lm}\} \\
    \{\phi^\dagger_{1,lm}\} \\
    \vdots \\
    \{\phi^\dagger_{G,lm}\}
    \end{bmatrix} =
    \begin{bmatrix}
    \{Q^\dagger_{0,lm}\} \\
    \{Q^\dagger_{1,lm}\} \\
    \vdots \\
    \{Q^\dagger_{G,lm}\}
    \end{bmatrix}

Conditioning of the 

Conditioned Discrete Ordinates form
--------------------------------------

The exact same arguments hold for the conditioned discrete ordinates form, such that the final structure is just


.. math::
    \scriptsize
    \begin{bmatrix}
    \mathbf{I} - \mathbf{M} \circ \overline{\mathbf{T}}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{00} & -\mathbf{M} \circ \overline{\mathbf{T}}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{10} & \cdots & -\mathbf{M} \circ \overline{\mathbf{T}}_0^{-1} \circ \mathbf{D} \circ \mathbf{S}_{G0}\\
    -\mathbf{M} \circ \overline{\mathbf{T}}_1^{-1} \circ \mathbf{D} \circ \mathbf{S}_{01} & \mathbf{I} -\mathbf{M} \circ \overline{\mathbf{T}}_1^{-1} \circ \mathbf{D} \circ \mathbf{S}_{11} & \cdots & -\mathbf{M} \circ \overline{\mathbf{T}}_1^{-1} \circ \mathbf{D} \circ \mathbf{S}_{G1}\\
    \vdots & \vdots & \ddots & \vdots\\
    -\mathbf{M} \circ \overline{\mathbf{T}}_G^{-1} \circ \mathbf{D} \circ \mathbf{S}_{0G} & -\mathbf{M} \circ \overline{\mathbf{T}}_G^{-1} \circ \mathbf{D} \circ \mathbf{S}_{1G} & \cdots & \mathbf{I} - \mathbf{M} \circ \overline{\mathbf{T}}_G^{-1} \circ \mathbf{D} \circ \mathbf{S}_{GG}
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
    \{\phi^\dagger_{0,lm}\} \\
    \{\phi^\dagger_{1,lm}\} \\
    \vdots \\
    \{\phi^\dagger_{G,lm}\}
    \end{bmatrix} =
    \begin{bmatrix}
    \mathbf{M} \circ \overline{\mathbf{T}}_0^{-1} \circ \mathbf{D} \circ \{Q^\dagger_{0,lm}\} \\
    \mathbf{M} \circ \overline{\mathbf{T}}_1^{-1} \circ \mathbf{D} \circ \{Q^\dagger_{1,lm}\} \\
    \vdots \\
    \mathbf{M} \circ \overline{\mathbf{T}}_G^{-1} \circ \mathbf{D} \circ \{Q^\dagger_{G,lm}\}
    \end{bmatrix}

Fully discrete form
----------------------

For the fully discrete form, we just put a mass operator on before the transport operator. Therefore, we have (note that we have already :ref:`shown <transport_operator_transposition>` that the transpose of the discrete transport operator is just the reversel of the angle, which corresponds directly to the sign change in the streaming term, such that we 
can associate :math:`\overline{\mathbf{T}}^{-1}\to \mathbf{T}^{-T}` )

.. math::
    \scriptsize
    \begin{bmatrix}
    \mathbf{I} - \mathbf{M} \circ \mathbf{T}_0^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{00} & -\mathbf{M} \circ \mathbf{T}_0^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{10} & \cdots & -\mathbf{M} \circ \mathbf{T}_0^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{G0}\\
    -\mathbf{M} \circ \mathbf{T}_1^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{01} & \mathbf{I} - \mathbf{M} \circ \mathbf{T}_1^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{11} & \cdots & -\mathbf{M} \circ \mathbf{T}_1^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{G1}\\
    \vdots & \vdots & \ddots & \vdots\\
    -\mathbf{M} \circ \mathbf{T}_G^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{0G} & -\mathbf{M} \circ \mathbf{T}_G^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{1G} & \cdots & \mathbf{I} - \mathbf{M} \circ \mathbf{T}_G^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \mathbf{S}_{GG}
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
    \{\phi^\dagger_{0,lm}\} \\
    \{\phi^\dagger_{1,lm}\} \\
    \vdots \\
    \{\phi^\dagger_{G,lm}\}
    \end{bmatrix} =
    \begin{bmatrix}
    \mathbf{M} \circ \mathbf{T}_0^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \{Q^\dagger_{0,lm}\} \\
    \mathbf{M} \circ \mathbf{T}_1^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \{Q^\dagger_{1,lm}\} \\
    \vdots \\
    \mathbf{M} \circ \mathbf{T}_G^{-T} \circ \mathbf{H} \circ \mathbf{D} \circ \{Q^\dagger_{G,lm}\}
    \end{bmatrix}


This is very similar to the normal multigroup equation. The big difference is that the transport operator is transposed, which can be directly
copied from the transpose of the TransportOperator, and the scattering is also backwards, which can be directly copied from the GatherScatterOperator.


