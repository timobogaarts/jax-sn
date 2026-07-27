.. _complete_discretisation:
Complete Discretisation
--------------------------

The total system is 

.. math::
    (\mathbf{I} -  \mathbf{M}\mathbf{T}^{-1} \mathbf{H}\mathbf{D}\mathbf{S})[\phi_{g\ell ei}] =\mathbf{M}\mathbf{T}^{-1} \mathbf{H} [q_{gkei}]

with the following operator definitions:

.. math::
    &\mathbf{M}[a_{\dots k \dots}]            &&= \sum_{k }w_k Y_{\ell k}a_{\dots k \dots}\\
    &\mathbf{D}[b_{\dots \ell \dots}]         &&= \sum_{\ell } Y_{\ell k}b_{\dots \ell \dots}\\
    &\mathbf{H}[c_{\dots ei}]                 &&= \sum_{i}  c_{\dots \epsilon i}\int_{\epsilon} H_{\epsilon j } H_{\epsilon i} \, d V \\
    &\mathbf{S}[d_{\tilde{g} \ell \epsilon \dots }]  &&= \sum_{\tilde{g}} \sigma^s_{\ell g \tilde{g} \epsilon } d_{\tilde{g} \ell \epsilon \dots } \\
    &\mathbf{T}[\psi_{gkei}]                  &&= \mathbf{G}_{gk\epsilon}[\psi_{gk\epsilon i}] - \sum_{\beta \in \beta^-_{\epsilon k}} \mathbf{F}_{k\epsilon,\epsilon(\beta)}[\psi_{gk\epsilon(\beta)i}] \\
    &                                      &&= \sum_i \psi_{gk\epsilon i} \left[\int_{\epsilon} H_{\epsilon i}\left( \sigma_g H_{\epsilon j} - \Omega_k \cdot \nabla H_{\epsilon j}\right)d V + \sum_{\beta \in \beta^+_{\epsilon k}}|\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{\epsilon i} H_{\epsilon j} d A \right] \nonumber\\
    &                                      &&\quad - \sum_{\beta \in \beta^-_{\epsilon k}}\sum_i \psi_{gk\epsilon (\beta)i}\,|\Omega_k \cdot \mathbf{n}_\beta| \int_{\beta} H_{\epsilon(\beta)i} H_{\epsilon j}d A \nonumber

and the efficient transport operator inversion algorithm described in the previous section. This defines the total discretisation.

There are several topics that are still open:

- As discussed in section :ref:`linear_systems_differentation`, for an efficient implicit differenitation, we still need to implement the transpose of this operator. 

- We can exploit the lower triangular block structure to efficiently solve the downscatter simulation.

- We have to choose a particular finite element basis. The above formulation supports any choice, but we will use simplices in most cases.

- For FW-CADIS we have to solve the adjoint neutron transport equation, which is not the same as the adjoint of the discrete equation.


