.. _simplex_elements:
Simplex simplification
---------------------
Simplices are elements with :math:`d+1` vertices in :math:`d` dimensions connected by straight lines or faces -- intervals, triangles, and tetrahedra. 
Since they are built from the same building blocks, the full discretisation can be written exactly the same for all three dimensions, reducing implementations.

We will use the single-angle version now:

.. math::
        &T[\psi_{ei}] &&= \mathbf{G}_{\epsilon}[\psi_{\epsilon i}] - \sum_{\beta \in \beta^{-}_{\epsilon}}  \mathbf{F}_{\epsilon \epsilon(\beta)}[\psi_{\epsilon (\beta) i}]\\ 
        &\mathbf{G}_{\epsilon}[\psi_{\epsilon i}] &&=\sum_i \psi_{\epsilon i } \left[\int_{\epsilon} H_{\epsilon i}\left( \sigma H_{\epsilon j} - \Omega \cdot \nabla   H_{\epsilon j}\right)\, d V  + \sum_{\beta \in \beta^+_{\epsilon}}|\Omega \cdot \mathbf{n}_\beta|\int_{\beta} H_{\epsilon i} H_{\epsilon j} \, d A \right] \\
        &\mathbf{F}_{\epsilon \epsilon(\beta)}[\psi_{\epsilon(\beta)i}] &&= \sum_{i}\psi_{\epsilon(\beta) i }|\Omega \cdot \mathbf{n}_\beta| \int_{\beta} H_{\epsilon(\beta) i} H_{\epsilon j}\, d A

Normally, these integrals are computed using quadrature rules, but for simplices, they can be computed fully analytically. The Jacobians do not change 
within such an element and can therefore be taken out of the integral.

For any element :math:`\epsilon` and a reference element :math:`\mathcal{E}` (unit interval, unit triangle, or unit tetrahedron),
the Jacobian matrix :math:`\mathbf{J}_\epsilon` relates the coordinate in the standard element :math:`\mathbf{q}` to the real spatial location :math:`\mathbf{r}` 
of any arbitrary simplex :math:`\epsilon`:

.. math::
    \mathbf{J}_\epsilon = J_{\epsilon ij} =  \frac{\partial r_i}{\partial q_j}

The basis functions are then defined in terms of the coordinate on the reference element:

.. math::
    H_{\epsilon i}(\mathbf{r}) = \hat{H}_i(\mathbf{q}(\mathbf{r})) \text{ and } \hat{H}_i(\mathbf{q}) = H_{\epsilon i}(\mathbf{r}(\mathbf{q}))

with hat denoting basis function on the reference element and :math:`\mathbf{q}(\mathbf{r})` is the inverse mapping from the real element :math:`\epsilon`
to the reference element (note this mapping naturally depends on the element, we just drop it for brevity here).

The deriviatves can then be obtained as 

.. math::
    \Omega \cdot \nabla H_{\epsilon i} = \sum_{uv} \Omega_{v} (\mathbf{J}_\epsilon)^{-1}_{uv}\frac{\partial \hat{H}_{i}}{\partial q_{u}}

For this simplex elements, the Jacobian determinant is indepent of the position within the element:

.. math::
    \int_{\epsilon} f(\mathbf{r}) dV = \int_{\mathcal{E}} f(\mathbf{r}(\mathbf{q})) |\det \mathbf{J}_\epsilon| d\mathbf{q} = |\det \mathbf{J}_\epsilon| \int_{\mathcal{E}} f(\mathbf{r}(\mathbf{q})) d\mathbf{q}

We just denote :math:`\mathcal{J}_\epsilon = |\det \mathbf{J}_\epsilon|` for brevity. Now, by assumption, the cross is constant within the element and we can write 
the volume element integral as 

.. math::
    \int_{\epsilon} H_{\epsilon i}\left( \sigma H_{\epsilon j} - \Omega \cdot \nabla   H_{\epsilon j}\right)\, d V &=  \displaystyle\mathcal{J}_\epsilon \left[\sigma_\epsilon\int_{\mathcal{E}}\hat{H}_{i } \hat{H}_{j}\, d \mathbf{q}  + \sum_{uv} \Omega_v (\mathbf{J}_\epsilon)^{-1}_{uv } \int_{\mathcal{E}} -\frac{\partial \hat{H}_{j}}{\partial q_u} \hat{H}_{i}\, d \mathbf{q}\right] \nonumber \\ 
    &= \mathcal{J}_\epsilon \left[\sigma_{\epsilon} \mathcal{A}_{ij} + \sum_{uv} \Omega_{v} (\mathbf{J}_{\epsilon})^{-1}_{uv} \mathcal{B}_{iju}\right]
    
The caligraphic matrices do not depend on the element and can be stored and retrieved from a :class:`central location <jax_sn.solution_domain.FEMMatrices>`.

For edge integrals, the same holds: a edge of a :math:`d`-dimensional simplex is a :math:`(d-1)`-dimensional simplex, and the Jacobian determinant is constant over the edge.
The index of the face in the reference element will be denotes as :math:`\beta`:

.. math::
    \int_{\beta(\epsilon)} H_{\epsilon i}H_{\epsilon j} \, d A = \mathcal{J}_{\beta} \int_{\beta(\mathcal{E})} \hat{H}_{i } \hat{H}_{j} \, d A  = \mathcal{J}_\beta \mathcal{C}_{\beta ij}

The connected face integral is somewhat more difficult. Each node of the connected simplex can be connected to each node of the connected face, leading to :math:`(d+1)!` possilbe permutations :math:`p`.
Then, :math:`p(\epsilon, \beta)` denotes the permutation of elemen :math:`\epsilon(\beta)` connected to element :math:`\epsilon` through face :math:`\beta`. We can precompute all permuted matrices:

.. math::
    \mathcal{D}^{p\beta}_{ij} = \int_{\beta(\mathcal{E})} \hat{H}^p_{i}\hat{H}_{j} \, d A 

and the total simplex transport operator becomes

.. math::
    &T[\psi_{ei}] &&= \mathbf{G}_{\epsilon }[\psi_{\epsilon i}]  - \sum_{\beta \in \beta^-_{\epsilon}}\mathbf{F}_{\epsilon \epsilon(\beta)}[\psi_{\epsilon(\beta)i}]\\ 
    &\mathbf{G}_{\epsilon}[\psi_{\epsilon i}] &&=\sum_{i}\psi_{\epsilon i} \left[\sigma_{\epsilon}  \mathcal{J}_\epsilon \mathcal{A}_{ij} +  \mathcal{J}_\epsilon\sum_{uv} \Omega_{v} (\mathbf{J}_{\epsilon})^{-1}_{uv} \mathcal{B}_{iju} + \sum_{\beta \ \in \beta_\epsilon^+} \mathcal{J}_\beta |\Omega \cdot \mathbf{n}_\beta|\mathcal{C}_{\beta ij} \right]\\ 
    &\mathbf{F}_{\epsilon \epsilon(\beta)}[\psi_{\epsilon(\beta)i}] &&= \sum_{i}\psi_{\epsilon(\beta)i} |\Omega \cdot \mathbf{n}_{\beta}|\mathcal{J}_\beta \mathcal{ D}^{p(\epsilon,\beta)\beta}_{ij}

This is only in terms of :class:`geometric quantities <jax_sn.solution_domain.PrecomputedGeometricData>` or in terms of :class:`precomputed fem matrices <jax_sn.solution_domain.FEMMatrices>`. 
Thus, during sweeping, no integration needs to be done and everything is just simple matrix-vector products.

