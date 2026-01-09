.. _transport_operator_transposition:

Transport operator transposition 
=======================================================

Here we prove that the transport operator :math:`\mathbf{T}` as defined in
:class:`jax_sn.operators.transport_operator.TransportOperator`
has as transpose the operator :math:`\mathbf{T}^T` defined by reversing the
direction of travel :math:`\Omega_k \rightarrow - \Omega_k` in the
definition of :math:`\mathbf{T}`.

This requires (note the element index stays the same in the transport operator), :math:`\mathbf{a}\in \mathcal{F}`, :math:`\mathbf{b} \in \mathcal{F}`:

.. math::

    <\mathbf{a}, \mathbf{T}^T[\mathbf{b}] >  = <\mathbf{T} [\mathbf{a}], \mathbf{b} > 


We prove it here for vacuum boundary conditions; the upwind flux is zero if not connected to another element.


Proof
-------

First, we note that the inner product on the space :math:`\mathcal{F}` is defined as:

.. math::
    <\mathbf{a}, \mathbf{b} > = \sum_{ei} a_{ei} b_{ei}

Then, we define:

.. math::
    \mathbf{A}\mathbf{a} &= \sum_{j} a_{ej}\int_{e} -  \Omega_k \cdot \nabla (H_i) H_j dV\\ 
    \mathbf{M}\mathbf{a} &= \sum_{j} a_{ej}\int_{e} \sigma^t_{ge} H_i H_j dV\\ 
    \mathbf{F}^+ \mathbf{a} &= \sum_{j} a_{ej} \sum_{\gamma, \Omega_k \cdot n_{\gamma} > 0} |\Omega_k \cdot n_{\gamma}| \int_{\gamma} H_i H_j dA\\
    \mathbf{F}^- \mathbf{a} &= \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0} |\Omega_k \cdot n_{\gamma}| \sum_{p} a_{e(\gamma),p } \int_{\gamma} H_i  H_p^{(e(\gamma))}dA

The supposed transpose variants (same but with :math:`\Omega_k` replaced by :math:`-\Omega_k`) are then:

.. math::
    \mathbf{A}^T\mathbf{b} &= \sum_{j} b_{ej}\int_{e}  \Omega_k \cdot \nabla (H_i) H_j dV\\ 
    \mathbf{M}^T\mathbf{b} &= \sum_{j} b_{ej}\int_{e} \sigma^t_{ge} H_i H_j dV\\ 
    (\mathbf{F}^+)^T \mathbf{b} &= \sum_{j} b_{ej} \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0} |\Omega_k \cdot n_{\gamma}| \int_{\gamma} H_i H_j dA\\
    (\mathbf{F}^-)^T \mathbf{b} &= \sum_{\gamma, \Omega_k \cdot n_{\gamma} > 0} |\Omega_k \cdot n_{\gamma}| \sum_{p} b_{e(\gamma),p } \int_{\gamma} H_i  H_p^{(e(\gamma))}dA
    

Note that 

.. math::
    < \mathbf{M}\mathbf{a}, \mathbf{b}> = \sum_{ei}\sum_j a_{ej}\int_e \sigma^t_{ge} H_i H_j dV b_{ei} =  <\mathbf{a}, \mathbf{M}^T \mathbf{b}>
    
There, we only have to prove the remaining parts:

.. math::
     <\mathbf{A}\mathbf{a} + \mathbf{F}^+ \mathbf{a} - \mathbf{F}^- \mathbf{a}, \mathbf{b}>  =<\mathbf{a}, \mathbf{A}^T \mathbf{b} + (\mathbf{F}^+)^T \mathbf{b} - (\mathbf{F}^-)^T \mathbf{b}> 


First, 

.. math::
    < \mathbf{A}\mathbf{a}, \mathbf{b}> = \sum_{ei} b_{ei}\left(\sum_j a_{ej} \int_{e} -  \Omega_k \cdot \nabla (H_i) H_j dV \right)

Partially integrate over the element :math:`e` (note a second partial integration in the DG method does *not* require another numerical flux.):

.. math::
    \sum_{ei}   b_{ei}\left(\sum_j a_{ej} \left[\int_{e} +  \Omega_k \cdot \nabla (H_j) H_i dV  - \sum_{\gamma, \Omega_k\cdot n_{\gamma} > 0} |\Omega_k \cdot n_{\gamma}| \int_{\gamma} H_i H_j dA +  \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0} |\Omega_k \cdot n_{\gamma}| \int H_i H_j^{e} dA\right]\right)

Thus, we have that the second to last-term exactly cancels the :math:`\mathbf{F}^+` term. The last term does not cancel the :math:`\mathbf{F}^-` term, since the indices are different. Combining, we have:

.. math::
    < \mathbf{A}\mathbf{a} + \mathbf{F}^+ \mathbf{a} - \mathbf{F}^- \mathbf{a}, \mathbf{b}>&  = \sum_{ei}   b_{ei}\left(\sum_j a_{ej} \int_{e} +  \Omega_k \cdot \nabla (H_j) H_i dV \right) \\ 
    & + \sum_{eij} b_{ei} a_{ej}|\Omega_k \cdot n_{\gamma}| \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0}  \int H_i H_j^{e} dA - < \mathbf{F}^- \mathbf{a}, \mathbf{b}>

The first term is exactly equal to (renaming dummy indices):

.. math::
    <\mathbf{a}, \mathbf{A}^T \mathbf{b}> = \sum_{ei}   a_{ei}\left(\sum_j b_{ej} \int_{e} +  \Omega_k \cdot \nabla (H_i) H_j dV \right)\equiv \sum_{eij}   a_{ej}\sum_i b_{ei} \int_{e} +  \Omega_k \cdot \nabla (H_j) H_i dV 

It suffices to prove that:

.. math::
      \sum_{eij} b_{ei} a_{ej} \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0} |\Omega_k \cdot n_{\gamma}| \int H_i H_j^{e} dA - < \mathbf{F}^- \mathbf{a}, \mathbf{b}> = <\mathbf{a}, (\mathbf{F}^+)^T \mathbf{b} - (\mathbf{F}^-)^T \mathbf{b}> 

From the definition:

.. math::
    &<\mathbf{a}, (\mathbf{F}^+)^T \mathbf{b} - (\mathbf{F}^-)^T \mathbf{b}> =\\ 
    & \sum_{ei} a_{ei} \left( \sum_{j} b_{ej} \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0} |\Omega_k \cdot n_{\gamma}| \int_{\gamma} H_j H_i dA - \sum_{\gamma, \Omega_k \cdot n_{\gamma} > 0} |\Omega_k \cdot n_{\gamma}| \sum_{p} b_{ e(\gamma),p } \int_{\gamma} H_i  H_p^{(e(\gamma))}dA \right)

The first term is equivalent (after renaming dummy indices) to the first term on the left-hand  side. Therefore, it suffices to prove that:

.. math::
    &- < \mathbf{F}^- \mathbf{a}, \mathbf{b}> = - <\mathbf{a}, (\mathbf{F}^-)^T \mathbf{b}> \\ 
    

or equivalently:

.. math::
     &- \sum_{ei} b_{ei} \left( \sum_{\gamma, \Omega_k \cdot n_{\gamma} < 0} |\Omega_k \cdot n_{\gamma}| \sum_{p} a_{ e(\gamma),p } \int_{\gamma} H_i  H_p^{(e(\gamma))}dA \right) \\
    = &- \sum_{ei} a_{ei} \left( \sum_{\gamma, \Omega_k \cdot n_{\gamma} > 0} |\Omega_k \cdot n_{\gamma}| \sum_{p} b_{ e(\gamma),p } \int_{\gamma} H_i  H_p^{(e(\gamma))}dA \right) 

Let us introduce a global face index :math:`f`. The face :math:`f` connects two elements :math:`e^+(f)` and :math:`e^-(f)`. By definition, :math:`\Omega_k \cdot n_{\gamma} > 0` for element :math:`e^+(f)` and :math:`\Omega_k \cdot n_{\gamma} < 0` for element :math:`e^-(f)`. Instead of summing over each element and then picking out
the faces connected to an element that are inflow or outflow, we can instead immediately sum over all faces (that are shared by two elements) and pick the elements depending on whether they are in or outflow for that specific face. We can thus rewrite the left-hand side as:

.. math::
    - \sum_{f i} b_{e^-(f)i} \left( |\Omega_k \cdot n_{f}| \sum_{p} a_{ e^+(f),p } \int_{f} H^{e^+(f)}_i  H_p^{(e^+(f))}dA \right)

Similarly, the right-hand side becomes:

.. math::
    - \sum_{f i} a_{e^+(f)i} \left( |\Omega_k \cdot n_{f}| \sum_{p} b_{ e^-(f),p } \int_{f} H_i{e^-(f)}  H_p^{(e^-(f))}dA \right)

These are equivalent. This completes the proof.


