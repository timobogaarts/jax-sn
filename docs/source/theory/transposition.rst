Transposition
--------------------------
The operator formalism can be easily transposed:

.. math::
    (\mathbf{I} -\mathbf{M}\mathbf{T}^{-1} \mathbf{H} \mathbf{D} \mathbf{S})^T = \mathbf{I} -\mathbf{S}^T\mathbf{D}^T \mathbf{H}^T \mathbf{T}^{-T}\mathbf{M}^T

Thus, we just have to transpose each operator.

This is defined by the following identity:

.. math::
    \mathbf{A} : V \to U \ \ x \in U, \ \ y\in V \\
    \langle \mathbf{A}^T[x], y \rangle = \langle x, \mathbf{A}[y]\rangle

Below we give the proofs of all other operators:

:math:`\mathbf{M}` operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have:

.. math::
    \mathbf{M}^T[ a_{\dots \ell \dots }] = \sum_{\ell } w_k Y_{\ell k } a_{\dots \ell \dots}


Proof (take :math:`a_{g\ell ei}, b_{gkei}`):


.. math::
    \langle \mathbf{M}^T[ a_{g \ell ei }], b_{g k ei}\rangle = \sum_{gkei} \sum_{\ell} w_k Y_{\ell k } a_{g\ell ei} b_{gkei} = \sum_{g\ell ei}a_{g\ell ei} \sum_{k} w_k Y_{\ell k } b_{gkei} = \langle a_{g\ell ei}, \mathbf{M}[b_{gkei}]\rangle

:math:`\mathbf{D}` operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have:

.. math::
    \mathbf{D}^T[ a_{\dots k \dots }] = \sum_{k } Y_{\ell k} a_{\dots k \dots}
    
Proof (take :math:`a_{gkei}, b_{g\ell ei}`):

.. math::
    \langle \mathbf{D}^T[ a_{g k ei }], b_{g \ell ei}\rangle = \sum_{g\ell ei} \sum_{k} Y_{\ell k } a_{g k ei} b_{g\ell ei} = \sum_{g k ei}a_{g k ei} \sum_{\ell} Y_{\ell k } b_{g\ell ei} = \langle a_{g k ei}, \mathbf{D}[b_{g\ell ei}]\rangle

:math:`\mathbf{H}` operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have:

.. math::
    \mathbf{H}^T[ a_{\dots ei }] = \mathbf{H}[ a_{\dots ei }]

It is just diagonal in all indices (dots are identity operations) and it is symmetric in the :math:`i` indices (symmetric mass matrix). Thus, it is self-adjoint.

:math:`\mathbf{S}` operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have:

.. math::
    \mathbf{S}^T[a_{g\ell e \dots }] = \sum_{g} \sigma^s_{\ell g\tilde{g} e} a_{g\ell e \dots}

Proof:

.. math::
    \langle \mathbf{S}^T[a_{g\ell ei}], b_{\tilde{g}\ell ei}\rangle = \sum_{\tilde{g}\ell ei}\sum_{g}\sigma^s_{\ell g\tilde{g}e}\,a_{g\ell ei}\,b_{\tilde{g}\ell ei} = \sum_{g\ell ei}a_{g\ell ei}\sum_{\tilde{g}}\sigma^s_{\ell g\tilde{g}e}\,b_{\tilde{g}\ell ei} = \langle a_{g\ell ei}, \mathbf{S}[b_{\tilde{g}\ell ei}]\rangle


:math:`\mathbf{T}^{-1}` operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is somewhat different: we will show below that the transpose of the inverse is the same as inverting a transport operator with all angles negated.

*Proposition* The transpose of the transport operator is obtained by substituion of :math:`\Omega_k \to -\Omega_k` in the transport operator, with inversion traversing the graph in reverse (i.e. with reversed topological order)

*Proof* Since :math:`\mathbf{T}` is diagonal, the transpose is just the tranpose of each individual single-angle single-group operator. Proving it for a single-group single-angle operator is sufficient.
Dropping all indices now.

First, we show that :math:`\mathbf{T}^T`  is equal to :math:`\mathbf{T}_r`, which denotes the same operator with its angle reversed (:math:`\Omega \to -\Omega`):

We now decompose the transport operator into several parts:

.. math::
    \mathbf{A}[a_{ei}]      &= \sum_{i} a_{ei}\int_{e} -\Omega_k \cdot \nabla(H_{ej})\,H_{ei}\,d V \\
    \mathbf{C}[a_{ei}]      &= \sum_{i} a_{ei}\int_{e} \sigma_{ge}\,H_{ej} H_{ei}\,d V \\
    \mathbf{F}^+[a_{ei}]    &= \sum_{i} a_{ei}\sum_{\beta \in \beta^+_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A \\
    \mathbf{F}^-[a_{ei}]    &= \sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\sum_{p} a_{e(\beta)p}\int_{\beta} H_{ej} H_{e(\beta)p}\,d A

and their respective reversed versions:

.. math::
    \mathbf{A}_{r}[b_{ei}]      &= \sum_{i} b_{ei}\int_{e} \Omega_k \cdot \nabla(H_{ej})\,H_{ei}\,d V \\
    \mathbf{C}_{r}[b_{ei}]      &= \sum_{i} b_{ei}\int_{e} \sigma_{ge}\,H_{ej} H_{ei}\,d V \\
    \mathbf{F}^+_r[b_{ei}]  &= \sum_{i} b_{ei}\sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A \\
    \mathbf{F}^-_r[b_{ei}]  &= \sum_{\beta \in \beta^+_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\sum_{p} b_{e(\beta)p}\int_{\beta} H_{ej} H_{e(\beta)p}\,d A

Now we have to show that this satisfies the transpose identity:

.. math::
    \langle \mathbf{T}_r[a_{ei}], b_{ei}\rangle = \langle a_{ei}, \mathbf{T}[b_{ei}]\rangle

First, the :math:`\mathbf{C}` operator is symmetric:

.. math::
    \langle \mathbf{C}[a], b\rangle = \sum_{ej}\sum_i a_{ei}\int_e \sigma_{ge} H_{ej} H_{ei}\,d V\, b_{ej} = \sum_{ei} a_{ei} \sum_j b_{ej}\int_e \sigma_{ge} H_{ei} H_{ej}\,d V = \langle a, \mathbf{C}_r[b]\rangle

and thus we require only:

.. math::
    \langle \mathbf{A}[a] + \mathbf{F}^+[a] - \mathbf{F}^-[a], b\rangle = \langle a, \mathbf{A}_r[b] + \mathbf{F}^+_r[b] - \mathbf{F}^-_r[b]\rangle.

Now, start from 

.. math::
    \langle \mathbf{A}[a], b\rangle = \sum_{ej} b_{ej}\left(\sum_i a_{ei}\int_{e} -\Omega_k \cdot \nabla(H_{ej})\,H_{ei}\,d V\right),

and integrate by parts (a second integration does *not* require additional numerical flux: within the element it is exact)

.. math::
    \sum_{ej} b_{ej}\left(\sum_i a_{ei}\left[\int_{e} \Omega_k \cdot \nabla(H_{ei})\,H_{ej}\,d V - \sum_{\beta \in \beta^+_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A + \sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A\right]\right)

The outflow term cancels :math:`\langle \mathbf{F}^+[a], b\rangle` while the inflow term does *not* cancel :math:`\langle \mathbf{F}^-[a], b\rangle` because the latter uses the 
neighbouring element basis :math:`H_{e(\beta)p}` instead of :math:`H_{ei}`. Combining with both :math:`\mathbf{F}^+` and :math:`\mathbf{F}^-` terms, we have

.. math::
    \langle \mathbf{A}[a] + \mathbf{F}^+[a] - \mathbf{F}^-[a], b\rangle &= \sum_{ej} b_{ej}\left(\sum_i a_{ei}\int_{e} \Omega_k \cdot \nabla(H_{ei})\,H_{ej}\,d V\right) \nonumber \\
    &\quad + \sum_{eij} b_{ej} a_{ei}\sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A - \langle \mathbf{F}^-[a], b\rangle.

The first term is precisely :math:`\langle a, \mathbf{A}_r[b]\rangle` after renaming dummy indices:

.. math::
    \langle a, \mathbf{A}_r[b]\rangle = \sum_{ej} a_{ej}\sum_i b_{ei}\int_{e} \Omega_k \cdot \nabla(H_{ej})\,H_{ei}\,d V.

Thus, we now only need to prove the remaining terms:

.. math::
    \sum_{eij} b_{ej} a_{ei}\sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A - \langle \mathbf{F}^-[a], b\rangle = \langle a, \mathbf{F}^+_r[b] - \mathbf{F}^-_r[b]\rangle.

From the definitions, :math:`\langle a, \mathbf{F}^+_r[b] - \mathbf{F}^-_r[b]\rangle` is equal to

.. math::
     \sum_{ej} a_{ej}\left(\sum_i b_{ei}\sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\int_{\beta} H_{ej} H_{ei}\,d A - \sum_{\beta \in \beta^+_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\sum_p b_{e(\beta)p}\int_{\beta} H_{ej} H_{e(\beta)p}\,d A\right).

The first term after renaming dummy indices equals the first term on the left-hand-side, so we only need to show :math:`\langle \mathbf{F}^-[a], b\rangle = \langle a, \mathbf{F}^-_r[b]\rangle`, or

.. math::
    &\sum_{ej} b_{ej}\sum_{\beta \in \beta^-_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\sum_p a_{e(\beta)p}\int_{\beta} H_{ej} H_{e(\beta)p}\,d A \nonumber \\
    = &\sum_{ej} a_{ej}\sum_{\beta \in \beta^+_{ek}} |\Omega_k \cdot \mathbf{n}_\beta|\sum_p b_{e(\beta)p}\int_{\beta} H_{ej} H_{e(\beta)p}\,d A.

Now we introduce a global face index :math:`f`, shared by elements :math:`e^+(f)` and :math:`e^-(f)`, with :math:`\Omega \cdot \mathbf{n}_f>0` for :math:`e^+(f)` and :math:`\Omega \cdot \mathbf{n}_f<0` for :math:`e^-(f)`. 
Instead of summing all elements and then selecting inflow/outflow, we can also sum directly over all faces (that are shared by two elements, i.e. not boundary faces), and then select the *element*
:math:`e^+(f)` or :math:`e^-(f)` for the inflow/outflow. This gives for the left-hand-side:

.. math::
    \sum_{fj} b_{e^-(f)j}\,|\Omega_k \cdot \mathbf{n}_f|\sum_p a_{e^+(f)p}\int_{f} H_{e^-(f)j} H_{e^+(f)p}\,d A,

and for the right-hand-side 

.. math::
    \sum_{fj} a_{e^+(f)j}\,|\Omega_k \cdot \mathbf{n}_f|\sum_p b_{e^-(f)p}\int_{f} H_{e^+(f)j} H_{e^-(f)p}\,d A.

These are equal if you rename :math:`j \leftrightarrow p` and use face integral symmetry -- completing the first part of the proof: :math:`\mathbf{T}^T = \mathbf{T}_r`.

For the inversion part, this operator :math:`T_r` is equally a single-angle transport operator, just with a different angle. It can therefore be inverted in the same algorithm.
Since reversing the angle reverses the graph, from graph theory it follows that the inverted topological sort is then a valid topological sort of the reversed graph. 
Thus, we can invert the transposed operator by traversing the graph in reverse order, using the same algorithm as for the original operator.
For the levels, they can be obtained by subtracting the level of each element from the highest level. Thus the efficient inversion algorithms work for the transpose as well, with no additional
topological sort required (just flip the array).

:math:`\blacksquare`