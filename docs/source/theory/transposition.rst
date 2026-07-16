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