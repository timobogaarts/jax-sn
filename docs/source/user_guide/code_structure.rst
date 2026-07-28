Code Structure
=======================

From the :ref:`theory <complete_discretisation>`, the discretised version of the linear neutron transport equation is 

.. math::
    \mathbf{A}[\phi_{g\ell ei}] = (\mathbf{I} -  \mathbf{M}\mathbf{T}^{-1} \mathbf{H}\mathbf{D}\mathbf{S})[\phi_{g\ell ei}] =\mathbf{M}\mathbf{T}^{-1} \mathbf{H} [q_{gkei}]

where the bold-faced letters indicate linear operators: :math:`\mathbf{I}` is the identity operator, 
:math:`\mathbf{M}` the operator converting values defined on the discrete angles to values on the unit sphere (moments), 
:math:`\mathbf{D}` the operator doing the reverse, :math:`\mathbf{T}^{-1}` the inverted transport operator (using an efficient sweep 
algorithm such that it is as expensive as the normal forward operator), :math:`\mathbf{H}` the finite element mass operator,
and :math:`\mathbf{S}` the scattering operator. For only using the code, their precise definition is not important.

This linear equation is solved by using a iterative solver which only requires applying the operators to an iteration vector :math:`\phi_{g\ell ei}^{(q)}`, 
we don't need to have a matrix representation of the full system. For differentiation, we should be able to solve the transpose system.

Below we will describe the code structure in a simplified way.

Energy Set
------------
Energy sets don't actually matter for the discretisation; they are implicit in the cross sections. This 
submodule just has some common energy sets and a way to collapse group fluxes to a different energy set.

Quadrature Set
----------------
This submodule defines the angular quadrature sets and provides functions for computing the required data for the 
moment and discrete operators :math:`\mathbf{M}, \mathbf{D}`. It also provides a way to reduce the dimensionality of a 3D 
quadrature set for lower dimensional problems.




GeometricDomain and Domain
-----------------------------
A :class:`GeometricDomain` defines how the domain is geometrically constructed: its vertices and the connection between 
each vertices. Can compute the required discontinuous-Galerkin connected to matrices (which face connects to which other face or element)
and Jacobian matrices. Also handles interpolation location (using BVH trees).

A :class:`Domain` also adds cross section information: which element has which cross section in :class:`CrossSectionIndexed`.

SolutionDomain 
---------------
:class:`SolutionDomain` defines which type of :class:`Element` is used on a :class:`Domain` (e.g. first order / second order) and 
has functions for computing the relevant finite element matrices. Also checks compatibility.

Operators and Strategies
--------------------------

The multigroup system is essentially an operator equation, which may consist of sub-operators. This module defines all operators 
and sub-operators needed for constructing it.

An operator is defined solely by its meaning (what does it calculate). To reflect this (see developer guide for a more indepth argument),
each operator is implemented as follows:

.. code-block:: python

    class Operator(BaseLinearOperator)

        operator_data     : OperatorData
        strategy : Strategy

        def mv(self, x ):
            return strategy.forward(self.operator_data, x)

where the strategy only defines the numerics (how it calculates). Constructing the multigroup operator thus requires picking 
a implementation of :class:`MultiGroupStrategyBase` and constructing the operator data (which defines the mesh, and other discretisation parameters).
Then, it can be passed to a `Lineax <https://github.com/patrick-kidger/lineax>`_ solver, which automatically handles the differentiation as well.

Strategies can be composed: in the multigroup system, we can either compute the entire action :math:`\mathbf{A}` using a single 
strategy (i.e. writing the whole kernel in one go), or we can decompose it into each sub-operator. This looks schematically as 

.. code-block:: python    

    class FusedMultiGroupStrategy(MultiGroupStrategyBase):
        def forward(data, x):
            return compute_fused_kernel(data, x)
    class MultiGroupChainGatherStrategy(StrategyA):
        strategy_m : StrategyM
        strategy_tinv : StrategyTinv
        strategy_h    : StrategyH 
        strategy_d    : StrategyD 
        strategy_s    : StrategyS

        def forward(data, x):
            return x - strategy_m.forward(data, 
                            strategy_tinv.forward(data, 
                                strategy_h.forward(data, 
                                    strategy_d.forward(data, 
                                        strategy_s.forward(data, x) ) ) ) )

In fact, since each operator can also be applied either for a single energy group (instead of multiple groups as above),
this can even be further decomposed into single-group strategies, and a group lifting (how to parallelize / serialize).
Even more, each *angle* can be further decomposed for a single group (do you fuse in a CUDA kernel or just ```jax.vmap```?). 
We want to remain flexible in these decisions since we don't know what is the best a priori for a given problem.

Solvers
-----------

This submodule defines some useful solvers, especially for fusion problems with only downscatter where a lower triangular block solver can be used.
This saves a lot of memory since the iteration vector is much smaller than in the full system.


Parametrisation
----------------
We are interested in parametrising arbitrary quantities. This is supported, but you do need to recalculate various quantities if you update 
the vertices (e.g. the Jacobian matrices or the sweep ordering for :math:`T^{-1}`). This submodule provides an easy 
way to do just that for several of the operators.


Variance Reduction
-------------------
Running stochastic simulations can be very expensive; this submodule provides a way to construct weight windows from 
deterministic simulations using the continuous adjoint FW-CADIS method.






.. code-

.. The code structure is motivated by the fact that we would like to be flexible in applying these operators since we *a priori* don't 
.. know which implementation is efficient: do we want to apply :math:`\mathbf{H}\mathbf{D}` in one single operation, or do we prefer to have 
.. them separate? Do we want to apply each operator in parallel over each group or do we just want to do one group at a time? Do we 
.. parallelize over the angles? If so, how? Do we solve a lower-triangular system for downscattering, or do we do it at the same time?

.. Operators and Strategies
.. ~~~~~~~~~~~~~~~~~~~~~~~~~
.. These questions did (and do) not have clear answers (which may also just depend on the problem), and therefore we want to keep the structure as flexible as possible.
.. This motivates the following pattern: each operator defines its meaning, its input/output structure, its transposition, and its required 
.. data using a `Lineax <https://github.com/patrick-kidger/lineax>`_ base operator (and some extra specific functions). Its numerics 
.. are defined by a *strategy*, which is essentially the kernel used to execute that operator. The basic idea is below:

.. .. code-block:: python

..     class Operator(BaseLinearOperator)

..         operator_data     : OperatorData
..         strategy : Strategy

..         def mv(self, x ):
..             return strategy.forward(self.operator_data, x)


.. Composited operators are then quite easy: let's assume we have an operator 

.. .. math::
..     \mathbf{A}[x] = (\mathbf{I} - \mathbf{B}\mathbf{C})[x]

.. We could implement two different types of strategies:

.. .. code-block:: python
    
..     class FusedStrategy(StrategyA):
..         def forward(data, x):
..             return compute_a_fused_kernel_I_minus_BC(data, x)

..     class ComposedAStrategy(StrategyA):
..         strategy_b : StrategyB
..         strategy_c : StrategyC 
..         def forward(data, x):
..             return x - strategy_b.forward(data, strategy_c.forward(data, x) )

.. and we can use all the machinery for the base operator to do e.g. solving, differentiating, etc.

.. .. admonition:: Why not inheritance?

..     Inheritance works, but you can't nest it without duplicating data or exploding your inheritance tree.



