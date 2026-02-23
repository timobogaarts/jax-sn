.. _linear_systems_differentation:

Linear Systems and Differentation
--------------------------------------

A finite-dimensional linear system can be written as:

.. math::
    \mathbf{A} x = b

where :math:`\mathbf{A}` is a linear operator, :math:`x` is the unknown solution, and :math:`b` is the right-hand side. Note that in the case of the neutron transport equation,
:math:`x` is a very large vector (since it contains the solution at all discretised points in six-dimensional phase space, often billions of DOF). Therefore, 
assembling the full representation of the linear operator :math:`\mathbf{A}` will be extremely memory intensive (most likely impossible).

Direct solvers will require assembly of the full matrix representation of :math:`\mathbf{A}`, which will not be feasible in most cases. 
Luckily, an efficient class of solvers exist, namely, materix-free-iterative solvers (e.g. GMRES or BiCGSTAB) exist, that only require matrix-vector products (i.e. :math:`\mathbf{A} x`). These do not 
require assembly of the full matrix, and thus lend itself well to the solution of large linear systems such as the one obtained from discretisation of the neutron transport equation.

Thus, our formulation of the neutron transport equation should lend itself to efficient implementation of matrix-vector products and we do not even have to explicitly provide 
a way to assemble it at all! Therefore, in the next sections, we will gradually switch every discretisation to an operator form instead of an explicit formulation of each 
equation.


Differentation
~~~~~~~~~~~~~~~

We are often not only interested in solving the neutron transport equation, but also in obtaining derivatives of a function solution of this equation with respect to some parameters (e.g., 
how the tritium breeding ratio in fusion blankets changes with respect to material composition or even geometry). This is especially common for design optimization: we often want to opimize 
using gradient-based methods, which require the gradient of an objective function with respect to the design parameters.

In other words, for a particular parameter vector (size :math:`n`) :math:`\theta`, and the linear system of size :math:`p \times p`:

.. math::
    \mathbf{A}(\theta) x = b(\theta) 

and a set of :math:`m` functions:

.. math::
    f(\theta; x(\theta)) = f(\theta; \mathbf{A}(\theta)^{-1} b(\theta)) \equiv f(\theta)

we are interested in obtaining the Jacobian of :math:`f` with respect to the parameter vector :math:`\theta`:

.. math::
    \frac{df}{d\theta} = \begin{bmatrix} \frac{df_1}{d\theta_1} & \cdots & \frac{df_1}{d\theta_n} \\ \vdots & \ddots & \vdots \\ \frac{df_m}{d\theta_1} & \cdots & \frac{df_m}{d\theta_n}  \end{bmatrix}


We can write this total derivative as:

.. math::
    \frac{df}{d\theta} = \frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial x} \cdot \frac{dx}{d\theta} 

This first term is easy to compute, since it only requires the explicit dependence of :math:`f` on :math:`\theta`. 
For the second term we can use :math:`\partial_\theta(A^{-1} A) = 0` to obtain:

.. math::
    \frac{dx}{d\theta} = \frac{d(\mathbf{A}^{-1} b)}{d\theta} = \mathbf{A}^{-1} \left[ - \frac{\partial \mathbf{A}}{\partial \theta} \cdot \mathbf{A}^{-1} b + \frac{\partial b}{\partial \theta}\right] 

such that 

.. math::

   \underbrace{\frac{df}{d\theta}
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{m\times n}
   =
   \underbrace{\frac{\partial f}{\partial \theta}
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{m\times n}
   +
   \underbrace{\frac{\partial f}{\partial x}
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{m\times p}
   \cdot
   \underbrace{\mathbf A^{-1}
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{p\times p}
   \cdot
   \underbrace{\left[
   - \frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b
   + \frac{\partial b}{\partial \theta}
   \right]}_{p\times n}

If we just follow this equation left to right (direct method), this gives the Jacobian, at the cost of solving :math:`n` linear systems (since the term in square brackets is :math:`p \times n`, we need to solve one linear system for each column) in addition to the 
first :math:`\mathbf{A}^{-1} b` solve.

However, often, we are interested in only a very small number of functions (e.g. some objective function for an optimisation and a few constraints) and a large number of parameters (e.g. the position of all elements in the geometry). Then, :math:`m \ll n`, but we still need to solve :math:`n` linear systems, which can be very expensive.
Alternatively, we can use the following identity

.. math::

   \biggl[
   \underbrace{A^{-T}
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{p\times p}
   \underbrace{\left(\frac{\partial f}{\partial x}\right)^T
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{p\times m}
   \biggr]^T
   =
   \frac{\partial f}{\partial x}\cdot \mathbf A^{-1}

to rewrite the Jacobian as:

.. math::

   \frac{df}{d\theta}
   =
   \frac{\partial f}{\partial \theta}
   +
   \biggl[
   \underbrace{A^{-T}
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{p\times p}
   \underbrace{\left(\frac{\partial f}{\partial x}\right)^T
   \vphantom{\left[-\frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b+\frac{\partial b}{\partial \theta}\right]}}_{p\times m}
   \biggr]^T
   \cdot
   \underbrace{\left[
   - \frac{\partial \mathbf A}{\partial \theta}\cdot \mathbf A^{-1} b
   + \frac{\partial b}{\partial \theta}
   \right]}_{p\times n}

Computing the term in the transposed brackets only takes :math:`m` linear solves of the transpose operator :math:`A^T`, so if :math:`m` is one, it does not matter how much parameters :math:`n` we have,
this will still only be one additional linear solve! This so-called adjoint method thus allows us to efficiently compute the Jacobian of :math:`f` with respect to :math:`\theta` even when we have a large number of parameters, as long as we have a small number of functions of interest.
See the table below for a comparison of the direct and adjoint method.

.. list-table:: Comparison of Direct vs Adjoint Methods
    :header-rows: 1
    :widths: 30 70 70

    * - Method
      - Linear Solves
      - Use
    * - Direct
      - :math:`(n+1)` solves of :math:`\mathbf{A}`
      - Few parameters, many functions
    * - Adjoint
      - :math:`m` solves of :math:`\mathbf{A}^T` and one solve of :math:`\mathbf{A}`
      - Many parameters, few functions
  
Now, we can note that the above doesn't depend on whether :math:`\mathbf{A}` is a matrix at all; all we used was invertibility and the existence of a transpose.
Therefore, the adjoint and direct methods are still applicable for a formulation with a neutron transport linear operator that only provides a matrix-vector product, as long as we can also provide the matrix-vector product for the transpose operator.

Furthermore, as shown here, if we want to have the total derivatives of :math:`f` with respect to :math:`\theta`, we need to be able to provide the partial derivatives of :math:`f` with respect to :math:`\theta` and :math:`x`, as well as efficient partial derivatives of :math:`\mathbf{A}` and :math:`b` with respect to :math:`\theta`.

Also, we do not need to differentiate through the linear solver itself at all, so this does not have to be a differentiable operation (which is good; since many iterative solvers are not differentiable at all!). 

Finally, even these relations can be automated and even nested, for example using `Lineax <https://docs.kidger.site/lineax/>`__, so the additional implementation cost of these methods is minimal, as long as 
we have a solver that can handle the linear system and its transpose and we can provide the necessary partial derivatives.

Conclusion
~~~~~~~~~~~~~~~

For our neutron transport equation to be efficiently solvable, we have to implement it as a linear operator :math:`A` that only has to provide matrix-vector products. Furthermore,
if we want our solver to be efficiently differentiable, we only need to ensure that the application of the linear operator :math:`A` is differentiable and its transpose is available and differentiable.
Then, any kind of external solver can be used.





    
