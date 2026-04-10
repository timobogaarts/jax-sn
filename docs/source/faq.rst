Frequently Asked Questions
===========================


My gradients are wrong!
---------------------------------
This can happen if your solve tolerances are not tight enough.

One dangerous path is a loose absolute tolerance: when backpropagating, a small magnitude cotangent (i.e. the one passed to the adjoint solve)
can be much much smaller than the forward source. Therefore, the tolerance can immediately exit the adjoint solve, which 
might not be converged at all!

