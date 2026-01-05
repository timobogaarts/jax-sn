
import numpy as onp 
import jax.numpy as jnp
def _check_adjoint_of_operator(operator):
    r'''
    Checks the adjoint identity of the operator for the case that it produces a non-batched output:

    <Ax, y> = <x, A^Ty>

    
    '''

    onp.random.seed(5012026)


    x = jnp.array(onp.random.rand(*operator.in_structure().shape))
    y = jnp.array(onp.random.rand(*operator.out_structure().shape))  
    Ox = operator.mv(x)
    OTy = operator.transpose().mv(y)
    lhs = onp.sum(Ox * y)
    rhs = onp.sum(x * OTy)
    onp.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

    # Idempotence of transpose
    OTT = operator.transpose().transpose()
    x2 = jnp.array(onp.random.rand(*operator.in_structure().shape))
    OTTx2 = OTT.mv(x2)
    Ox2   = operator.mv(x2)
    onp.testing.assert_allclose(OTTx2, Ox2, rtol=1e-12, atol=1e-12)

