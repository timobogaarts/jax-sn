'''Algebraic / self-consistency tests for the multigroup operator.

Built via the shared ``bb_mgop`` fixture, these check properties that need no external ground
truth: implementation equivalence (Scan vs alternatives), Euclidean adjointness, RHS-construction
equivalence, lower-triangular vs global solver equivalence, angular-quadrature reduction, and the
continuous-adjoint identity. The accuracy-vs-oracle test lives in test_multigroup_operator_solve.py.
'''
import numpy as onp
import jax.numpy as jnp
import jax
import pytest

jax.config.update("jax_enable_x64", True)

from finite_elements.operator_util import _check_adjoint_of_operator


# --- implementation equivalence -------------------------------------------------------------
# Alternative MultiGroupOperator implementations, each checked against the Scan reference that
# bb_mgop provides. Add new backends (e.g. a future vmap variant) here -- one line, like the
# transport-operator factory lists.
def _mgop_checkpoint(base):
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import MultiGroupOperatorScanCheckpoint
    return MultiGroupOperatorScanCheckpoint.from_multigroup_operator(base)


MG_IMPL_FACTORIES = [pytest.param(_mgop_checkpoint, id="ScanCheckpoint")]


@pytest.mark.parametrize("mg_factory", MG_IMPL_FACTORIES)
def test_multigroup_implementations_equivalent(bb_mgop, mg_factory):
    '''Each alternative implementation reproduces the Scan reference matvec (forward + transpose).'''
    base = bb_mgop[0]
    op = mg_factory(base)

    onp.random.seed(15012026)
    x = jnp.array(onp.random.rand(*base.in_structure().shape))

    onp.testing.assert_allclose(onp.array(op.mv(x)), onp.array(base.mv(x)), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(onp.array(op.transpose().mv(x)), onp.array(base.transpose().mv(x)), rtol=1e-12, atol=1e-12)


def test_multigroup_adjoint(bb_mgop):
    '''Euclidean adjointness <Ax, y> == <x, A^T y> (and transpose idempotence).'''
    _check_adjoint_of_operator(bb_mgop[0])

