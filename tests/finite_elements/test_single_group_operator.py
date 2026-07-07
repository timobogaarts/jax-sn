
import jax
jax.config.update("jax_enable_x64", True)

from finite_elements.operator_util import _check_adjoint_of_operator


def test_transpose_single_group_operator(bb_sgop):
    # Euclidean adjointness of the assembled single-group operator (group 0 of the BB problem).
    sgop, *_ = bb_sgop
    _check_adjoint_of_operator(sgop)
