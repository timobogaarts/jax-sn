
import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import pytest

from .test_anisotropic_scattering import _get_anisotropic_args, _setup_anisotropic_scattering_args
from jax_sn.operators.single_group_operator import SingleGroupOperator


def test_transpose_single_group_operator():
    ani_args = _get_anisotropic_args(2,3)



    Aop, source_total, solution_domain = _setup_anisotropic_scattering_args(*ani_args)

    from finite_elements.operator_util import _check_adjoint_of_operator

    _check_adjoint_of_operator(Aop)