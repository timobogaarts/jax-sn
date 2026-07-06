
import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import pytest

def setup_anisotropic(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order):

    from jax_sn.domain.geometric_domain import CrossSectionData
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex
    
    from jax_sn.quadrature_set import create_tn_quadrature_set

    water = CrossSectionData(jnp.array([15.2090, 24.4140]), jnp.array([[[5.5069, 0.0], 
                                                                        [6.6227, 11.4080]], 
                                                                        [[1.158567, 0.0],
                                                                         [1.1234767, 2.4733]]]))
    
    iron = CrossSectionData(jnp.array([30.3110, 26.8760]), jnp.array([[[18.25, 0.0], 
                                                                       [6.45294, 21.3570]], 
                                                                      [[4.363, 0.0],
                                                                       [-0.003508, 2.3457]]]))
    void = CrossSectionData.constant(1e-10, 2, 1)
    other_dims   = [[(d_other_dim, n_other_dim)]]*(dim - 1)

    region_sizes = [0.2, 0.1,     0.5,   0.2,  0.1,  0.1,    0.1]
    xs_regions   = [water, water, water, iron, void, water , void]

    region_size_res = [(size, n_elem) for size, n_elem in zip(region_sizes, n_elem_regions)]

    source_eg0 = [0.0, 9.7702e4, 0, 0,0,0,0]

    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [region_size_res] + other_dims, xs_regions, source_eg0, 0)
    element_type                         = BasixLagrangianSimplex(degree, dim)
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
     # convert source to Source class
    sources = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    sources_eg0 = sources 
    sources_eg1 = jnp.where(sources_eg0 > 0, 4.5441e5, 0.0)



    return solution_domain, (sources_eg0, sources_eg1), create_tn_quadrature_set(tn_order)


def _get_expected_values_reference(*args_ani):
    # Frozen reference: interpolated scalar flux (see tests/reference_data/_generate.py)
    from reference_data import load, key_anisotropic
    dim = args_ani[0]
    return load(key_anisotropic(dim))['phi']

def _get_anisotropic_args(dim, degree):
    n_elem_regions = [4, 2, 10, 4, 2, 2, 2]
    n_elem_regions = [6, 3, 15, 6, 3, 3, 3]
    d_other_dim = 5.0
    n_other_dim = 1
    tn_order = 3
    return (dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order)

def _interp_x():
    return jnp.linspace(0 + 1e-5, 1.3 - 1e-5, 1000)

def _setup_anisotropic_scattering_args(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order):
    from jax_sn.operators.scatter_operator import ScatterOperator
    from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, VmapForward, VmapSweep, VmapAngles
    from jax_sn.operators.mass_operator import MassOperator
    from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
    from jax_sn.operators.transport_operator import BasicSweepData
    from jax_sn.operators.single_group_operator import SingleGroupOperator    
            
    solution_domain, sources, quadrature_set = setup_anisotropic(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order) 

    sweep_data = BasicSweepData.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAngles(TransportOperator(solution_domain.solution_domain_data, sweep_data, quadrature_set.angles, 0, VmapForward(), VmapSweep()), VmapAngles())
    mass_operator             = MassOperator.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = 1, elements_basis_shape=solution_domain.elements_basis_shape)
    source_total              = jnp.broadcast_to(sources[0][None, :, :], (quadrature_set.angles.shape[0],) + sources[0].shape) # source for all angles
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = 1, elements_basis_shape=solution_domain.elements_basis_shape)
    Aop = SingleGroupOperator.from_operators(moment_operator, transport_operator, mass_operator, discrete_operator, quadrature_set)
    return Aop, source_total, solution_domain

@pytest.fixture(params=[1,2,3], scope='session')
def _setup_anisotropic_scattering(request):    
    degree = 2 
    dim    = request.param
    
    ani_args = _get_anisotropic_args(dim, degree)
    return *_setup_anisotropic_scattering_args(*ani_args), ani_args

def test_anisotropic_scattering_fixed_point(_setup_anisotropic_scattering, expected_value_function = _get_expected_values_reference):

    Aop, source_total, solution_domain, ani_args = _setup_anisotropic_scattering

    rhs = Aop.moment_operator.mv(Aop.transport_operator.inv(Aop.mass_operator.mv(source_total)))
    n_fixed_point = 200
    xk = jnp.zeros_like(rhs)
    
    for i in range(n_fixed_point):
        xk = xk - Aop.mv(xk) + rhs

    x = _interp_x()    
    
    dim = ani_args[0]
    d_other = ani_args[-3]
    xyz = jnp.stack([x] + [jnp.full_like(x, d_other / 2)] * (dim-1), axis = -1)

    phi_jsn = solution_domain.interpolate(xyz, xk[0]) 
    phi_reference  = expected_value_function(*ani_args)

    rtol = [1e-10, 1e-10, 3e-2][dim-1] # tetrahedral connectivity is different; so we cannot expected numerically same results in this non-converged regime!

    onp.testing.assert_allclose(phi_jsn, phi_reference, rtol = rtol)
