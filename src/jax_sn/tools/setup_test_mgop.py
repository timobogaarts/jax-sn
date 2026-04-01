import jax 
import jax.numpy as jnp



def setup_slab_mgop(dim : int, degree : int, resolution : int, tn_quadrature_set : int, n_groups : int, l_order : int, use_numpy : bool = False,
                    **kwargs):
    '''
    Convenience function to quickly create a simple slab domain. This can be used to test performance or memory usage of the multigroup operator.

    Parameters
    ----------
    dim : int
        The spatial dimension of the problem.

    degree : int
        The degree of the finite element basis functions.
    resolution : int
        The number of elements in each spatial dimension.
    tn_quadrature_set : int
        The order of the quadrature set to use for the angular discretization.
    n_groups : int
        The number of energy groups.
    l_order : int
        The order of the Legendre expansion for the scattering cross section (0 for isotropic scattering, 1 for linearly anisotropic scattering, etc.)
    **kwargs : dict
        Additional keyword arguments to pass to the multi-group operator creation function (create_multi_group_operator). Used for specifying specific operators.

    Returns
    -------
    mgop : MultiGroupOperator
        The created multi-group operator for the specified slab domain.
    
    '''

    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import CrossSectionData, Domain
    from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
    from jax_sn.solution_domain import SolutionDomain
    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.quadrature_set import create_tn_quadrature_set, QuadratureSetReduced

    cross_section_data = CrossSectionData(jnp.arange(1, n_groups + 1) / n_groups,  jnp.ones((l_order, n_groups, n_groups)) * 0.9)
    vertices, connectivity, layered_xs, layered_source = create_cartesian_setup(dim, [[(1.0, resolution)]]*dim, [cross_section_data], [1.0], 0)

    element = BasixLagrangianSimplex(degree, dim)
    if use_numpy:
        domain = Domain.from_mesh_and_cross_sections_numpy((vertices, connectivity), element.face_template, layered_xs)
    else:
        domain = Domain.from_mesh_and_cross_sections((vertices, connectivity), element.face_template, layered_xs)

    
    solution_domain = SolutionDomain.from_element_and_domain(element, domain)

    tn_quadrature_set = QuadratureSetReduced.from_quadrature_set(create_tn_quadrature_set(tn_quadrature_set), n_dim = dim)

    mgop = create_multi_group_operator(solution_domain, tn_quadrature_set, **kwargs)
    return mgop