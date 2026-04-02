import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, Tuple
from jax_sn.domain import CrossSectionData

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


def make_xs_test(material: str, n_groups: int, l_order: int) -> "CrossSectionData":
    """
    Create a physically reasonable CrossSectionData for testing purposes.
    
    The scattering matrix satisfies ``sum_{g'} sigma_s[0, g', g] <= sigma_t[g]``
    and is purely downscattering (no upscattering).  Higher Legendre moments decay
    as ``sigma_s[l] = sigma_s[0] * mu_bar**l`` with ``mu_bar = 0.3``.

    Parameters
    ----------
    material : str
        One of ``'fuel'``, ``'moderator'``, ``'absorber'``, ``'reflector'``,
        ``'void'``.
    n_groups : int
        Number of energy groups. Groups are indexed fast (0) → thermal (n_groups-1).
    l_order : int
        Number of scattering moments; the scattering array has shape
        ``(l_order, n_groups, n_groups)``.

    Material definitions
    --------------------
    fuel        sigma_t linearly 0.8→1.2, 50 % scatter, 65% within-group,
                35% to adjacent lower-energy group.
    moderator   sigma_t linearly 0.5→0.9, 90 % scatter, 30% within-group,
                70% to adjacent lower-energy group (strong thermalisation).
    absorber    sigma_t linearly 2.0→3.0, 10 % scatter, within-group only.
    reflector   sigma_t linearly 1.0→1.5, 80 % scatter, 70% within-group,
                30% to adjacent lower-energy group.
    void        sigma_t = 1e-4 (uniform), 50 % scatter, within-group only.
    """
    from jax_sn.domain import CrossSectionData

    configs = {
        "fuel":       (np.linspace(0.8, 1.2, n_groups), 0.50, 0.65, 0.35),
        "moderator":  (np.linspace(0.5, 0.9, n_groups), 0.90, 0.30, 0.70),
        "absorber":   (np.linspace(2.0, 3.0, n_groups), 0.10, 1.00, 0.00),
        "reflector":  (np.linspace(1.0, 1.5, n_groups), 0.80, 0.70, 0.30),
        "void":       (np.full(n_groups, 1e-4),          0.50, 1.00, 0.00),
    }
    if material not in configs:
        raise ValueError(f"Unknown material '{material}'. Choose from: {list(configs)}")

    sigma_t, scatter_frac, within_frac, down_frac = configs[material]

    scatter = np.zeros((max(l_order, 1), n_groups, n_groups))
    for g in range(n_groups):
        total_scatter = scatter_frac * sigma_t[g]
        scatter[0, g, g] += within_frac * total_scatter
        if g < n_groups - 1:
            scatter[0, g + 1, g] += down_frac * total_scatter
        else:
            scatter[0, g, g] += down_frac * total_scatter  # last group: no lower-energy group

    mu_bar = 0.3
    for l in range(1, max(l_order, 1)):
        scatter[l] = scatter[0] * (mu_bar ** l)

    return CrossSectionData(jnp.array(sigma_t), jnp.array(scatter[:l_order]))



def setup_layered_slab_mgop(
    dim: int,
    degree: int,
    tn_quadrature_set: int,
    region_sizes: List[float],
    region_resolutions: List[int],
    region_xs: List["CrossSectionData"],
    region_sources: Optional[List[float]] = None,
    transverse_specs: Optional[List[Tuple[float, int]]] = None,
    use_numpy: bool = False,
    **kwargs,
):
    """Create a multi-region slab domain with per-region cross-sections.

    Regions vary along the x-axis (inhomogeneity direction = 0).  For dim > 1
    the transverse dimensions (y, z) are uniform and specified via
    ``transverse_specs``.

    Used for testing and for benchmarking sweep implementations.

    Parameters
    ----------
    dim : int
        Spatial dimension (1, 2, or 3).
    degree : int
        Polynomial degree of the FE basis.
    tn_quadrature_set : int
        T_N quadrature order.
    region_sizes : list of float
        Physical size of each region along x.
    region_resolutions : list of int
        Number of elements in each region along x.
    region_xs : list of CrossSectionData
        Cross-section data for each region.  ``make_xs`` is a convenient
        helper for creating physically reasonable values.
    region_sources : list of float, optional
        Volumetric source strength per region.  Defaults to 1.0 in the first
        region and 0.0 elsewhere.
    transverse_specs : list of (float, int), optional
        ``[(size, resolution), ...]``, one entry per transverse dimension
        (y for dim=2; y and z for dim=3).  Defaults to ``[(1.0, 5)]`` per
        transverse dimension.
    use_numpy : bool
        Use numpy-based domain construction (faster for large meshes).
    **kwargs
        Passed through to ``create_multi_group_operator``.

    Returns
    -------
    mgop : MultiGroupOperator

    Examples
    --------
    >>> fuel = make_xs('fuel', n_groups=5, l_order=2)
    >>> mod  = make_xs('moderator', n_groups=5, l_order=2)
    >>> abs_ = make_xs('absorber', n_groups=5, l_order=2)
    >>> mgop = setup_layered_slab_mgop(
    ...     dim=2, degree=1, tn_quadrature_set=3,
    ...     region_sizes=[2.0, 4.0, 2.0],
    ...     region_resolutions=[10, 20, 10],
    ...     region_xs=[fuel, mod, abs_],
    ...     transverse_specs=[(3.0, 8)],
    ... )
    """
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
    from jax_sn.solution_domain import SolutionDomain
    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.quadrature_set import create_tn_quadrature_set, QuadratureSetReduced

    n_regions = len(region_sizes)
    assert len(region_resolutions) == n_regions, "region_resolutions must have the same length as region_sizes"
    assert len(region_xs) == n_regions, "region_xs must have the same length as region_sizes"

    if region_sources is None:
        region_sources = [1.0] + [0.0] * (n_regions - 1)
    assert len(region_sources) == n_regions, "region_sources must have the same length as region_sizes"

    if transverse_specs is None:
        transverse_specs = [(1.0, 5)] * (dim - 1)
    assert len(transverse_specs) == dim - 1, "transverse_specs must have dim-1 entries"

    x_specs = list(zip(region_sizes, region_resolutions))
    region_size_res = [x_specs] + [list(transverse_specs[i:i+1]) for i in range(dim - 1)]

    vertices, connectivity, layered_xs, _ = create_cartesian_setup(
        dim, region_size_res, region_xs, region_sources, inhomogeneity_direction=0
    )

    element = BasixLagrangianSimplex(degree, dim)
    if use_numpy:
        domain = Domain.from_mesh_and_cross_sections_numpy((vertices, connectivity), element.face_template, layered_xs)
    else:
        domain = Domain.from_mesh_and_cross_sections((vertices, connectivity), element.face_template, layered_xs)

    solution_domain = SolutionDomain.from_element_and_domain(element, domain)
    tn_qs = QuadratureSetReduced.from_quadrature_set(create_tn_quadrature_set(tn_quadrature_set), n_dim=dim)
    return create_multi_group_operator(solution_domain, tn_qs, **kwargs)