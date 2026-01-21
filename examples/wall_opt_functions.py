import jax.numpy as jnp

from jax_sbgeom.flux_surfaces import FluxSurface, ToroidalExtent
import jax_sbgeom as jsb
import equinox as eqx
import jax
import jax_sn
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan

class SOLSettings(eqx.Module):
    flux_surface    : FluxSurface
    toroidal_extent : ToroidalExtent
    n_theta         : int
    n_phi           : int
    n_sol           : int


def setup_plasma_mgop(flux_surface : FluxSurface, tor_extent : ToroidalExtent, n_plasma : int, n_sol : int, sol_edge_d : float, n_theta : int , n_phi : int, source_kwargs : dict, degree : int, batch_size : int):
    from jax_sn.solution_domain import BasixLagrangianSimplex
    from jax_sn.domain import GeometricDomain, CrossSectionsIndexed, CrossSectionData, TopologicalSortedAngle
    s_mesh = jnp.concatenate([jnp.linspace(0.0, 1.0, n_plasma)**2, jnp.linspace(1.0, 1.0 + sol_edge_d, n_sol+1)[1:]])
    mesh_plasma = jsb.flux_surfaces.mesh_tetrahedra(flux_surface, s_mesh, tor_extent, n_theta, n_phi)

    # Source computation
    s_values, theta_values, phi_values = jsb.flux_surfaces.flux_surface_meshing._mesh_tetrahedra_points_coordinates(flux_surface, s_mesh, True, *tor_extent, n_theta, n_phi)
    fs_values = jnp.nan_to_num(jsb.interfaces.plasma.flux_surface_reaction_rates_simple(s_values, **source_kwargs), nan = 0.0)
    fs_values_elem = jnp.mean(fs_values[mesh_plasma[1]], axis=-1)
    element                   = jax_sn.solution_domain.BasixLagrangianSimplex(degree,3)

    geometric_domain          = jax_sn.domain.GeometricDomain.from_mesh(mesh_plasma, element.face_template)
    
    batched_xs                = jax.tree_util.tree_map(lambda *x : jnp.stack(x), CrossSectionData.zero(1,0))
    cross_sections_idx        = CrossSectionsIndexed(jnp.arange(0, geometric_domain.n_elements), batched_xs)

    domain                    = jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(geometric_domain, cross_sections_idx)
    quadrature_set            = jax_sn.quadrature_set.create_tn_quadrature_set(3)
    
    solution_domain           = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(element, domain)
    source_eg0                = jnp.repeat(fs_values_elem, repeats=solution_domain.n_basis, axis=-1).reshape(domain.n_elements, solution_domain.n_basis)
    source_eg0_all_angles     = jnp.stack([source_eg0]*quadrature_set.n_angles, axis=0)
    topological_sorted_angles = TopologicalSortedAngle.from_domain_quadrature_set(geometric_domain, quadrature_set, 100)

    from jax_sn.operators.transport_operator import TransportOperatorMultipleAnglesVmap, TransportOperatorBatchedPallas
    from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
    batch_size         = 100
    transport_operator = TransportOperatorMultipleAnglesVmap(TransportOperatorBatchedPallas(solution_domain.solution_domain_data, topological_sorted_angles,0, batch_size))
    mass_operator      = jax_sn.operators.mass_operator.MassOperatorVmap.from_transport_operator(transport_operator)
    moment_operator    = MomentOperator.from_quadrature_set(quadrature_set, l_max=0, elements_basis_shape= solution_domain.elements_basis_shape)
    discrete_operator  = DiscreteOperator.from_quadrature_set(quadrature_set, l_max=0, elements_basis_shape= solution_domain.elements_basis_shape)
    gather_scatter_operator = jax_sn.operators.scatter_operator.GatherScatterOperatorBroadcasted(solution_domain.domain.cross_sections_indexed)
    from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan

    MGOP = MultiGroupOperatorScan(moment_operator, discrete_operator, transport_operator, mass_operator,gather_scatter_operator, 1 )

    return MGOP, source_eg0_all_angles, SOLSettings(flux_surface, tor_extent, n_theta, n_phi, n_sol)

def plot_resulting_d(d, tor_extent):
    import matplotlib.pyplot as plt
    theta = jnp.linspace(0, 2 * jnp.pi, 200)
    phi   = jnp.linspace(tor_extent.start, tor_extent.end, 200)
    theta_mg, phi_mg = jnp.meshgrid(theta, phi, indexing='ij')
    d_values = d.interpolate_d(theta_mg, phi_mg, tor_extent)
    plt.figure(figsize=(8,6))
    plt.contourf(phi_mg, theta_mg, d_values, levels=50)
    plt.colorbar(label='d (m)')
    plt.xlabel('Phi (rad)')
    plt.ylabel('Theta (rad)')
    plt.title('Wall distance d(theta, phi)')
    plt.show()


def interpolate_theta_phi_d(parameters, theta, phi, toroidal_extent):
    """
    parameters: array of shape (n_phi, 1 + 2*K)
                columns: [a0, a1, b1, a2, b2, ..., aK, bK]
    theta: scalar or array
    phi: scalar or array
    toroidal_extent: (phi_min, phi_max)
    """

    phi_min, phi_max = (toroidal_extent.start, toroidal_extent.end)
    n_phi, n_coeff = parameters.shape

    # Infer number of Fourier modes
    K = (n_coeff - 1) // 2

    # --- Map phi to grid coordinate ---
    # Continuous grid index in [0, n_phi-1]
    t = (phi - phi_min) / (phi_max - phi_min) * (n_phi - 1)

    i0 = jnp.floor(t).astype(int)
    i1 = jnp.clip(i0 + 1, 0, n_phi - 1)

    w = t - i0

    # Clamp for safety
    i0 = jnp.clip(i0, 0, n_phi - 1)
    i1 = jnp.clip(i1, 0, n_phi - 1)

    # --- Interpolate Fourier coefficients in φ ---
    coeff0 = parameters[i0]    # shape (..., 1+2K)
    coeff1 = parameters[i1]

    coeff = (1 - w)[..., None] * coeff0 + w[..., None] * coeff1

    # Split coefficients
    a0 = coeff[..., 0]
    ak = coeff[..., 1::2]   # shape (..., K)
    bk = coeff[..., 2::2]   # shape (..., K)

    # --- Reconstruct Fourier series in θ ---
    k = jnp.arange(1, K + 1)

    # Broadcasting handles vector θ automatically
    coskt = jnp.cos(theta[..., None] * k)
    sinkt = jnp.sin(theta[..., None] * k)

    return a0 + jnp.sum(ak * coskt + bk * sinkt, axis=-1)

from abc import abstractmethod

class ParametrisedDistance(eqx.Module):

    @abstractmethod
    def interpolate_d(self, theta, phi, toroidal_extent):
        ...


@eqx.filter_jit
def create_sol_theta(d_grid : ParametrisedDistance, sol_settings : SOLSettings) -> jnp.ndarray:    
    flux_surface                = sol_settings.flux_surface
    toroidal_extent             = sol_settings.toroidal_extent
    n_theta                     = sol_settings.n_theta
    n_phi                       = sol_settings.n_phi
    n_sol                       = sol_settings.n_sol

    theta_mg, phi_mg             = jnp.meshgrid(jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False), jnp.linspace(toroidal_extent.start, toroidal_extent.end, n_phi, endpoint=True), indexing='ij')
    
    d_edge = d_grid.interpolate_d(theta_mg, phi_mg, toroidal_extent)

    
    s_linspace                   = jnp.linspace(0, 1, n_sol +1)[1:]
    locations_sol                = flux_surface.cartesian_position(1.0 + d_edge[None, :, :] * s_linspace[:, None, None], theta_mg[ :, :], phi_mg[:, :])
    return locations_sol


@eqx.filter_jit
def recreate_mesh(d_grid : ParametrisedDistance, sol_settings : SOLSettings, vertices_lcfs : jnp.ndarray):    
    return jnp.concatenate([vertices_lcfs, create_sol_theta(d_grid, sol_settings).reshape(-1,3)], axis=0)

def compute_flux(source_all_angles : jnp.ndarray, moment_operator, transport_operator, mass_operator):
    return moment_operator.mv(transport_operator.inv(mass_operator.mv(source_all_angles)))

def block_flux(fluxes : jnp.ndarray, Ij : jnp.ndarray, sol_settings : SOLSettings):
    fluxes_ll = jnp.einsum("i,ji->j", Ij, fluxes[0, -sol_settings.n_theta * (sol_settings.n_phi - 1) * 6:, :]) #[n_theta * n_phi* 6]    
    return jnp.mean(fluxes_ll.reshape( sol_settings.n_phi -1 ,sol_settings.n_theta, 6), axis=-1)



def block_flux_MGOP(MGOP : MultiGroupOperatorScan, source_eg0_all_angles, sol_settings : SOLSettings):
    phi_flux = MGOP.moment_operator.mv(MGOP.transport_operator.inv(MGOP.mass_operator.mv(source_eg0_all_angles)))
    Ij = MGOP.transport_operator.transport_operator.solution_domain_data.fem_matrices.integration_coeff * 6.0
    return block_flux(phi_flux, Ij, sol_settings)

def d_grid_to_domain(d_grid, MGOP : MultiGroupOperatorScan, sol_settings : SOLSettings):
    vertices_lcfs = MGOP.transport_operator.transport_operator.solution_domain_data.domain.vertices[: - sol_settings.n_theta * sol_settings.n_phi * sol_settings.n_sol, :]     
    vertices_new  = recreate_mesh(d_grid, sol_settings, vertices_lcfs)
    return vertices_new

from jax_sn.parametrisation.multigroup_parametrisation import MultigroupOperatorParametrisedVerticesXS, VerticesXSTheta


def d_grid_to_MGOP(d_grid, MGOP : MultiGroupOperatorScan, sol_settings : SOLSettings) -> MultiGroupOperatorScan:
    vertices_new = d_grid_to_domain(d_grid, MGOP, sol_settings)
    theta_base     = MultigroupOperatorParametrisedVerticesXS.theta_base(MGOP)
    theta_base_new = VerticesXSTheta(vertices_new, theta_base.cross_section_data)
    MGOP_new       = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(MGOP, theta_base_new)
    return MGOP_new

@eqx.filter_jit
def d_grid_to_values(d_grid : ParametrisedDistance, MGOP : MultiGroupOperatorScan, source_eg0_all_angles : jnp.ndarray, sol_settings : jnp.ndarray) -> jnp.ndarray:
    MGOP_new = d_grid_to_MGOP(d_grid, MGOP, sol_settings)

    phi_fluxes     = MGOP_new.moment_operator.mv(MGOP_new.transport_operator.inv(MGOP_new.mass_operator.mv(source_eg0_all_angles)))
    Ij             = MGOP_new.transport_operator.transport_operator.solution_domain_data.fem_matrices.integration_coeff * 6.0 # V/(det(Jv)) = 6.0 for tetrahedral simplices

    return block_flux(phi_fluxes, Ij, sol_settings)    

@eqx.filter_jit
def d_grid_to_flux_elements(d_grid : ParametrisedDistance, MGOP : MultiGroupOperatorScan, source_eg0_all_angles : jnp.ndarray, sol_settings : jnp.ndarray) -> jnp.ndarray:
    MGOP_new = d_grid_to_MGOP(d_grid, MGOP, sol_settings)

    phi_fluxes     = MGOP_new.moment_operator.mv(MGOP_new.transport_operator.inv(MGOP_new.mass_operator.mv(source_eg0_all_angles)))
    Ij             = MGOP_new.transport_operator.transport_operator.solution_domain_data.fem_matrices.integration_coeff * 6.0 # V/(det(Jv)) = 6.0 for tetrahedral simplices

    return jnp.einsum("i,ji->j", Ij, phi_fluxes[0,:, :]) #[n_theta * n_phi* 6]

