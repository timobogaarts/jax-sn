# %%
import jax.numpy as jnp
import jax_sn
import jax
import jax_sbgeom as jsb
jax.config.update("jax_enable_x64", True)
import pyvista as pv
from typing import Iterable
from jax_sn.domain.cross_sections import CrossSectionData, CrossSectionsIndexed
import numpy as onp 
import matplotlib.pyplot as plt
import equinox as eqx


TWOD_OPT_DATA_DIRECTORY = "/home/tbogaarts/blanket_optimizations/twod_optimizations/data/"
# %%
def run_optimization(phi, theta_inboard, sigma_inboard):
    vmec_file = [TWOD_OPT_DATA_DIRECTORY + "helias5_vmec.nc4"][0]
    fs_base = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(vmec_file)
    # %%
    d_edge_norm = 1.8
    flux_surface_extended = jsb.flux_surfaces.create_extended_flux_surface_d_interp_equal_arclength(fs_base, d_edge_norm, 70, 80, 100)

    # %%
    dim = 2
    degree = 4
    tn_order = 3
    n_elem_regions = (10, 1, 1, 10, 5, 1, 10, 2)
    groups_list = (14.1e6, 10e6, 3e6, 1e6, 1e4, 300, 10,0)

    d_regions_base = (1.2, 0.002, 0.025, 0.5, 0.385, 0.06, 0.2, 0.06)
    l_order = 4
    source_kwargs = {
        "n0"    : 2.4e20,
        "Ti0"   : 15.2,
        "nalpha": 0.35,
        "Tialpha": 1.2,
    }
    element = jax_sn.solution_domain.BasixLagrangianSimplex(degree, dim)

    # %%
    import pickle


    try:
        with open(TWOD_OPT_DATA_DIRECTORY + "new_bb_data_dict.pkl", "rb") as f:
            new_bb_data_dict = pickle.load(f)
        with open(TWOD_OPT_DATA_DIRECTORY + "new_aux_data_dict.pkl", "rb") as f:
            new_aux_data_dict = pickle.load(f)
        with open(TWOD_OPT_DATA_DIRECTORY + "tally_result_dict.pkl", "rb") as f:
            tally_result_dict = pickle.load(f)
    except:
        from breeding_blanket_setup import _get_hcpb_data_openmc    
        new_bb_data_dict, new_aux_data_dict, tally_result_dict = _get_hcpb_data_openmc(d_regions_base, l_order, 1e5, groups_list)

    # %%    
    n_theta          = 50

    # %%
    from breeding_blanket_setup import _get_hcpb_data_openmc, setup_mgop
    MGOP_EA, solution_domain_ea, source_all_angles_ea = setup_mgop(d_regions_base, phi, n_theta, n_elem_regions, tn_order, flux_surface_extended, source_kwargs, new_bb_data_dict, element, normalisation_distance = d_edge_norm)

    # %%
    from jax_sn.operators.multi_group_operator.lower_triangular_solver import MultiGroupLowerTriangularBlockSolver
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
    from breeding_blanket_setup import line_interpolate, plot_domain_with_lines_grid
    import lineax as lx
    mgsolver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(rtol = 1e-10, atol=1e-10, max_steps = 250))


    # %%
    from jax_sbgeom.jax_utils.splines import bspline, periodic_bspline, greville_abscissa_periodic_bspline

    # %%
    from typing import Tuple

    class BZShield2DParametrisation(eqx.Module):
        c_bz     : jnp.ndarray
        c_shield : jnp.ndarray

        @classmethod
        def from_settings(cls, n_ctrl, settings : "BZShield2DSettings"):

            c_bz = jnp.full(n_ctrl, settings.d_regions_base[3])
            c_shield = jnp.full(n_ctrl,settings.d_regions_base[-2])
            return cls(c_bz, c_shield)

    class BZShield2DSettings(eqx.Module):
        k_spline           : int
        n_theta_blanket    : int
        d_regions_base     : Tuple[float]
        n_elem_regions     : Tuple[int]
        phi                : float
        normalise_distance : float
        s_power_sampling   : float 
        flux_surface_extended : jsb.flux_surfaces.FluxSurface    

    @eqx.filter_jit
    def _get_flux_surface_coordinates(param : BZShield2DParametrisation, settings : BZShield2DSettings):

        theta_single_layer = jnp.linspace(0, 2*jnp.pi, settings.n_theta_blanket, endpoint=False)    
        d_regions_before   = jnp.array([jnp.full_like(theta_single_layer, d) for d in settings.d_regions_base[0:3]])
        d_bz               = periodic_bspline(theta_single_layer, param.c_bz, settings.k_spline)
        d_regions_middle   = jnp.array([jnp.full_like(theta_single_layer, d) for d in settings.d_regions_base[4:-2]])
        d_shield           = periodic_bspline(theta_single_layer, param.c_shield, settings.k_spline)
        d_regions_after    = jnp.array([jnp.full_like(theta_single_layer, d) for d in settings.d_regions_base[-1:]])

        total_d_array = jnp.concatenate([d_regions_before, d_bz[None, :], d_regions_middle, d_shield[None, :], d_regions_after], axis=0) #[n_regions, n_theta]

        total_d_array_cumsum = jnp.concatenate([jnp.zeros((1,n_theta)), jnp.cumsum(total_d_array, axis=0)], axis=0)  # [n_regions, n_theta]

        def _single_theta_linspaces(i):
            s_from_edge = jnp.concatenate([jnp.linspace(total_d_array_cumsum[j, i], total_d_array_cumsum[j+1, i], n_elem_regions[j], endpoint=False) for j in range(0, len(settings.d_regions_base))] + [jnp.array([total_d_array_cumsum[-1, i]])])

            return jnp.where(s_from_edge < 1.0, s_from_edge**settings.s_power_sampling, 1.0 + (s_from_edge - 1.0) / (settings.normalise_distance))[1:] # do not include the axis!

        d_total_array = jax.vmap(_single_theta_linspaces, in_axes=(0))(jnp.arange(settings.n_theta_blanket))
        theta_total = jnp.meshgrid(theta_single_layer, jnp.arange(d_total_array.shape[1]), indexing='ij')[0]  # [n_theta, n_regions*n_elem_regions]

        return theta_total.T, jnp.full_like(theta_total, settings.phi).T, d_total_array.T
        
    @eqx.filter_jit
    def _get_mesh_points(param : BZShield2DParametrisation, settings : BZShield2DSettings):

        theta_total, phi_total, d_total_array = _get_flux_surface_coordinates(param, settings)

        

        return settings.flux_surface_extended.cylindrical_position(jnp.concatenate([jnp.array([0]), d_total_array.ravel()]), 
                                                                jnp.concatenate([jnp.array([0]), theta_total.ravel()]),
                                                                jnp.concatenate([jnp.array([phi_total[0,0]]), phi_total.ravel()]))[..., :2].reshape(-1,2)

    settings = BZShield2DSettings(
        k_spline = 3, 
        n_theta_blanket = n_theta,
        d_regions_base = d_regions_base,
        n_elem_regions = n_elem_regions,
        phi = phi,
        normalise_distance = d_edge_norm,
        s_power_sampling= 2,
        flux_surface_extended = flux_surface_extended    
    )
    n_ctrl = 16
    param = BZShield2DParametrisation.from_settings(n_ctrl, settings)
    coordinates = _get_mesh_points(param, settings)

    # %%
    onp.testing.assert_allclose(MGOP_EA.transport_operator.solution_domain_data.domain.vertices, coordinates)

    # %%
    from jax_sn.tools.parametrisation_tools import ParametrisedModuleTheta
    from jax_sn.parametrisation.multigroup_parametrisation import MultigroupOperatorParametrisedVertices, MultigroupOperatorParametrisedVerticesNoToposort, MultigroupOperatorParametrisedObjective
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import MultiGroupOperator

    class BZShield2DMGOP(ParametrisedModuleTheta):
        additional_settings : BZShield2DSettings    

        def __check_init__(self):
            assert isinstance(self.additional_settings, BZShield2DSettings)
            assert isinstance(self.base_module, MultiGroupOperator)

        @eqx.filter_jit
        def to_module(self, theta : BZShield2DParametrisation):        
            return MultigroupOperatorParametrisedVertices.reconstruct_operator(self.base_module, _get_mesh_points(theta, self.additional_settings))
        
        @eqx.filter_jit
        def to_module_no_toposort(self, theta : BZShield2DParametrisation):        
            return MultigroupOperatorParametrisedVerticesNoToposort.reconstruct_operator(self.base_module, _get_mesh_points(theta, self.additional_settings))
        
    def get_total_plasma_source(solution_domain_data, source_total):
        isotropic_source_first_eg = source_total[0,0] # [first energy group and first angle, isotropic anyway!]

        return jnp.einsum("eb, b, e->", isotropic_source_first_eg, solution_domain_data.fem_matrices.integration_coeff, solution_domain_data.precomputed_geometric_data.vol_jacobians_det)

        
    def _tritium_breeding_ratio_imp(solution, solution_domain_data, source_total, settings : BZShield2DSettings, tritium_xs_bz : jnp.ndarray):
        bz_elements_start =  2 * sum(settings.n_elem_regions[0:3]) * settings.n_theta_blanket - 1 * settings.n_theta_blanket # to correct for the axis wedge    
        n_bz_elements     =  2 * settings.n_elem_regions[3] * settings.n_theta_blanket    
        solution_bz_eg0   = jax.lax.dynamic_slice(solution, (0,0,bz_elements_start,0), slice_sizes = (solution.shape[0], 1, n_bz_elements, solution.shape[-1]))[:,0, ...]
        jacobian_dets_bz  = jax.lax.dynamic_slice(solution_domain_data.precomputed_geometric_data.vol_jacobians_det, (bz_elements_start,), (n_bz_elements,) )        
        return jnp.einsum("g, geb,b,e->", tritium_xs_bz, solution_bz_eg0, solution_domain_data.fem_matrices.integration_coeff, jacobian_dets_bz ) / jnp.sqrt(4 * jnp.pi) / source_total
        
    def _fast_flux_on_edge_blocks(solution_value : jnp.ndarray, solution_domain_data, settings : BZShield2DSettings, energy_set_boundaries, fast_eg = [14.1e6, 1e6]):
        n_edge_elements         = 2 * settings.n_theta_blanket
        edge_elements_start     = 2 * sum(settings.n_elem_regions) * settings.n_theta_blanket - 1 * settings.n_theta_blanket - n_edge_elements    
        solution_edge_eg0       = jax.lax.dynamic_slice(solution_value, (0,0,edge_elements_start,0), slice_sizes = (solution_value.shape[0], 1, n_edge_elements, solution_value.shape[-1]))[:,0, ...]    
        overlap_matrix          = jax_sn.energy_set.compute_overlap_matrix(energy_set_boundaries, fast_eg)    
        egs_fast = jnp.einsum("gi, ieb->geb", overlap_matrix, solution_edge_eg0)[0,...]    
        return (jnp.einsum("eb,b->e", egs_fast, solution_domain_data.fem_matrices.integration_coeff) / jnp.sqrt(4 * jnp.pi)).reshape(-1,2).mean(axis=-1)

    def _distance_edge_blocks(theta : BZShield2DParametrisation, settings : BZShield2DSettings, n_theta : int = 40):
        theta_single_layer = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)    

        d_regions_before   = jnp.array([jnp.full_like(theta_single_layer, d) for d in settings.d_regions_base[0:3]])
        d_bz               = periodic_bspline(theta_single_layer, theta.c_bz, settings.k_spline)
        d_regions_middle   = jnp.array([jnp.full_like(theta_single_layer, d) for d in settings.d_regions_base[4:-2]])
        d_shield           = periodic_bspline(theta_single_layer, theta.c_shield, settings.k_spline)
        d_regions_after    = jnp.array([jnp.full_like(theta_single_layer, d) for d in settings.d_regions_base[-1:]])

        total_d_array = jnp.concatenate([d_regions_before, d_bz[None, :], d_regions_middle, d_shield[None, :], d_regions_after], axis=0) #[n_regions, n_theta]

        total_d_array_cumsum = jnp.concatenate([jnp.zeros((1,n_theta)), jnp.cumsum(total_d_array, axis=0)], axis=0)  # [n_regions, n_theta]
        
        return total_d_array_cumsum[-1]# - total_d_array_cumsum[-2] # distance of edge blocks from the last flux surface

    def smooth_relu(x, beta=50.0):
        return jnp.log(1 + jnp.exp(beta*x)) / beta

    def inboard_weight(theta, theta0, sigma=0.5):
        """
        theta : [N] poloidal angle
        sigma : localization width [rad]

        returns w_IB(θ)
        """
        return jnp.exp(-((theta - theta0)**2) / (2.0 * sigma**2))

    def blanket_objective_normalized(
        d,
        d0,
        theta,
        theta0,
        lam=10.0,
        sigma=0.5,
        beta=50.0
    ):

        w_IB = inboard_weight(theta, theta0, sigma)

        J_IB = jnp.mean(w_IB * d)

        growth = smooth_relu(d - d0, beta)
        J_grow = jnp.mean(growth**2)

        return J_IB + lam * J_grow

    def smooth_max(x, alpha=50.0):
        return jnp.log(jnp.sum(jnp.exp(alpha * x))) / alpha

    def ks_agg(x, alpha=50.0):
        n = x.shape[0]
        m = jnp.max(x)
        return m + jnp.log(jnp.sum(jnp.exp(alpha * (x - m )))) / alpha - jnp.log(n)/ alpha
    
    
        
    class BZShield2DMultiGroupObjective(MultigroupOperatorParametrisedObjective):

        nt_data_bz          : jnp.ndarray
        total_plasma_source : jnp.ndarray
        groups_boundaries  : Tuple[float]

        tbr_constraint : jnp.ndarray 
        ffn_constraint : jnp.ndarray

        scaling_ffn_constraint : float  = 1.0
        lambda_distance_edge_blocks : float = 10.0

        penalize_larger_outboard_than : float = 2.4
        inboard_theta :float = theta_inboard
        sigma_inboard : float = sigma_inboard


        @eqx.filter_jit
        def objective_and_constraints(self, theta : BZShield2DParametrisation):
            mgop_theta = self.theta_parametrised_module.to_module(theta)
            solution_value = self._to_solution(mgop_theta).value

            tbr_constraint =   _tritium_breeding_ratio_imp(solution_value, mgop_theta.transport_operator.solution_domain_data, self.total_plasma_source, self.theta_parametrised_module.additional_settings, self.nt_data_bz) -self.tbr_constraint
            ffn_constraints = - ks_agg(_fast_flux_on_edge_blocks(solution_value, mgop_theta.transport_operator.solution_domain_data, self.theta_parametrised_module.additional_settings, self.groups_boundaries) / self.ffn_constraint - 1.0) / self.scaling_ffn_constraint

            d_total = _distance_edge_blocks(theta, self.theta_parametrised_module.additional_settings)
            theta_sample = jnp.linspace(0, 2*jnp.pi, d_total.shape[0], endpoint=False)

            blanket_obj = blanket_objective_normalized(d_total, d0=self.penalize_larger_outboard_than, theta=theta_sample, theta0=self.inboard_theta, sigma=self.sigma_inboard)
            return jnp.array([blanket_obj, tbr_constraint, ffn_constraints])
            
        
        def tritium_breeding_ratio(self, theta : BZShield2DParametrisation):        
            mgop_theta = self.theta_parametrised_module.to_module(theta)
            solution_value = self._to_solution(mgop_theta).value
            return _tritium_breeding_ratio_imp(solution_value, mgop_theta.transport_operator.solution_domain_data, self.total_plasma_source, self.theta_parametrised_module.additional_settings, self.nt_data_bz )
        

        def edge_fast_flux(self, theta : BZShield2DParametrisation):
            mgop_theta = self.theta_parametrised_module.to_module(theta)
            solution_value = self._to_solution(mgop_theta).value
            return _fast_flux_on_edge_blocks(solution_value, mgop_theta.transport_operator.solution_domain_data, self.theta_parametrised_module.additional_settings, self.groups_boundaries)
        

    def get_solution_domain(param, mgop_objective):
        return jax_sn.solution_domain.SolutionDomain(mgop_objective.to_module_no_toposort(param).transport_operator.solution_domain_data, element)

        
    mgsolver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(rtol = 1e-7, atol=1e-7, max_steps = 350))
    mgop_parametrised = BZShield2DMGOP(MGOP_EA, additional_settings = settings)
    mgop_objective = BZShield2DMultiGroupObjective(mgop_parametrised, source_all_angles_ea, mgsolver, nt_data_bz= new_aux_data_dict['BreedingZone']['nt'], total_plasma_source= get_total_plasma_source(MGOP_EA.solution_domain_data, source_all_angles_ea), groups_boundaries=groups_list, tbr_constraint = jnp.array(1.4), ffn_constraint = jnp.array(1e14))


    # %%
    from jax_sn.tools.parametrisation_tools import ParametrisedScipyInterface

    scipy_interface = ParametrisedScipyInterface(mgop_objective, (lambda x : BZShield2DParametrisation(x[:len(x)//2], x[len(x)//2:]), lambda y : onp.concatenate([y.c_bz, y.c_shield], axis=-1)))

    # %%
    onp.set_printoptions(linewidth = 200)

    # %%
    constraints = [{"type" : "ineq", "fun" : scipy_interface.constraints, "jac": scipy_interface.jac_constraints},
        ]

    start_theta = scipy_interface.f_theta_parametrisation_values[1](param)


    def visualize_domain_vtk(x_i, filename):
        theta = BZShield2DParametrisation(x_i[:len(x_i)//2], x_i[len(x_i)//2:])
        mgop_theta = mgop_objective.theta_parametrised_module.to_module_no_toposort(theta)
        import meshio 
        domain = meshio.Mesh(points = mgop_theta.transport_operator.solution_domain_data.domain.vertices, cells = {"triangle" : mgop_theta.transport_operator.solution_domain_data.domain.connectivity}, cell_data ={"region_ids" : [onp.array(mgop_theta.transport_operator.solution_domain_data.domain.cross_sections_indexed.indices, dtype=float)]})

        domain.write("iterations/"+filename)

    from scipy.optimize import minimize

    x_hist = []
    iteration = {"count": 0}  # use dict to allow mutation in inner function

    visualize_domain_vtk(start_theta, f"domain_{iteration['count']}.vtk")

    def callback(intermediate_result):
        print("\n\nIteration" , iteration['count'] , " resulted in xk:", intermediate_result.x, "\n\n")
        x_hist.append(intermediate_result.x.copy())    
        iteration['count'] += 1
        visualize_domain_vtk(intermediate_result.x, f"domain_{iteration['count']+1}.vtk")
        onp.save(f"x_{iteration['count']}.npy", intermediate_result.x)

    aa = minimize(
        fun = scipy_interface.objective,
        jac = scipy_interface.grad_objective,
        x0 = start_theta,
        constraints = constraints,
        method = "SLSQP",
        options = {"disp" : True, "ftol" : 1e-4, "maxiter" : 200},
        bounds = [(1e-6, 1.0)] * len(start_theta),
        callback = callback
    )

if __name__ == "__main__":
    import pickle
    with open("setup.pkl", "rb") as f:
        setup_dict = pickle.load(f)
    
    phi = setup_dict["phi"]
    theta_inboard = setup_dict["theta_inboard"]
    sigma_inboard = setup_dict["sigma_inboard"]
    
    run_optimization(phi, theta_inboard, sigma_inboard)