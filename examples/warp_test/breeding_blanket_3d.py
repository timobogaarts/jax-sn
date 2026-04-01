import jax 
import jax.numpy as jnp
from jax_sbgeom.interfaces.blanket_creation import LayeredDiscreteBlanket, LayeredDiscreteBlanketPlasmaTransformed
from jax_sbgeom.flux_surfaces import ToroidalExtent
import jax_sbgeom as jsb

import pickle
import jax_sn
import functools

def ENERGY_GROUPS():    
    GROUPS_LIST    = (14.1e6, 10e6, 3e6, 1e6, 1e4, 300, 10, 0)
    return jnp.array(GROUPS_LIST)

def DATA_DIRECTORY():
    return "/home/tbogaarts/blanket_optimization/3d/data/"

def D_EDGE_NORM():
    return 1.8

def BASE_REGION_SIZE():
    d_regions_base = (1.2, 0.002, 0.025, 0.5, 0.385, 0.06, 0.2, 0.06)
    return d_regions_base

def S_POWER_SAMPLING():
    return 2

def SOURCE_KWARGS():
    return {    
    "n0"     : 2.4e20,
    "Ti0"    : 15.2,
    "nalpha" : 0.35,
    "Tialpha": 1.2,
    }

def SOURCE_CLOSURE():
    return functools.partial(jsb.interfaces.plasma.flux_surface_reaction_rates_simple, **SOURCE_KWARGS())

def get_fs_extended(vmec_file):
    # Extends the flux surface to the D_EDGE_NORM() value.
    flux_surface = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(vmec_file)
    flux_surface_extended = jsb.flux_surfaces.create_extended_flux_surface_d_interp_equal_arclength(flux_surface, D_EDGE_NORM(), 70, 80, 100)
    return  flux_surface_extended

def load_nuclear_data():
    
    with open(DATA_DIRECTORY() + "new_bb_data_dict.pkl", "rb") as f:
        new_bb_data_dict = pickle.load(f)
    with open(DATA_DIRECTORY() + "new_aux_data_dict.pkl", "rb") as f:
        new_aux_data_dict = pickle.load(f)
    with open(DATA_DIRECTORY() + "tally_result_dict.pkl", "rb") as f:
        tally_result_dict = pickle.load(f)
    
    return new_bb_data_dict, new_aux_data_dict, tally_result_dict

def L_ORDER():
    return 3

def get_blanket_base_case():
    with open(DATA_DIRECTORY() + "hcpb_base_case.pkl", "rb") as f:
        blanket_1d_dict = pickle.load(f)
     
    resolutions_blanket = [10, 1, 1, 6,4,3,4,3]
    n_theta_blanket = 50
    n_phi_blanket = 60
    
    flux_surface =get_base_flux_surface_layers()
    _, d_physical = _get_base_distance_matrices()



    discrete_blanket = LayeredDiscreteBlanketPlasmaTransformed(n_theta_blanket, n_phi_blanket, tuple(resolutions_blanket), ToroidalExtent.half_module(flux_surface.flux_surface_extended))

    bb_data, aux_data, _ = load_nuclear_data()

    total_cross_section_data = [bb_data[layer_name][0] for layer_name in blanket_1d_dict["layer_names"]]
    
    scatter_matrix_data = [bb_data[layer_name][1][:(L_ORDER() + 1)] for layer_name in blanket_1d_dict["layer_names"]]
    print(scatter_matrix_data[0].shape)
    nt_matrix_data = [aux_data[layer_name]['nt'] for layer_name in blanket_1d_dict["layer_names"]]

    stacked_cross_section_data = jnp.stack(total_cross_section_data, axis=0)
    stacked_scatter_matrix_data = jnp.stack(scatter_matrix_data, axis=0)
    stacked_nt_matrix_data = jnp.stack(nt_matrix_data, axis=0)

    cross_section_data = jax_sn.domain.CrossSectionData(stacked_cross_section_data, stacked_scatter_matrix_data)

    return discrete_blanket, flux_surface, cross_section_data, stacked_nt_matrix_data, _total_distance_to_diff_and_plasma(d_physical)

def _get_base_distance_matrices():
    d_layers_cumsum = (jnp.cumsum(jnp.array(BASE_REGION_SIZE())) - 1.0)/D_EDGE_NORM()
    d_layers = jnp.broadcast_to(d_layers_cumsum[:, None, None], (len(BASE_REGION_SIZE()), 10, 10))    
    return d_layers, d_layers_cumsum * D_EDGE_NORM() # d_normalized, and d physical

def get_base_flux_surface_layers():
    d_layers, _ = _get_base_distance_matrices()
    return jsb.flux_surfaces.FluxSurfaceExtendedDistanceMatrix(get_fs_extended(DATA_DIRECTORY() + "helias5_vmec.nc4"), d_layers)


def get_source(blanket : LayeredDiscreteBlanketPlasmaTransformed):
    radial_source = jnp.nan_to_num(jsb.interfaces.plasma.flux_surface_reaction_rates_simple(blanket.s_spacing, **SOURCE_KWARGS()))

    radial_source_midpoint = (radial_source[:-1] + radial_source[1:])/2
    
    return blanket.volume_mesh_structure.map_radial_array_to_layers(radial_source_midpoint)
    

def _get_mcbb_blanket(discrete_blanket : LayeredDiscreteBlanketPlasmaTransformed):
    import mcbb

    
    
def _total_distance_to_diff_and_plasma(d_physical):
    d_diff = jnp.diff(d_physical, axis=0, prepend=0)    
    return d_diff.at[0].set(d_diff[0] + 1.0)