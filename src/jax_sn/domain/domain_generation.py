import jax.numpy as jnp
import jax 
from typing import List, Tuple  
from .cross_sections import CrossSectionData
from .geometric_domain import CrossSectionIndexed
from .geometric_domain import Domain

def create_cartesian_setup(dimension : int, region_size_res : List[List[Tuple]], cross_sections_regions : List[CrossSectionData], source_regions : List[float], inhomogeneity_direction : int):
    import jax_sbgeom
    '''
    Convenience function to create a Cartesian mesh with layered cross sections and sources. This is not necessarily a differentiable function and is therefore not intended to be used 
    in optimization. Mainly for testing purposes.

    Parameters    
    ----------
    dimension : int
        The spatial dimension of the domain (1, 2, or 3).
    region_size_res : List[List[Tuple]]
        A list of lists specifying the size and resolution of each region in each dimension. Each inner list corresponds to a dimension and contains tuples of (size, resolution) for each region in that dimension.
    cross_sections_regions : List[float]
        A list of cross section values for each region in the inhomogeneity direction.
    source_regions : List[float]
        A list of source values for each region in the inhomogeneity direction.
    inhomogeneity_direction : int
        The dimension index (0-indexed) along which the cross sections and sources vary.
    Returns
    -------
    vertices : jnp.ndarray
        The vertices of the Cartesian mesh.
    connectivity : jnp.ndarray
        The connectivity of the Cartesian mesh.
    layered_cross_section : jnp.ndarray
        The cross section values for each element in the mesh.
    layered_source : jnp.ndarray
        The source values for each element in the mesh.
    '''
    assert inhomogeneity_direction < dimension, f"Inhomogeneity direction must be less than dimension [0 indexed]. Dimension: {dimension}, inhomogeneity_direction: {inhomogeneity_direction}"
    assert len(region_size_res) == dimension, f"Regions size and resolution must be specified for all dimensions. Dimension: {dimension}, provided: {len(region_size_res)}"
    assert len(region_size_res[inhomogeneity_direction]) == len(cross_sections_regions) == len(source_regions),\
          f"Number of regions in inhomogeneity direction must match number of cross sections and sources. Provided: {len(region_size_res[inhomogeneity_direction])} sizes & resolutions, {len(cross_sections_regions)} cross sections, {len(source_regions)} sources."
    
    nodes = []
    for dim_i in range(len(region_size_res)):
        assert all([isinstance(x, List) or isinstance(x, Tuple) and len(x) == 2 for x in region_size_res[dim_i]]), f"Each entry in region_size_res must be a list of tuples (size, resolution). Provided: {region_size_res[dim_i]}"
        regions_dim   = region_size_res[dim_i]
        n_regions_dim = len(region_size_res[dim_i])        
        cumsum_nodes  = jnp.concatenate([jnp.array([0]), jnp.cumsum(jnp.array([sz for sz, res in regions_dim]))])        
        nodes.append(jnp.concatenate([jnp.linspace(cumsum_nodes[i], cumsum_nodes[i+1], region_size_res[dim_i][i][1], endpoint=False) for i in range(n_regions_dim)] + [jnp.array([cumsum_nodes[-1]])]))

    meshgrid = jnp.meshgrid(*nodes, indexing='ij')

    def make_element_array(scalar_on_regions, region_size_res, inhomogeneity_direction):
        scalar_inhom_layer = jnp.concatenate([jnp.full((region_size_res[inhomogeneity_direction][i][1],), scalar_on_regions[i]) for i in range(len(scalar_on_regions))])
        dummy_scalars      = [jnp.concatenate([jnp.zeros(region_size_res[dim_i][i][1]) for i in range(len(region_size_res[dim_i]))]) for dim_i in range(dimension)]
        scalar_list        = [dummy_scalars[dim_i] if dim_i != inhomogeneity_direction else scalar_inhom_layer for dim_i in range(dimension)]
        scalar_mg          = jnp.meshgrid(*scalar_list, indexing='ij')[inhomogeneity_direction]

        n_per_block = [None, 1, 2, 6][dimension]
        return jnp.repeat(scalar_mg, repeats=n_per_block, axis=-1).reshape(scalar_mg.shape + (n_per_block,))
    
    cross_section_data = CrossSectionData.from_list(cross_sections_regions)

    cross_section_indices = jnp.arange(len(cross_sections_regions)) # [0,1,2,...]

    layered_indices       = make_element_array(cross_section_indices, region_size_res, inhomogeneity_direction)

    layered_cross_section = CrossSectionIndexed(layered_indices.flatten(), cross_section_data)
    
    layered_source        = make_element_array(source_regions, region_size_res, inhomogeneity_direction)

    if dimension == 1:
        vertices = meshgrid[0].reshape(-1,1)
        connectivity = jnp.array([[i, i+1] for i in range(vertices.shape[0]-1)])
    elif dimension == 2:
        vertices  = jnp.stack(meshgrid, axis=-1).reshape(-1,2) # (n_points, 2) [implicit shape (n_x, n_y, 2)]
        connectivity = jax_sbgeom.flux_surfaces.flux_surface_meshing._build_open_triangle_surface(nodes[0].shape[0], nodes[1].shape[0], True)
    elif dimension == 3:
        vertices  = jnp.stack(meshgrid, axis=-1).reshape(-1,3) # (n_points, 3) [implicit shape (n_x, n_y, n_z, 3)]
        connectivity = jax_sbgeom.flux_surfaces.flux_surface_meshing._tetrahedral_open_cube(nodes[0].shape[0], nodes[1].shape[0], nodes[2].shape[0])
    else:
        raise NotImplementedError(f"Dimension {dimension} not supported. Only 1, 2, and 3 are supported.")
        
    return vertices, connectivity, layered_cross_section, layered_source.flatten()
            