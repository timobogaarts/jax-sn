import jax 
import jax.numpy as jnp
from functools import partial
from .quadrature_set_base import QuadratureSet
@partial(jax.jit, static_argnums= 0)
def _create_tn_octants( N : int):
    N = N + 1
    points_per_level = jnp.arange(1, N+1)
    fractions        = jnp.linspace(0, 1, N)
    xz_points        =  jnp.array([0, 0, 1]) + fractions[:, None] * jnp.array([1, 0, -1])
    yz_points        = xz_points[:, [1, 0, 2]]

    total_points_octant = N * ( N +1) // 2 #  n_points = jnp.sum(points_per_level)


    level_numbers    = jnp.repeat(jnp.arange(N), jnp.arange(1, N +1), total_repeat_length = total_points_octant)        
    start_vec        = xz_points[level_numbers]
    diff_vec         = yz_points[level_numbers] - xz_points[level_numbers]
    fractions_lvl    = jnp.concatenate([jnp.linspace(0, 1, i ) for i in range(1, N + 1)])    
    points           = start_vec + fractions_lvl[:, None] * diff_vec 
    
    
    def index_in_points(point_number_in_this_level, level_number):
        return  level_number * (level_number + 1) // 2 + point_number_in_this_level # we only call this with valid indices
    
    # Create triangles
    # We go from the top down.
    #               0
    #              / \
    #             1---2
    #            / \ / \
    #           3---4---5
    #          / \ / \ / \
    #         6---7---8---9
    # 
    # We can for each point distinguish two cases:
    # 1. It has a triangle down left and down right (i.e. point 0,1,2,3,4,5, just not the last layer)
    # 2. It has a triangle right same and right down (i.e. point 0,1,3,4, just not the last point of each layer and not the last layer)
    double_down_triangles                = jnp.array(jnp.arange(0, total_points_octant - (N)))   # all points but ignore last layer                                           
    double_down_triangles_point_in_level = jnp.concatenate([jnp.arange(i) for i in range(1, N)])  # all points but ignore last layer (we arange to the number of points in each layer)
    double_down_levels                   = jnp.repeat(jnp.arange(N - 1), jnp.arange(1, N), total_repeat_length = total_points_octant - (N)) #  repeat levels accordingly
    
    
    right_same_triangles       = jnp.concatenate([jnp.arange(i * (i + 1) // 2, (i + 1) * (i + 2) // 2 - 1) for i in range(N - 1)])  # ignore last point of each layer and last layer by aranging from start of layer to end -1 for each layer - last layer
    right_same_points_in_level = jnp.concatenate([jnp.arange(i) for i in range(1, N - 1)])    # ignore last point of each layer and last layer [the last point is automatically ignored by the range ]    
    right_same_levels          = jnp.repeat(jnp.arange(1, N - 1), jnp.arange(1, N - 1), total_repeat_length = right_same_points_in_level.shape[0])    # ignore last layer and last point of each layer and thus start from level 1 

    vertices_double_down = jnp.stack([
        double_down_triangles,
        index_in_points(double_down_triangles_point_in_level, double_down_levels + 1),
        index_in_points(double_down_triangles_point_in_level + 1, double_down_levels + 1)
    ], axis=1)  
    
    vertices_right_same = jnp.stack([
        right_same_triangles,
        index_in_points(right_same_points_in_level + 1, right_same_levels),
        index_in_points(right_same_points_in_level + 1, right_same_levels + 1)
    ], axis=1)

    # Now we need to interleave the get in the order you encounter them (not necessary but for historic reasons)
    def interleave_vertices(vertices_dd, vertices_rs):
        total_triangles = vertices_dd.shape[0] + vertices_rs.shape[0]
                
        def sum_triangles_up_to_level(level):
            return level **2
        
        dd_indices = jnp.concatenate([jnp.arange(sum_triangles_up_to_level(i), sum_triangles_up_to_level(i+1), 2) for i in range(N - 1)])
        rs_indices = jnp.concatenate([jnp.arange(sum_triangles_up_to_level(i) + 1, sum_triangles_up_to_level(i+1), 2) for i in range(N - 1)])

        total_vertices = jnp.zeros((total_triangles, 3), dtype = vertices_dd.dtype)
        total_vertices = total_vertices.at[dd_indices].set(vertices_dd)
        total_vertices = total_vertices.at[rs_indices].set(vertices_rs)
        return total_vertices    
    
    return points, interleave_vertices(vertices_double_down, vertices_right_same)
    
@partial(jax.jit, static_argnums= 0)
def create_tn_quadrature_set(N : int):
    '''
    Creates a TN quadrature set with N levels per octant. See [1].

    Number of angles is:
    
    .. math::
        8 N^2.

    [1] Thurgood, C. P., Pollard, A., & Becker, H. A. (1995). The TN quadrature set for the discrete ordinates method.

    Parameters
    ----------
    N : int
        Number of levels per octant.
    Returns
    -------
    total_angles : jnp.ndarray
        Array of shape [8 * N**2, 3] containing the discrete ordinate angles.
    total_weights : jnp.ndarray
        Array of shape [8 * N**2] containing the weights.
    '''
    
    points, vertices = _create_tn_octants(N)
    
    def create_angle_area(vertex : jnp.ndarray) -> jnp.ndarray:
        centre = jnp.mean(vertex, axis=-2)
        points_proj = vertex / jnp.linalg.norm(vertex, axis=-1, keepdims=True)
        points_avg_proj = centre / jnp.linalg.norm(centre, axis=-1, keepdims=True)

        A = points_proj[..., 0,:]
        B = points_proj[..., 1,:]
        C = points_proj[..., 2,:]

        a = jnp.arccos(jnp.sum(C * B, axis=-1))
        b = jnp.arccos(jnp.sum(A * C, axis=-1))
        c = jnp.arccos(jnp.sum(A * B, axis=-1))
        alpha = jnp.arccos( ( jnp.cos(a) - jnp.cos(b) * jnp.cos(c)) / ( jnp.sin(b) * jnp.sin(c)))
        beta  = jnp.arccos( ( jnp.cos(b) - jnp.cos(c) * jnp.cos(a)) / ( jnp.sin(c) * jnp.sin(a)))
        gamma = jnp.arccos( ( jnp.cos(c) - jnp.cos(a) * jnp.cos(b)) / ( jnp.sin(a) * jnp.sin(b)))
        return points_avg_proj, alpha + beta + gamma - jnp.pi        
    
    triangles = points[vertices]    
    angles, areas = create_angle_area(triangles)
    # map to all octants
    weight_fractions = jnp.stack(jnp.meshgrid(jnp.array([-1,1]), jnp.array([-1,1]), jnp.array([-1,1]), indexing='ij'), axis=-1).reshape(-1,3)            
    total_angles     = (angles[:, jnp.newaxis, :] * weight_fractions[jnp.newaxis, :, :]).reshape(-1,3)        
    total_weights    = jnp.repeat(areas, 8) / jnp.sum(areas) / 8 * 4 * jnp.pi
    return QuadratureSet(total_angles, total_weights)
    