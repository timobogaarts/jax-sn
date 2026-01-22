import pytest
import jax_sn
import numpy as onp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# Old StellBlanket implementation for verification
def create_TN(N: int):
    import numpy as np
    def create_octants(N: int):
        N=N+1
        assert(N>1)
        points_per_level = np.arange(1,N+1)
        fractions = np.linspace(0,1,N)
        xz_points = np.array([0,0,1]) + fractions[:,np.newaxis] * np.array([1,0,-1])
        yz_points = xz_points[:,[1,0,2]]
        points = np.zeros((np.sum(points_per_level),3))
        vertices = []
        for level_number, number_of_points_this_level in enumerate(points_per_level):
            
            index_start = np.sum(points_per_level[0:level_number])
            
            start_vec = xz_points[level_number]
            diff_vec = yz_points[level_number] - xz_points[level_number]
            fractions_lvl = np.linspace(0,1,number_of_points_this_level)
            points[index_start:(index_start+number_of_points_this_level),:] = start_vec + fractions_lvl[:,np.newaxis] * diff_vec
            
            ## We create the vertices ere
        
        def index_in_points(point_number_in_this_level, level_number):
            if(level_number >= N):
                return -1
            if(point_number_in_this_level >= points_per_level[level_number]):
                return -1
            return np.sum(points_per_level[0:level_number]) + point_number_in_this_level

        for level_number, number_of_points_this_level in enumerate(points_per_level):
            for point_number_in_this_level in range(number_of_points_this_level):
                self_index = index_in_points(point_number_in_this_level,level_number)
                left_down  = index_in_points(point_number_in_this_level, level_number + 1)
                right_down = index_in_points(point_number_in_this_level + 1, level_number + 1)

                right_same = index_in_points(point_number_in_this_level + 1, level_number)
                
                if left_down > -1 and right_down > -1:
                    vertices.append([self_index,left_down, right_down])

                if right_same > -1 and right_down > -1:
                    vertices.append([self_index,right_same, right_down])


                
                
        return points,vertices
                
    points, vertices = create_octants(N)

    discrete_ordinate_angles = np.zeros((len(vertices),3))
    triangle_proj = np.zeros((len(vertices),3,3))

    vertices_plot = np.zeros((len(vertices),3),dtype=int)
    area   = np.zeros(len(vertices))
    for triangle_number, triangle in enumerate(vertices):
        points_tri = points[triangle]
        points_avg = (points_tri[0,:]+points_tri[1,:]+points_tri[2,:]) /3.0
    
        points_tri_proj = points_tri / np.sqrt(np.einsum("ij,ij->i",points_tri,points_tri))[:,np.newaxis]
        points_avg_proj =points_avg / np.sqrt(np.einsum("i,i",points_avg, points_avg))
        discrete_ordinate_angles[triangle_number,:] = points_avg_proj
        triangle_proj[triangle_number,:,:] = points_tri_proj
        vertices_plot[triangle_number,:] = [3 * triangle_number + 0,3 * triangle_number + 1,3 * triangle_number + 2 ]

        #See https://en.wikipedia.org/wiki/Solution_of_triangles#Solving_spherical_triangles for definition of angles
        A = points_tri_proj[0,:]
        B = points_tri_proj[1,:]
        C = points_tri_proj[2,:]

        a = np.arccos(np.dot(C,B))
        b = np.arccos(np.dot(A,C))
        c = np.arccos(np.dot(A,B))
        alpha = np.arccos( ( np.cos(a) - np.cos(b) * np.cos(c)) / ( np.sin(b) * np.sin(c)))
        beta  = np.arccos( ( np.cos(b) - np.cos(c) * np.cos(a)) / ( np.sin(c) * np.sin(a)))
        gamma = np.arccos( ( np.cos(c) - np.cos(a) * np.cos(b)) / ( np.sin(a) * np.sin(b)))
        spherical_excess = alpha + beta + gamma -np.pi
        area[triangle_number] = spherical_excess
        
        


    total_angles = np.zeros((8 * len(discrete_ordinate_angles),3))

    total_weights = np.zeros(8 * len(discrete_ordinate_angles))


    for i, angle in enumerate(discrete_ordinate_angles):
        j = 0 
        for x_sym  in [-1,1]:
            for y_sym in [-1,1]:
                for z_sym in [-1,1]:
                    total_angles[8*i +j] =  [x_sym * angle[0], y_sym * angle[1], z_sym * angle[2] ]
                    
                    total_weights[8*i + j]=  area[i]
                    j += 1

    total_weights = total_weights / np.sum(total_weights)
    total_angles
    return total_angles, total_weights

@pytest.mark.Parametrise("N", [2,3,4,5,6])
def test_tn(N : int):
    qset = jax_sn.quadrature_set.tn.create_tn_quadrature_set(N)

    angles_ref, weights_ref = create_TN(N)
    onp.testing.assert_allclose(qset.angles, angles_ref)
    onp.testing.assert_allclose(qset.weights / 4 / onp.pi, weights_ref)


