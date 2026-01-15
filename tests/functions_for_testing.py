import jax.numpy as jnp 
import numpy as onp

def plot_levels(geometric_domain, angle):
    def get_arrow_next_to_domain(geometric_domain, angle):
        rmin = jnp.min(jnp.sqrt(geometric_domain.vertices[:,0]**2 + geometric_domain.vertices[:,1]**2))
        phi = jnp.mean(jnp.arctan2(geometric_domain.vertices[:,1],geometric_domain.vertices[:,0]))

        point = jnp.array([rmin * 0.9 * jnp.cos(phi), rmin * 0.9* jnp.sin(phi), 0.0])
        return onp.array(point), onp.array(angle)
    
    from matplotlib.colors import ListedColormap
    from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
    import jax_sn as jsn
    import pyvista as pv
    levels = jsn.domain.graph_generation.compute_levels_domain_safe(geometric_domain, angle, 100)

    plotter =pv.Plotter()
    mesh_base = (geometric_domain.vertices, geometric_domain.connectivity)
    mesh_pv = mesh_to_pyvista_mesh(mesh_base)
    mesh_pv.cell_data["levels"] = levels
    random_colors = onp.random.rand(jnp.max(levels)+1, 3)
    cmap = ListedColormap(random_colors)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_to_pyvista_mesh(mesh_base), scalars=levels + 0.5, show_edges=True, cmap=cmap, clim = (0, jnp.max(levels) + 1))
    plotter.add_arrows(*get_arrow_next_to_domain(geometric_domain, angle), color="black", mag=2) 
    plotter.show()

def solve_and_plot(MGOP, rhs, sol_domain_2, lower_triangular_solver):
    import matplotlib.pyplot as plt
    import lineax as lx
    x               = jnp.linspace(0,2.432, 1000)
    y               = jnp.full_like(x, 2.5)
    xy              = jnp.stack([x,y], axis=-1)
    solution_i      = lx.linear_solve(MGOP, rhs, lower_triangular_solver)
    solution_result = [sol_domain_2.interpolate(xy, solution_i.value[i, 0]) for i in range(MGOP.n_energy_groups)]
    plt.semilogy(x, onp.array(solution_result).T)
    plt.xlabel("x [m]")
    plt.ylabel("$\\phi$")