import jax 
import jax.numpy as jnp
import numpy as onp
def psi_isotropic_box(points : onp.ndarray, box_size : float, omega : onp.ndarray, sigma : float, q : float):
    '''
    Compute the analytical solution of the transport equation in a box with isotropic source and absorption.

    Parameters
    ----------
    points : onp.ndarray [n_points, n_dim]
        The points at which to compute the solution.
    box_size : float
        The size of the box (assumed to be a cube from (0,0,0) to (a,a,a)).
    omega : onp.ndarray [3]
        The direction of travel.
    sigma : float
        The absorption cross section.
    q : float
        The isotropic source strength.
    '''
    
    ndim = points.shape[1]

    if ndim == 1:
        x = points[...,0]
        s = onp.where(omega[0] > 0, x/omega[0], onp.where(omega[0] < 0, (box_size - x)/(-omega[0]), onp.inf))
        
        return psi
    elif ndim == 2:
        x = points[...,0]; y = points[...,1]        
        mu, eta = onp.asarray(omega)                
        t_x = onp.where(mu > 0, x/mu, onp.where(mu < 0, (box_size - x)/(-mu), onp.inf))
        t_y = onp.where(eta> 0, y/eta,onp.where(eta < 0, (box_size - y)/(-eta), onp.inf))
        s = onp.minimum(t_x, t_y)
    elif ndim == 3:
        x = points[...,0]; y = points[...,1]; z = points[...,2]
        mu, eta, xi = onp.asarray(omega)
        t_x = onp.where(mu > 0, x/mu, onp.where(mu < 0, (box_size - x)/(-mu), onp.inf))
        t_y = onp.where(eta> 0, y/eta, onp.where(eta < 0, (box_size - y)/(-eta), onp.inf))
        t_z = onp.where(xi > 0, z/xi, onp.where(xi < 0, (box_size - z)/(-xi), onp.inf)) 
        s = onp.minimum(onp.minimum(t_x, t_y), t_z)
    else:
        raise ValueError("Only 1D, 2D and 3D are supported.")
    
    if sigma == 0.0:
        psi = q * s
    else:
        psi = (q / sigma) * (1.0 - onp.exp(-sigma * s))
    return psi