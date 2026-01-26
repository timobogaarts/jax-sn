import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def compute_overlap_matrix(old_boundaries, new_boundaries):

    """
    Computes the matrix of weighted averages between two sets of boundaries.
    The overlap is defined as the fraction of the old group that overlaps with the new group.

    Works for both a higher resolution new group structure and a lower resolution new group structure.
    E.g. if the old group structure is [8,6,2,1] and the new group structure is [8,7,6,5,4,3,2,1], then the overlap matrix will be:
    [[0.5 , 0.  , 0.  ],
     [0.5 , 0.  , 0.  ],
     [0.  , 0.25, 0.  ],
     [0.  , 0.25, 0.  ],
     [0.  , 0.25, 0.  ],
     [0.  , 0.25, 0.  ],
     [0.  , 0.  , 1.  ]]
    
    and thus the first new energy group is half of the first old energy group, the second new energy group is half of the first old energy group,
    the third new energy group is a quarter of the second old energy group, and so on.

    This overlap matrix can be used to compute values defined on the old group structure on a new group structure:

    new_group_values = overlap_matrix @ old_group_values


    Parameters
    ----------
    old_boundaries : 1D array-like
        The boundaries of the old groups.
    new_boundaries : 1D array-like         
        The boundaries of the new groups.
    Returns
    -------
    overlap : 2D array
        The overlap matrix, where overlap[i, j] is the fraction of the old group i that overlaps with the new group j.    
    """
    
    old_boundaries = jnp.asarray(old_boundaries, dtype=float)
    new_boundaries = jnp.asarray(new_boundaries, dtype=float)

    old_boundaries = eqx.error_if(old_boundaries, jnp.any(jnp.diff(old_boundaries) >= 0), "old_boundaries must be strictly decreasing (high to low) and non-degenerate.")
    new_boundaries = eqx.error_if(new_boundaries, jnp.any(jnp.diff(new_boundaries) >= 0), "new_boundaries must be strictly decreasing (high to low) and non-degenerate.")
        
    old_hi = old_boundaries[:-1][:, None]  # (n_old, 1)
    old_lo = old_boundaries[1:][:, None]   # (n_old, 1)
    new_hi = new_boundaries[:-1][None, :]  # (1, n_new)
    new_lo = new_boundaries[1:][None, :]   # (1, n_new)

    overlap_hi = jnp.minimum(old_hi, new_hi)
    overlap_lo = jnp.maximum(old_lo, new_lo)
    width = jnp.clip(overlap_hi - overlap_lo, 0, None)  # (n_old, n_new)
    old_width = (old_hi - old_lo)

    overlap = width / old_width
    return overlap.T
