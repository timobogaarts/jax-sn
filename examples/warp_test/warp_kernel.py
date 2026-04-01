"""
Warp implementation of the transport sweep kernel.

The sweep is:
    ψ_i = G_inv_i @ ( x_i + Σ_f |Ω·n_f| J_f  D_f  ψ_neighbour_f )

Elements within the same topological level are independent and processed in
parallel; levels are stepped through sequentially from the Python side.

Currently specialised for degree-1 3-D simplices (n_basis = n_faces = 4).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from dataclasses import dataclass

# ── Warp structured types ─────────────────────────────────────────────────────
# Defined at module level so the kernel can capture them at decoration time.
mat4d = wp.mat(shape=(4, 4), dtype=wp.float64)
vec4d = wp.vec(length=4,     dtype=wp.float64)


@wp.kernel
def _sweep_level_kernel(
    topo_offset : int,
    argsort     : wp.array(dtype=wp.int32),
    gij_inv     : wp.array(dtype=mat4d),         # [N_topo]
    dij         : wp.array2d(dtype=mat4d),        # [N_topo, n_faces]
    omega       : wp.array2d(dtype=wp.float64),   # [N_topo, n_faces]  |Ω·n| J_f
    etoe        : wp.array2d(dtype=wp.int32),     # [N_total, n_faces] -1 = boundary
    x           : wp.array(dtype=vec4d),          # [N_total]
    result      : wp.array(dtype=vec4d),          # [N_total]  read + written
):
    tid      = wp.tid()
    topo_i   = topo_offset + tid
    elem_idx = int(argsort[topo_i])

    # s_dj = Σ_f  |Ω·n_f| J_f  *  D_f  @  ψ_neighbour_f
    z = wp.float64(0.0)
    s = vec4d(z, z, z, z)
    for f in range(4):
        nb = int(etoe[elem_idx, f])
        if nb >= 0:                               # -1 signals a boundary face
            s = s + omega[topo_i, f] * (dij[topo_i, f] @ result[nb])

    result[elem_idx] = gij_inv[topo_i] @ (x[elem_idx] + s)


@wp.kernel
def _sweep_sequential_kernel(
    n_topo  : int,
    argsort : wp.array(dtype=wp.int32),
    gij_inv : wp.array(dtype=mat4d),         # [N_topo]
    dij     : wp.array2d(dtype=mat4d),        # [N_topo, n_faces]
    omega   : wp.array2d(dtype=wp.float64),   # [N_topo, n_faces]
    etoe    : wp.array2d(dtype=wp.int32),     # [N_total, n_faces]
    x       : wp.array(dtype=vec4d),          # [N_total]
    result  : wp.array(dtype=vec4d),          # [N_total]  read + written
):
    # Single thread; processes all elements sequentially in topological order.
    for i in range(n_topo):
        elem_idx = int(argsort[i])
        z = wp.float64(0.0)
        s = vec4d(z, z, z, z)
        for f in range(4):
            nb = int(etoe[elem_idx, f])            
            s = s + omega[i, f] * (dij[i, f] @ result[nb])
        result[elem_idx] = gij_inv[i] @ (x[elem_idx] + s)


@wp.kernel
def _sweep_simd_kernel(
    n_topo    : int,
    argsort   : wp.array(dtype=wp.int32),
    gij_inv   : wp.array3d(dtype=wp.float64),          # [N_topo]
    dij       : wp.array4d(dtype=wp.float64),         # [N_topo, n_faces]
    omega     : wp.array2d(dtype=wp.float64),    # [N_topo, n_faces]
    etoe      : wp.array2d(dtype=wp.int32),      # [N_total, n_faces]
    x         : wp.array2d(dtype=wp.float64),    # [N_total, 4]    
    result    : wp.array2d(dtype=wp.float64),    # [N_total, 4]  each thread writes its own row
):
    # slowest
    tile_i = wp.tid()
    wp.tile
    for i in range(n_topo):
        elem_idx = int(argsort[i])

        nb = wp.tile_load(etoe, (1,4), (elem_idx, 0))  # [1,4] tile of neighbor indices for this element

        # we can now load all 4 neighbor ψ vectors in one go
        # result = [N, 4]
        psi      = wp.tile_load_indexed(result, nb[0], (4,4), (0,0), axis = 0)  # [4,4] tile of neighbor ψ vectors for this element (nb[k], 0:4)
        omega_if = wp.tile_load(omega, (1,4), (i, 0))  # [1,4] tile of omega for this element
        dij_if   = wp.tile_load(dij, (1,4,4,4), (i, 0, 0, 0))  # [4,4,4] tile of dij for this element [nf, n_basis, n_basis] =[4,4,4]
        gij_invf = wp.tile_load(gij_inv, (1,4,4), (i, 0, 0))  # [1,4,4] tile of G_inv for this element
        x_tile = wp.tile_load(x, (1,4), (elem_idx, 0))  # [1,4] tile of x for this element        
        s_tile = wp.tile_zeros(shape = (4, 1), dtype=wp.float64)                


        for f in range(4):
            wp.tile_matmul(dij_if[0, f],  wp.tile_reshape(psi[f], (4,1)), s_tile, alpha = omega_if[0,f])

        result_tile = wp.tile_reshape(wp.tile_matmul(gij_invf[0], s_tile +  wp.tile_reshape(x_tile, (4,1))), (1,4))

        wp.tile_store(result, result_tile, (elem_idx, 0))  # store result back to global memory so all threads can read it after the barrier

# ── Helper ────────────────────────────────────────────────────────────────────

def _compute_level_bounds(levels_np: np.ndarray, levels_argsort_np: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), ...] index slices into levels_argsort, one per level."""
    sorted_levels      = levels_np[levels_argsort_np]
    _, first_occ       = np.unique(sorted_levels, return_index=True)
    ends               = np.append(first_occ[1:], len(sorted_levels))
    return list(zip(first_occ.tolist(), ends.tolist()))


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class WarpSweepOperator:
    """
    Holds precomputed GPU arrays for the transport sweep of a single angle.

    Geometry arrays (gij_inv, dij, omega, etoe) are uploaded once at construction
    and reused across calls to `sweep`.  Only `x` (the RHS) is re-uploaded each
    call because it changes between iterations.

    Parameters
    ----------
    transport_operator
        A single-angle transport operator (provides topological sort and domain).
    gij_inv_all : array [N_topo, 4, 4]
        Inverse element matrices in topological order.
    dij_f_p_fm_all : array [N_topo, 4, 4, 4]
        Inflow face coupling matrices in topological order.
    abs_omega_dot_n_Je : array [N_topo, 4]
        |Ω·n| * J_f for each element face in topological order.
    device : str
        Warp device string, default ``"cuda"``.
    """

    # GPU arrays (geometry, fixed across sweeps)
    gij_inv_wp    : wp.array   # mat4d dtype, shape [N_topo]
    dij_wp        : wp.array   # mat4d dtype, shape [N_topo, 4]
    gij_inv_wp_3d : wp.array   # float64, shape [N_topo, 4, 4]   for tile_matmul
    dij_wp_4d     : wp.array   # float64, shape [N_topo, 4, 4, 4] for tile_matmul
    omega_wp      : wp.array
    etoe_wp       : wp.array
    argsort_wp    : wp.array
    s_scratch_wp  : wp.array   # size-4 scratch (kept for compatibility)
    # level structure
    level_bounds : list
    # dimensions
    n_total      : int
    n_topo       : int
    device       : str

    @classmethod
    def from_transport_operator(
        cls,
        transport_operator,
        gij_inv_all,
        dij_f_p_fm_all,
        abs_omega_dot_n_Je,
        device: str = "cuda",
    ) -> "WarpSweepOperator":
        tsa      = transport_operator.topological_sorted_angle
        domain   = transport_operator.solution_domain_data.domain

        gij_inv_np = np.asarray(gij_inv_all, dtype=np.float64)
        dij_np     = np.asarray(dij_f_p_fm_all, dtype=np.float64)
        omega_np   = np.asarray(abs_omega_dot_n_Je, dtype=np.float64)
        etoe_np    = np.asarray(domain.etoe, dtype=np.int32)
        argsort_np = np.asarray(tsa.levels_argsort, dtype=np.int32)
        levels_np  = np.asarray(tsa.levels, dtype=np.int32)

        n_basis = gij_inv_np.shape[1]
        n_faces = dij_np.shape[1]
        if n_basis != 4 or n_faces != 4:
            raise ValueError(
                f"Kernel requires n_basis=n_faces=4 (degree-1 3-D simplices), "
                f"got n_basis={n_basis}, n_faces={n_faces}."
            )

        return cls(
            gij_inv_wp    = wp.array(gij_inv_np, dtype=mat4d,      device=device),
            dij_wp        = wp.array(dij_np,     dtype=mat4d,      device=device),
            gij_inv_wp_3d = wp.array(gij_inv_np, dtype=wp.float64, device=device),
            dij_wp_4d     = wp.array(dij_np,     dtype=wp.float64, device=device),
            omega_wp      = wp.array(omega_np,   dtype=wp.float64, device=device),
            etoe_wp       = wp.array(etoe_np,    dtype=wp.int32,   device=device),
            argsort_wp    = wp.array(argsort_np, dtype=wp.int32,   device=device),
            s_scratch_wp  = wp.zeros(4,          dtype=wp.float64, device=device),
            level_bounds  = _compute_level_bounds(levels_np, argsort_np),
            n_total       = etoe_np.shape[0],
            n_topo        = argsort_np.shape[0],
            device        = device,
        )

    def sweep(self, x: np.ndarray) -> np.ndarray:
        """
        Run the sweep for a given RHS ``x`` and return the result as a numpy array.

        Parameters
        ----------
        x : array [N_total, 4]
            Source vector (numpy or JAX array; will be copied to GPU).

        Returns
        -------
        result : np.ndarray [N_total, 4]
        """
        x_wp      = wp.array(np.asarray(x, dtype=np.float64), dtype=vec4d, device=self.device)
        result_wp = wp.zeros(self.n_total, dtype=vec4d, device=self.device)
        self._sweep_inplace(x_wp, result_wp)
        return result_wp.numpy()

    def sweep_sequential(self, x: np.ndarray) -> np.ndarray:
        """
        Run the sweep with a single GPU thread iterating over all elements in
        topological order.  Correct but slow — useful as a baseline when the
        level-parallel sweep is batched alongside other work.

        Parameters
        ----------
        x : array [N_total, 4]

        Returns
        -------
        result : np.ndarray [N_total, 4]
        """
        x_wp      = wp.array(np.asarray(x, dtype=np.float64), dtype=vec4d, device=self.device)
        result_wp = wp.zeros(self.n_total, dtype=vec4d, device=self.device)
        self._sweep_sequential_inplace(x_wp, result_wp)
        return result_wp.numpy()

    def _sweep_sequential_inplace(self, x_wp: wp.array, result_wp: wp.array) -> None:
        """In-place sequential sweep (single thread, no host copies)."""
        result_wp.zero_()
        wp.launch(
            kernel=_sweep_sequential_kernel,
            dim=1,
            inputs=[
                self.n_topo,
                self.argsort_wp,
                self.gij_inv_wp,
                self.dij_wp,
                self.omega_wp,
                self.etoe_wp,
                x_wp,
                result_wp,
            ],
            device=self.device,
        )
        wp.synchronize()

    def sweep_simd(self, x: np.ndarray) -> np.ndarray:
        """
        Run the sweep with 4 cooperative GPU threads — one per basis component.
        Each element is still processed sequentially, but the per-element matvec
        (s accumulation + G_inv multiply) is split across 4 threads.

        Uses a global-memory scratch buffer + wp.sync_threads() so each thread
        can read the other threads' s values before the G_inv multiply.
        Launched with dim=4, block_dim=4.

        Parameters
        ----------
        x : array [N_total, 4]

        Returns
        -------
        result : np.ndarray [N_total, 4]
        """
        x_wp      = wp.array(np.asarray(x, dtype=np.float64), dtype=wp.float64, device=self.device)
        result_wp = wp.zeros((self.n_total, 4), dtype=wp.float64, device=self.device)
        self._sweep_simd_inplace(x_wp, result_wp)
        return result_wp.numpy()

    def _sweep_simd_inplace(self, x_wp: wp.array, result_wp: wp.array) -> None:
        """In-place SIMD sweep (4 cooperative threads, global scratch, no host copies)."""
        result_wp.zero_()
        TILE_THREADS = 32
        wp.launch_tiled(
            kernel=_sweep_simd_kernel,
            dim=1,
            block_dim=TILE_THREADS,
            inputs=[
                self.n_topo,
                self.argsort_wp,
                self.gij_inv_wp_3d,
                self.dij_wp_4d,
                self.omega_wp,
                self.etoe_wp,
                x_wp,                
                result_wp,
            ],
            device=self.device,
        )
        wp.synchronize()

    def _sweep_inplace(self, x_wp: wp.array, result_wp: wp.array) -> None:
        """Run the sweep in-place on pre-allocated Warp arrays (no host copies)."""
        result_wp.zero_()
        for start, end in self.level_bounds:
            wp.launch(
                kernel=_sweep_level_kernel,
                dim=end - start,
                inputs=[
                    start,
                    self.argsort_wp,
                    self.gij_inv_wp,
                    self.dij_wp,
                    self.omega_wp,
                    self.etoe_wp,
                    x_wp,
                    result_wp,
                ],
                device=self.device,
            )
        wp.synchronize()


    
