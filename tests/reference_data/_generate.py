'''
Regenerate the frozen StellBlanket (SB) reference data used by the tests.

Run (with StellBlanket installed) from the repo root::

    python tests/reference_data/_generate.py

'''
import os
import sys

# Make ``src`` (jax_sn), ``tests`` (reference_data, finite_elements, ...) importable
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, 'src'))
sys.path.insert(0, os.path.join(_REPO, 'tests'))

os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as onp
import jax.numpy as jnp

import reference_data as ref


# ===========================================================================
# SB helpers (ported verbatim from the test files, preserving the exact calls)
# ===========================================================================
def _switch_element_type(dimension):
    import StellBlanket as SB
    if dimension == 1:
        return SB.Lines.Line_Element
    elif dimension == 2:
        return SB.Triangles.Triangle_Element
    elif dimension == 3:
        return SB.Tetrahedrons.Tetrahedron_Element
    else:
        raise ValueError("Dimension must be 1, 2 or 3.")


def _arbitrary_source_values(n_elements):
    return onp.linspace(1.0, 10.0, n_elements)


def tn_order_index():
    return 4, 3


def _angle_function_indexed(tn_order, index):
    import StellBlanket as SB
    quad_set = SB.Generic_Elements.TN_Quadrature_Set(tn_order)
    return quad_set.Directions()[index:index + 1, :]


# --- transport: mesh / source / cross sections / single-angle sweep ---------
def _test_domain_from_sb(dimension, degree):
    import StellBlanket as SB
    material_0 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(0.673))
    material_1 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(2.5))
    region_sizes = [0.8, 0.2]
    resolutions = [4, 3]
    size_other_dim = 7.54
    resolutions_other_dim = 3
    element = _switch_element_type(dimension)
    return SB.Generic_Elements.Create_Slab(element, region_sizes, resolutions, [material_0, material_1], size_other_dim, resolutions_other_dim)


def _matfree_nte_transport(dimension, degree, tn_order, angle_index):
    import StellBlanket as SB
    domain = _test_domain_from_sb(dimension, degree)
    quad_set = SB.Generic_Elements.Quadrature_Set(onp.array(_angle_function_indexed(tn_order, angle_index)), onp.array([1.0]))
    energy_set = SB.Generic_Elements.Energy_Set([0, 1])
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, domain, degree, 0)
    source_values = _arbitrary_source_values(domain.connectivity.shape[0])
    sources = [SB.Generic_Elements.Constant_Source(_switch_element_type(dimension), val, energy_set[0]) for val in source_values]
    total_source = SB.Generic_Elements.Per_Element_Source(sources, domain)
    return matfreente, total_source


def gen_transport_bundle(dimension, degree):
    '''verts, conn, source array, cross-section (indices/total/scattering), permuted sweep reference.'''
    from finite_elements.test_finite_elements import create_permutation_dimension

    domain = _test_domain_from_sb(dimension, degree)
    verts = onp.asarray(domain.vertices)
    conn = onp.asarray(domain.connectivity)

    nte, total_source = _matfree_nte_transport(dimension, degree, *tn_order_index())
    n_elements = nte.geometric_domain.connectivity.shape[0]

    # source: linspace per element, tiled over the per-element basis count
    source_values = _arbitrary_source_values(n_elements)
    n_values = int(nte.solution_domain.total_size / n_elements)
    source = onp.stack([source_values] * n_values, axis=-1)

    # cross sections: per-element total, uniquified into (indices -> vals)
    cross_sections = onp.array([nte.geometric_domain[i].Material().Cross_Section().Value_Total(nte.energy_set[0]) for i in range(n_elements)]).reshape((n_elements, 1))
    vals, inverse = onp.unique(cross_sections[:, 0], return_inverse=True)
    xs_scattering = onp.zeros((vals.shape[0], 1, 1, 1))
    xs_total = onp.array(vals).reshape((vals.shape[0], 1))

    # single-angle sweep reference, permuted to jax-sn DOF ordering
    nte.simulation.Sweep(nte.energy_set[0], nte.quadrature_set[0], total_source)
    values = nte.global_solution.All_Solution_Values(nte.energy_set[0], nte.quadrature_set[0]).reshape(n_elements, -1)
    permutation = create_permutation_dimension(dimension, degree)
    expected = onp.asarray(values[:, permutation])

    return dict(verts=verts, conn=conn, source=source,
                xs_indices=onp.asarray(inverse), xs_total=xs_total, xs_scattering=xs_scattering,
                expected=expected)


def gen_transport_mesh_only(dimension, degree):
    '''Only verts/conn (used by test_local_real_inversion at degree 1).'''
    domain = _test_domain_from_sb(dimension, degree)
    return dict(verts=onp.asarray(domain.vertices), conn=onp.asarray(domain.connectivity))


def gen_transport_angle():
    return onp.asarray(_angle_function_indexed(*tn_order_index())[0])


# --- moment: multi-angle moment solution ------------------------------------
def _matfree_nte_moment(dimension, degree, lmax, tn_order):
    import StellBlanket as SB
    domain = _test_domain_from_sb(dimension, degree)
    quad_set = SB.Generic_Elements.TN_Quadrature_Set(tn_order)
    energy_set = SB.Generic_Elements.Energy_Set([0, 1])
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, domain, degree, lmax)
    source_values = _arbitrary_source_values(domain.connectivity.shape[0])
    sources = [SB.Generic_Elements.Constant_Source(_switch_element_type(dimension), val, energy_set[0]) for val in source_values]
    total_source = SB.Generic_Elements.Per_Element_Source(sources, domain)
    return matfreente, total_source


def _lm_sb_to_norm(l, m):
    return jnp.where(m == 0, l ** 2, jnp.where(m < 0, l ** 2 + l + abs(m), l ** 2 + abs(m)))


def _convert_expansion_values_sb_to_norm(sb_values):
    import jax_sn
    lmax = int(onp.sqrt(sb_values.shape[-1])) - 1
    l, m = jax_sn.quadrature_set.quadrature_set_base._lm_real_expansion(lmax)
    permutation = _lm_sb_to_norm(l, m)
    sb_values__perm = sb_values[..., permutation]
    return onp.moveaxis(onp.where(m != 0, sb_values__perm * onp.sqrt(2), sb_values__perm) * onp.sqrt(4 * onp.pi), -1, 0)


def gen_moment(dimension, degree, tn_order, lmax):
    from finite_elements.test_finite_elements import create_permutation_dimension

    mfreente, source = _matfree_nte_moment(dimension, degree, lmax, tn_order)
    mfreente.simulation.Total_Sweep(mfreente.energy_set[0], source)
    permutation = create_permutation_dimension(dimension, degree)
    mfreente.global_moment_solution.Setup_Energy_Group(mfreente.energy_set[0])
    sb_values = mfreente.global_moment_solution.Moment_Solution_Values(mfreente.energy_set[0]).reshape(-1, permutation.shape[0], (lmax + 1) ** 2)[:, permutation, :]
    return dict(expected=onp.asarray(_convert_expansion_values_sb_to_norm(sb_values)))


# --- lm expansions ----------------------------------------------------------
def gen_lm(tn_set, l_max):
    '''Reference real spherical-harmonic table, converted to jax-sn convention.

    SB uses a different l,m ordering and normalisation; we bake that conversion in
    here so the stored reference is directly comparable to jax-sn output.
    '''
    import StellBlanket as SB
    import jax_sn
    qs_sb = SB.Generic_Elements.TN_Quadrature_Set(tn_set)
    values_sb = onp.asarray(qs_sb.Calculate_Yklm_real(l_max))
    l, m = jax_sn.quadrature_set.quadrature_set_base._lm_real_expansion(l_max)
    m = onp.asarray(m)
    indices = onp.asarray(_lm_sb_to_norm(l, m))
    converted = onp.where(m != 0, values_sb[..., indices] * onp.sqrt(2), values_sb[..., indices]) / onp.sqrt(4 * onp.pi)
    return dict(yklm=onp.asarray(converted))


# --- anisotropic scattering -------------------------------------------------
def _anisotropic_sb(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order):
    import StellBlanket as SB
    water = SB.Generic_Elements.Material(SB.Generic_Elements.Multigroup_Cross_Section(onp.array([15.2090, 24.4140]), onp.array([[[5.5069, 0.0],
                                                                        [6.6227, 11.4080]],
                                                                        [[1.158567, 0.0],
                                                                         [1.1234767, 2.4733]]])))
    iron = SB.Generic_Elements.Material(SB.Generic_Elements.Multigroup_Cross_Section(onp.array([30.3110, 26.8760]), onp.array([[[18.25, 0.0],
                                                                       [6.45294, 21.3570]],
                                                                      [[4.363, 0.0],
                                                                       [-0.003508, 2.3457]]])))
    void = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(1e-10))
    region_sizes = [0.2, 0.1, 0.5, 0.2, 0.1, 0.1, 0.1]
    xs_regions = [water, water, water, iron, void, water, void]
    quad_set = SB.Generic_Elements.Quadrature_Set(*SB.Generic_Elements.create_TN(tn_order))
    element = _switch_element_type(dim)
    energy_set = SB.Generic_Elements.Energy_Set([2, 1, 0])
    source_water = SB.Generic_Elements.Constant_Multigroup_Source(element, onp.array([9.7702e4, 4.5441e5]), energy_set)
    source_void = SB.Generic_Elements.Zero_Source(element)
    sources = [source_void, source_water, source_void, source_void, source_void, source_void, source_void]
    cartesian_domain, source = SB.Generic_Elements.Create_Slab(element, region_sizes, n_elem_regions, xs_regions, d_other_dim, n_other_dim, sources)
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, cartesian_domain, degree, 1)
    return matfreente, source


def gen_anisotropic(dim, degree):
    import StellBlanket as SB
    from finite_elements.test_anisotropic_scattering import _get_anisotropic_args, _interp_x

    args_ani = _get_anisotropic_args(dim, degree)
    nte, sb_source = _anisotropic_sb(*args_ani)
    solution_options = SB.Generic_Elements.Solution_Options(tolerance=1e-20)
    nte.Solve(nte.energy_set[0], sb_source, solution_options.Return_Dictionary())

    x = onp.array(_interp_x())
    xyz = onp.stack([x, onp.zeros_like(x), onp.zeros_like(x)], axis=-1)
    phi = SB.Generic_Elements.Interpolate_Moments(xyz, nte.global_moment_solution, nte.energy_set[0], 0, 0)
    return dict(phi=onp.asarray(phi * onp.sqrt(4 * onp.pi)))


# --- breeding blanket -------------------------------------------------------
def _bb_sb(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order):
    import StellBlanket as SB
    from finite_elements.test_breeding_blanket import bb_data_dict, base_region_sizes

    xs_data_dict = bb_data_dict()
    region_sizes = base_region_sizes()
    region_names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone', 'BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']
    xs_regions = [SB.Generic_Elements.Material(SB.Generic_Elements.Multigroup_Cross_Section(xs_data_dict[i][0], xs_data_dict[i][1])) for i in region_names]
    quad_set = SB.Generic_Elements.Quadrature_Set(*SB.Generic_Elements.create_TN(tn_order))
    element = _switch_element_type(dim)
    energy_set = SB.Generic_Elements.Energy_Set(onp.arange(xs_data_dict['Plasma'][0].shape[0] + 1))
    source_p = SB.Generic_Elements.Constant_Source(element, 1.0, energy_set[0])
    source_0 = SB.Generic_Elements.Zero_Source(element)
    sources = [source_p if i == 0 else source_0 for i in range(len(xs_regions))]
    lmax = xs_data_dict['Plasma'][1].shape[0] - 1
    cartesian_domain, source = SB.Generic_Elements.Create_Slab(element, region_sizes, n_elem_regions, xs_regions, d_other_dim, n_other_dim, sources)
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, cartesian_domain, degree, lmax)
    return matfreente, source


def gen_bb(dim, degree):
    import StellBlanket as SB
    from finite_elements.test_breeding_blanket import _get_bb_args, _interp_x

    dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order = _get_bb_args(dim, degree)
    nte, source = _bb_sb(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order)
    solution_options = SB.Generic_Elements.Solution_Options(tolerance=1e-20, maximum_iterations=200)
    n_groups = nte.energy_set.Boundaries().shape[0] - 1
    SB.Generic_Elements.Run_Forward(nte, source, solution_options)

    x = onp.array(_interp_x())
    xyz = onp.stack([x, onp.zeros_like(x), onp.zeros_like(x)], axis=-1)
    phisb = [SB.Generic_Elements.Interpolate_Moments(xyz, nte.global_moment_solution, nte.energy_set[i], 0, 0) * onp.sqrt(4 * onp.pi) for i in range(n_groups)]
    return dict(phi=onp.asarray(phisb))


# ===========================================================================
# Enumerate every combination the tests request, generate, and save.
# ===========================================================================
# transport bundles (mesh + source + xs + single-angle sweep reference)
_TRANSPORT_FULL = [(1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (1, 5)]
# transport meshes only (degree-1, test_local_real_inversion)
_TRANSPORT_MESH_ONLY = [(1, 1), (2, 1), (3, 1)]
# moment: (dim, degree, lmax, tn_order)
_MOMENT = [(1, 5, 3, 3), (2, 3, 5, 4), (3, 2, 4, 3)]
_LM_TN_SET = 8
_LM_LMAX = [0, 1, 2, 3, 4, 5]
_DIMS = [1, 2, 3]
_DEGREE = 2


def main():
    print(f"Writing reference data to {ref.DATA_DIR}")

    ref.save(ref.key_transport_angle(), angle=gen_transport_angle())
    print("  transport angle")

    for dim, degree in _TRANSPORT_FULL:
        ref.save(ref.key_transport(dim, degree), **gen_transport_bundle(dim, degree))
        print(f"  transport d{dim} deg{degree}")
    for dim, degree in _TRANSPORT_MESH_ONLY:
        ref.save(ref.key_transport(dim, degree), **gen_transport_mesh_only(dim, degree))
        print(f"  transport (mesh only) d{dim} deg{degree}")

    for dim, degree, lmax, tn_order in _MOMENT:
        ref.save(ref.key_moment(dim, degree, lmax, tn_order), **gen_moment(dim, degree, tn_order, lmax))
        print(f"  moment d{dim} deg{degree} lmax{lmax} tn{tn_order}")

    for l_max in _LM_LMAX:
        ref.save(ref.key_lm(_LM_TN_SET, l_max), **gen_lm(_LM_TN_SET, l_max))
        print(f"  lm tn{_LM_TN_SET} lmax{l_max}")

    for dim in _DIMS:
        ref.save(ref.key_anisotropic(dim), **gen_anisotropic(dim, _DEGREE))
        print(f"  anisotropic d{dim}")

    for dim in _DIMS:
        ref.save(ref.key_bb(dim), **gen_bb(dim, _DEGREE))
        print(f"  bb d{dim}")

    print("Done.")


if __name__ == '__main__':
    main()
