'''
Plotly functions for visualizing domains, solutions, elements, quadrature sets, and the derivation
graphs the builders are written in.

Used by docs for interactive visualization; not a core part of the library and not eagerly imported.
'''
import numpy as onp
import jax.numpy as jnp
import plotly.graph_objects as go
import plotly.colors as pc

_DEFAULT_COLORS = pc.qualitative.Plotly


def _color(colors, idx):
    palette = colors or _DEFAULT_COLORS
    return palette[idx % len(palette)]


def _material_indices(domain, color_by_material):
    n_elements = onp.asarray(domain.connectivity).shape[0]
    if not color_by_material:
        return onp.zeros(n_elements, dtype=int)
    return onp.asarray(domain.cross_sections_indexed.indices)


def _material_label(material, material_names):
    '''`material_names` may be a {index: name} mapping or a name-per-index sequence.'''
    if material_names is None:
        return f"material {material}"
    if isinstance(material_names, dict):
        return material_names.get(material, f"material {material}")
    return material_names[material]


def _undirected_adjacency_edges(domain):
    '''Every `etoe` neighbour pair, once each (etoe is symmetric).'''
    etoe     = onp.asarray(domain.etoe)
    elem_idx = onp.repeat(onp.arange(etoe.shape[0]), etoe.shape[1])
    neighbor = etoe.reshape(-1)
    mask     = (neighbor >= 0) & (elem_idx < neighbor)
    return onp.stack([elem_idx[mask], neighbor[mask]], axis=1)


def _pad_to_3d(points):
    points = onp.atleast_2d(onp.asarray(points, dtype=float))
    pad_width = 3 - points.shape[1]
    return onp.pad(points, [(0, 0), (0, pad_width)]) if pad_width > 0 else points


def _pad_angle_to_3d(angle):
    '''`face_normals` (and hence `compute_face_inflow`) are always 3-component, regardless of
    the domain's actual spatial dimension -- so the angle needs the same padding.'''
    return jnp.asarray(_pad_to_3d(angle)[0])


def _directed_sweep_edges(domain, angle):
    '''(upstream, downstream) element pairs for the sweep dependency graph at `angle`.'''
    from jax_sn.domain.graph_generation import compute_face_inflow

    etoe   = onp.asarray(domain.etoe)
    inflow = onp.asarray(compute_face_inflow(domain, _pad_angle_to_3d(angle)))
    mask   = inflow & (etoe >= 0)
    downstream, face = onp.where(mask)
    upstream = etoe[downstream, face]
    return onp.stack([upstream, downstream], axis=1)


# In 1D the mesh itself is drawn on the y=0 baseline (with boundary ticks spanning [-0.5, 0.5]
# and the axis range fixed to [-1, 1]) -- graph overlays (nodes/edges, level labels) are lifted
# above that so they don't sit on top of the mesh line.
_GRAPH_Y_OFFSET_1D = 0.7


def _element_centroids(dim, domain):
    vertices     = onp.asarray(domain.vertices)
    connectivity = onp.asarray(domain.connectivity)
    centroids    = _pad_to_3d(vertices[connectivity].mean(axis=1))
    if dim == 1:
        centroids[:, 1] += _GRAPH_Y_OFFSET_1D
    return centroids


def _element_levels(domain, graph_angle):
    from jax_sn.domain.graph_generation import compute_levels_domain_safe
    n_elements = onp.asarray(domain.connectivity).shape[0]
    return onp.asarray(compute_levels_domain_safe(domain, _pad_angle_to_3d(graph_angle),
                                                   n_iterations_per_loop=n_elements))


def _element_topological_order(domain, graph_angle):
    '''
    Each element's position in the sweep's topological order (0 = computed first), the inverse
    permutation of `jnp.argsort(levels)` -- exactly how `BasicSweepData.levels_argsort` is built
    from the same per-element `levels` for a single angle (see `sweep_data.py`). Unlike
    `_element_levels`, which groups mutually-independent (same-level) elements under one shared
    number, this gives every element its own distinct rank, ties within a level broken by
    (stable-sort) element index -- the actual sequential order a serial sweep computes them in.
    '''
    levels = _element_levels(domain, graph_angle)
    return onp.argsort(onp.argsort(levels, kind="stable"), kind="stable")


def _bearing_degrees(delta):
    '''Compass bearing (0 = +y/"up", clockwise) matching `marker.angle`'s default `angleref="up"`,
    so a `symbol="triangle-up"` rotated by this angle points along `delta`.'''
    return (90.0 - onp.degrees(onp.arctan2(delta[:, 1], delta[:, 0]))) % 360.0


def _add_angle_compass(fig, dim, domain, angle, *, z=None):
    '''
    A small arrow, legend-labelled "Angle", showing the overall sweep direction `angle` itself
    (as opposed to the per-edge arrowheads, which show the resulting element dependencies).
    Placed just outside the domain's bounding box, scaled to its extent, as a real trace (so it
    gets a legend entry, unlike a paper-space annotation) -- in 3D this also means it rotates
    correctly with the scene camera.

    `z`, if given, overrides the tail/tip height and forces the `Scatter3d`+`Cone` (3D-scene)
    rendering regardless of `dim` -- for `plot_2d_solution`, whose figure is always a 3D `scene`
    (even for a flat, `elevate=False` field) so the plain 2D `Scatter` this function otherwise
    draws for `dim` 1/2 (`plot_domain`'s own, non-scene figures) wouldn't render inside it. A 2D
    domain has no inherent z extent to place the compass at, hence the override.
    '''
    direction = _pad_to_3d(angle)[0]
    norm = onp.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    vertices = onp.asarray(domain.vertices)
    if dim == 1:
        # the mesh itself is 1D; keep the compass horizontal, above the graph overlay.
        extent = max(float(vertices[:, 0].max() - vertices[:, 0].min()), 1e-9)
        tail   = onp.array([vertices[:, 0].min() - 0.05 * extent, _GRAPH_Y_OFFSET_1D + 0.15, 0.0])
    else:
        vertices_3d = _pad_to_3d(vertices)
        bbox_min, bbox_max = vertices_3d.min(axis=0), vertices_3d.max(axis=0)
        extent = max(float(onp.max(bbox_max - bbox_min)), 1e-9)
        tail   = bbox_min - 0.05 * extent
    if z is not None:
        tail = tail.copy()
        tail[2] = z
    tip = tail + direction * 0.08 * extent

    angle_color = "red"  # matches the default plotly qualitative palette's first color

    if dim == 3 or z is not None:
        fig.add_trace(go.Scatter3d(
            x=[tail[0], tip[0]], y=[tail[1], tip[1]], z=[tail[2], tip[2]], mode="lines",
            line=dict(width=4, color=angle_color), name="Angle", legendgroup="angle",
            showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[direction[0]], v=[direction[1]], w=[direction[2]],
            sizemode="absolute", sizeref=0.035 * extent, anchor="tail", showscale=False,
            colorscale=[[0, angle_color], [1, angle_color]], name="Angle", legendgroup="angle",
            showlegend=False, hoverinfo="skip",
        ))
    else:
        bearing = _bearing_degrees(onp.array([direction[:2]]))[0]
        fig.add_trace(go.Scatter(
            x=[tail[0], tip[0]], y=[tail[1], tip[1]], mode="lines+markers",
            line=dict(width=3, color=angle_color),
            marker=dict(size=[0, 12], symbol=["circle", "triangle-up"], color=angle_color,
                        angle=[0, bearing]),
            name="Angle", showlegend=True, hoverinfo="skip",
        ))


def _add_graph_and_levels(fig, dim, domain, *, show_graph, graph_angle, show_levels, graph_color,
                           level_mode, show_element_numbers=False):
    '''
    Overlay the element-adjacency graph and/or per-element sweep levels/order/index, all in
    `graph_color` so the overlay reads as one layer distinct from the (black-outlined) mesh
    underneath.

    Nodes: a plain dot if neither `show_levels` nor `show_element_numbers` is set, otherwise a
    white/`graph_color`-outlined circle with a number inside it. `show_levels` requires
    `graph_angle` (angle-dependent); `show_element_numbers` does not (it's just each element's
    own index, `range(n_elements)`) and takes priority if both are somehow requested with
    conflicting values -- in practice callers set at most one of the two. `level_mode` picks what
    `show_levels`'s number is: `"levels"` is the sweep level (topological-sort *group* --
    elements in the same level are mutually independent and solved in parallel, from
    `compute_levels_domain_safe`), `"order"` is each element's own position in the topological
    sort (0 = computed first, see `_element_topological_order`) -- the actual per-element
    sequential order a serial sweep processes them in.

    Edges (only when `show_graph`): with `graph_angle` given, the directed sweep-dependency graph
    for that angle (upstream -> downstream), with a small arrowhead at the midpoint of each edge
    (a rotated triangle in 1D/2D; `Scatter3d` markers can't be rotated, so 3D falls back to an
    off-centre dot). Without `graph_angle`, edges are the plain undirected `etoe` adjacency.
    '''
    centroids   = _element_centroids(dim, domain)
    scatter_cls = go.Scatter3d if dim == 3 else go.Scatter

    def _trace(x, y, z, **kwargs):
        extra = dict(z=z) if dim == 3 else {}
        return scatter_cls(x=x, y=y, **extra, showlegend=False, hoverinfo="skip", **kwargs)

    if show_graph:
        directed = graph_angle is not None
        edges = _directed_sweep_edges(domain, graph_angle) if directed else _undirected_adjacency_edges(domain)

        edge_x, edge_y, edge_z = [], [], []
        for src, dst in edges:
            p0, p1 = centroids[src], centroids[dst]
            edge_x += [p0[0], p1[0], None]
            edge_y += [p0[1], p1[1], None]
            edge_z += [p0[2], p1[2], None]
        fig.add_trace(_trace(edge_x, edge_y, edge_z, mode="lines",
                              line=dict(width=1.5, color=graph_color), name="graph edges"))

        if directed and len(edges) > 0:
            src, dst = centroids[edges[:, 0]], centroids[edges[:, 1]]
            mid = (src + dst) / 2
            if dim == 3:
                # Scatter3d markers have no rotation, so approximate the arrowhead with a dot
                # offset toward the downstream element instead of a true oriented triangle.
                tip = src * 0.25 + dst * 0.75
                fig.add_trace(_trace(tip[:, 0], tip[:, 1], tip[:, 2], mode="markers",
                                      marker=dict(size=4, color=graph_color, symbol="circle"),
                                      name="sweep direction"))
            else:
                bearing = _bearing_degrees(dst - src)
                fig.add_trace(_trace(mid[:, 0], mid[:, 1], None, mode="markers",
                                      marker=dict(size=9, color=graph_color, symbol="triangle-up",
                                                  angle=bearing),
                                      name="sweep direction"))
            _add_angle_compass(fig, dim, domain, graph_angle)

    if show_element_numbers:
        values, name = onp.arange(centroids.shape[0]), "element"
    elif show_levels:
        values = (_element_levels(domain, graph_angle) if level_mode == "levels"
                  else _element_topological_order(domain, graph_angle))
        name = level_mode
    else:
        values = None

    if values is not None:
        fig.add_trace(_trace(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], mode="markers+text",
            marker=dict(size=20, color="white", line=dict(color=graph_color, width=1.5), symbol="circle"),
            text=[str(v) for v in values], textposition="middle center",
            textfont=dict(size=10, color=graph_color), name=name,
        ))
    elif show_graph:
        fig.add_trace(_trace(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], mode="markers",
            marker=dict(size=5, color=graph_color, symbol="circle"), name="graph nodes",
        ))


def _plot_domain_1d(domain, *, color_by_material, show_element_boundaries, colors, material_names,
                     show_graph, graph_angle, show_levels, graph_color, mesh_opacity, level_mode,
                     show_element_numbers=False):
    vertices     = onp.asarray(domain.vertices)[:, 0]
    connectivity = onp.asarray(domain.connectivity)
    materials    = _material_indices(domain, color_by_material)

    fig = go.Figure()
    for color_idx, material in enumerate(onp.unique(materials)):
        elem_idx = onp.where(materials == material)[0]
        x, y = [], []
        for e in elem_idx:
            v0, v1 = connectivity[e]
            x += [vertices[v0], vertices[v1], None]
            y += [0.0, 0.0, None]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers" if show_element_boundaries else "lines",
            line=dict(width=4, color=_color(colors, color_idx)),
            marker=dict(size=6, color=_color(colors, color_idx)), opacity=mesh_opacity,
            name=_material_label(material, material_names), showlegend=color_by_material,
        ))

    if show_element_boundaries:
        # a vertical tick at every vertex, marking element boundaries.
        tick_x, tick_y = [], []
        for v in onp.unique(vertices):
            tick_x += [v, v, None]
            tick_y += [-0.5, 0.5, None]
        fig.add_trace(go.Scatter(
            x=tick_x, y=tick_y, mode="lines", opacity=mesh_opacity,
            line=dict(width=1, color="black"), showlegend=False, hoverinfo="skip",
        ))

    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(xaxis_title="x", margin=dict(l=0, r=0, t=40, b=40))
    if show_graph or show_levels or show_element_numbers:
        _add_graph_and_levels(fig, 1, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels, graph_color=graph_color, level_mode=level_mode,
                               show_element_numbers=show_element_numbers)
    return fig


def _plot_domain_2d(domain, *, color_by_material, show_element_boundaries, colors, material_names,
                     show_graph, graph_angle, show_levels, graph_color, mesh_opacity, level_mode,
                     show_element_numbers=False):
    vertices     = onp.asarray(domain.vertices)
    connectivity = onp.asarray(domain.connectivity)
    materials    = _material_indices(domain, color_by_material)

    fig = go.Figure()
    for color_idx, material in enumerate(onp.unique(materials)):
        elem_idx = onp.where(materials == material)[0]
        for k, e in enumerate(elem_idx):
            tri = vertices[connectivity[e]]
            x = list(tri[:, 0]) + [tri[0, 0]]
            y = list(tri[:, 1]) + [tri[0, 1]]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", fill="toself", opacity=mesh_opacity,
                line=dict(width=1 if show_element_boundaries else 0, color="black"),
                fillcolor=_color(colors, color_idx),
                name=_material_label(material, material_names), legendgroup=f"material-{material}",
                showlegend=color_by_material and k == 0,
            ))
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    if show_graph or show_levels or show_element_numbers:
        _add_graph_and_levels(fig, 2, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels, graph_color=graph_color, level_mode=level_mode,
                               show_element_numbers=show_element_numbers)
    return fig


def _extract_faces(domain, *, color_by_material, show_material_interfaces):
    '''
    Group the domain's visible faces by material: a face is visible if it is on the domain's
    own exterior boundary (no element-to-element neighbour), or -- when both `color_by_material`
    and `show_material_interfaces` are set -- if its neighbour belongs to a different material,
    so that internal material regions become visible inside a volumetric mesh. A shared interface
    face is emitted once per side (i.e. once into each of the two materials it borders), since it
    is genuinely part of both regions' boundaries.
    '''
    connectivity  = onp.asarray(domain.connectivity)
    face_template = onp.asarray(domain.face_template)
    etoe          = onp.asarray(domain.etoe)
    materials     = _material_indices(domain, color_by_material)

    groups = {}
    for local_face in range(face_template.shape[0]):
        neighbor          = etoe[:, local_face]
        is_boundary       = neighbor < 0
        neighbor_material = materials[onp.clip(neighbor, 0, None)]
        is_interface      = color_by_material & show_material_interfaces & ~is_boundary & (neighbor_material != materials)
        elem_idx          = onp.where(is_boundary | is_interface)[0]
        faces             = connectivity[elem_idx][:, face_template[local_face]]
        for m, f in zip(materials[elem_idx], faces):
            groups.setdefault(int(m), []).append(f)
    return {m: onp.array(f) for m, f in groups.items()}


def _triangle_edges(vertices, faces):
    edges = onp.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    x, y, z = [], [], []
    for a, b in edges:
        x += [vertices[a, 0], vertices[b, 0], None]
        y += [vertices[a, 1], vertices[b, 1], None]
        z += [vertices[a, 2], vertices[b, 2], None]
    return x, y, z


def _plot_domain_3d(domain, *, color_by_material, show_element_boundaries, show_material_interfaces,
                     colors, material_names, show_graph, graph_angle, show_levels, graph_color,
                     mesh_opacity, level_mode, show_element_numbers=False):
    vertices = onp.asarray(domain.vertices)
    groups   = _extract_faces(domain, color_by_material=color_by_material,
                               show_material_interfaces=show_material_interfaces)

    fig = go.Figure()
    for color_idx, (material, faces) in enumerate(sorted(groups.items())):
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=_color(colors, color_idx), opacity=0.6 * mesh_opacity, flatshading=True,
            name=_material_label(material, material_names), showlegend=color_by_material,
        ))
        if show_element_boundaries:
            edge_x, edge_y, edge_z = _triangle_edges(vertices, faces)
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode="lines", opacity=mesh_opacity,
                line=dict(width=2, color="black"), showlegend=False, hoverinfo="skip",
            ))
    fig.update_layout(scene=dict(aspectmode="data"))
    if show_graph or show_levels or show_element_numbers:
        _add_graph_and_levels(fig, 3, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels, graph_color=graph_color, level_mode=level_mode,
                               show_element_numbers=show_element_numbers)
    return fig


def _dg_element_profile(solution_domain, solution, n_points_per_element):
    '''
    Sample `solution` (one `[n_elements, n_basis]` coefficient array) at `n_points_per_element`
    points local to *each* element independently -- never through `SolutionDomain.interpolate`,
    whose `containing_cell` search picks a single owning element per point and so silently
    smears any jump at a shared node/face across two neighbours. Returns flat `(x, y)` lists
    with a `None` inserted between elements, so a single Plotly `Scatter(mode="lines")` trace
    draws each element as its own segment and leaves a true gap -- rather than a connecting
    line -- at every discontinuity.
    '''
    element      = solution_domain.element
    domain       = solution_domain.domain
    n_elements   = solution_domain.n_elements
    vertices     = onp.asarray(domain.vertices)[:, 0]
    connectivity = onp.asarray(domain.connectivity)

    xi = jnp.linspace(0.0, 1.0, n_points_per_element)
    local_coordinates = jnp.tile(xi[:, None], (n_elements, 1))
    element_indices   = jnp.repeat(jnp.arange(n_elements), n_points_per_element)
    values = onp.asarray(element.interpolate_from_local_coordinate(
        local_coordinates, element_indices, solution)).reshape(n_elements, n_points_per_element)

    v0, v1 = vertices[connectivity[:, 0]], vertices[connectivity[:, 1]]
    x = v0[:, None] + onp.asarray(xi)[None, :] * (v1 - v0)[:, None]

    x_flat, y_flat = [], []
    for e in range(n_elements):
        x_flat += list(x[e]) + [None]
        y_flat += list(values[e]) + [None]
    return x_flat, y_flat


def plot_1d_solution(solution_domain, solution, *, reference=None, n_points_per_element=20,
                      n_reference_points=200, solution_names=None, reference_name="Analytical",
                      colors=None, xaxis_title="x", yaxis_title="ψ", width=None, height=None):
    '''
    Plot one or more DG solution profiles on a 1D `SolutionDomain`, optionally against a
    continuous analytical reference curve. Each element is sampled and drawn independently
    (see `_dg_element_profile`) so that jumps across element faces render as real gaps in the
    line, instead of `SolutionDomain.interpolate` on a shared grid, which would silently
    connect across the discontinuity.

    Parameters
    ----------
    solution_domain : jax_sn.solution_domain.SolutionDomain
        Must be 1D (`solution_domain.dimension == 1`).
    solution : jnp.ndarray [n_elements, n_basis] or [n_solutions, n_elements, n_basis]
        One or more solution coefficient arrays (e.g. `psi` from a transport-operator solve).
    reference : Callable[[jnp.ndarray], jnp.ndarray], optional
        A closed-form reference, called as `reference(x)` with `x` of shape `[n_reference_points, 1]`
        and expected to return values broadcastable to `[n_reference_points]`; drawn as a
        continuous dashed black line (analytical solutions have no element-wise discontinuities
        to preserve, so they're sampled on one shared dense grid instead of per element).
    n_points_per_element : int
        Points sampled per element for each `solution` row.
    n_reference_points : int
        Points sampled (uniformly over the domain's vertex extent) for `reference`.
    solution_names : Sequence[str], optional
        Legend label per row of `solution`; defaults to "Numerical" for a single profile,
        else "solution {i}".
    reference_name : str
        Legend label for `reference`.
    colors : Sequence[str], optional
        Colour palette to cycle through per solution profile; defaults to plotly's
        qualitative `Plotly` palette.
    xaxis_title, yaxis_title : str
        Axis labels.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly
        autosizes to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    assert solution_domain.dimension == 1, f"plot_1d_solution requires a 1D solution_domain, got dimension {solution_domain.dimension}"
    solution = onp.atleast_3d(onp.asarray(solution).reshape((-1,) + solution_domain.elements_basis_shape.shape))

    fig = go.Figure()
    if reference is not None:
        vertices = onp.asarray(solution_domain.domain.vertices)[:, 0]
        x_ref = jnp.linspace(vertices.min(), vertices.max(), n_reference_points)[:, None]
        fig.add_trace(go.Scatter(
            x=x_ref[:, 0], y=onp.asarray(reference(x_ref)).reshape(-1), mode="lines",
            line=dict(width=2, color="black", dash="dash"), name=reference_name,
        ))
    for idx, solution_i in enumerate(solution):
        if solution_names is not None:
            name = solution_names[idx]
        else:
            name = "Numerical" if solution.shape[0] == 1 else f"solution {idx}"
        x, y = _dg_element_profile(solution_domain, solution_i, n_points_per_element)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", line=dict(width=3, color=_color(colors, idx)), name=name,
        ))

    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title, margin=dict(l=0, r=0, t=40, b=40),
                       width=width, height=height)
    return fig


def _reference_triangle_subdivision(n_subdivisions):
    '''
    Uniformly refine the reference triangle (local coordinates (0,0), (1,0), (0,1), matching
    `connectivity`'s per-element vertex order -- see `_real_coordinate_simplex` in `basix_fem.py`)
    into `n_subdivisions ** 2` congruent sub-triangles. `n_subdivisions=1` is a no-op: just the
    triangle's own 3 corners, one sub-triangle -- exact for a P1 field, since `Mesh3d` linearly
    (Gouraud) interpolates colour between a face's vertices. Higher values sample more points
    per element, needed for a visually smooth gradient from higher-degree fields.

    Returns
    -------
    local_coordinates : onp.ndarray [n_local_points, 2]
    triangles : onp.ndarray [n_local_triangles, 3]
        Index triples into `local_coordinates`.
    '''
    n = n_subdivisions
    i, j = onp.mgrid[0:n + 1, 0:n + 1]
    mask = (i + j) <= n
    i, j = i[mask], j[mask]
    local_coordinates = onp.stack([i, j], axis=1) / n

    index = -onp.ones((n + 1, n + 1), dtype=int)
    index[i, j] = onp.arange(len(i))

    triangles = []
    for a in range(n):
        for b in range(n - a):
            triangles.append([index[a, b], index[a + 1, b], index[a, b + 1]])
            if a + b < n - 1:
                triangles.append([index[a + 1, b], index[a + 1, b + 1], index[a, b + 1]])
    return local_coordinates, onp.array(triangles)


def _dg_element_field_2d(solution_domain, solution, n_subdivisions):
    '''
    Sample `solution` (one `[n_elements, n_basis]` coefficient array) on a fine, per-element
    sub-triangulation of the reference triangle (`_reference_triangle_subdivision`), evaluated
    independently for *each* element -- exactly as `_dg_element_profile` does for 1D -- so the
    returned `triangles` never index a vertex shared between two elements. A `Mesh3d` built from
    this therefore renders each element as its own disconnected coloured patch, preserving jumps
    across element faces instead of blending them the way `SolutionDomain.interpolate`'s
    single-owner `containing_cell` lookup would.

    Returns
    -------
    x, y : onp.ndarray [n_elements * n_local_points]
        Real-space coordinates of every sampled point.
    values : onp.ndarray [n_elements * n_local_points]
        `solution` evaluated at each point.
    triangles : onp.ndarray [n_elements * n_local_triangles, 3]
        Index triples into `x`/`y`/`values`.
    '''
    element    = solution_domain.element
    domain     = solution_domain.domain
    n_elements = solution_domain.n_elements

    local_coords, local_triangles = _reference_triangle_subdivision(n_subdivisions)
    n_local = local_coords.shape[0]

    local_coordinates = jnp.tile(jnp.asarray(local_coords), (n_elements, 1))
    element_indices   = jnp.repeat(jnp.arange(n_elements), n_local)

    values = onp.asarray(element.interpolate_from_local_coordinate(
        local_coordinates, element_indices, solution))
    real = onp.asarray(element.real_coordinates(
        local_coordinates, element_indices, domain, solution_domain.precomputed_geometric_data))

    offsets   = (onp.arange(n_elements) * n_local)[:, None, None]
    triangles = (local_triangles[None, :, :] + offsets).reshape(-1, 3)
    return real[:, 0], real[:, 1], values, triangles


def plot_2d_solution(solution_domain, solution, *, n_subdivisions=4, colorscale="Viridis",
                      colorbar_title="ψ", show_element_boundaries=False, elevate=False, z_scale=1.0,
                      zmin=None, zmax=None, xaxis_title="x", yaxis_title="y", zaxis_title="ψ",
                      show_domain_mesh=False, show_domain_mesh_boundaries=True, color_by_material=True,
                      domain_mesh_colors=None, material_names=None, domain_mesh_opacity=0.6,
                      angle=None, width=None, height=None):
    '''
    Plot a DG solution field on a 2D `SolutionDomain` as a colour map, shaded smoothly within
    each element and discontinuous across element faces (see `_dg_element_field_2d`).
z
    Rendered as a `Mesh3d` -- plotly has no native "coloured triangle mesh" trace outside its 3D
    scene types, but `Mesh3d`'s per-vertex (`intensitymode="vertex"`) colouring does exactly the
    Gouraud-shaded triangle fill a colour map needs, so it's reused here. By default (`elevate=False`)
    the mesh is flat (`z=0`) and the camera is locked to a top-down orthographic view, so it reads
    as a plain 2D colour map; the scene can still be freely rotated/zoomed like any other plotly
    3D plot. With `elevate=True`, `z` is `solution` itself (still colour-mapped the same way), giving
    an actual height-field surface -- the only way a 3D view of a 2D field is informative rather than
    a flat plane seen edge-on -- and the camera switches to a normal angled 3D view instead.

    Parameters
    ----------
    solution_domain : jax_sn.solution_domain.SolutionDomain
        Must be 2D (`solution_domain.dimension == 2`).
    solution : jnp.ndarray [n_elements, n_basis]
        One solution coefficient array (e.g. `psi` from a transport-operator solve, for one
        angle/energy-group).
    n_subdivisions : int
        Sub-triangles per element per side (see `_reference_triangle_subdivision`); `1` is exact
        for a P1 field, higher values give a smoother-looking gradient/surface for higher-degree
        fields at the cost of a bigger mesh.
    colorscale : str or list
        Plotly colourscale name (or explicit scale) for `solution`'s values.
    colorbar_title : str
        Title of the colour bar.
    show_element_boundaries : bool
        Overlay a black wireframe along element edges (following the surface when `elevate=True`).
    show_domain_mesh : bool
        Draw `solution_domain.domain`'s mesh, coloured by material exactly like `plot_domain`
        (same `_color`/qualitative-palette scheme), as a flat floor beneath the solution surface
        -- at `z=0` when `elevate=False` (partially transparent, so it still shows through/around
        the solution surface sitting right on top of it) or just below the surface's lowest point
        when `elevate=True` (so the surface visibly floats above its own geometry/material footprint).
    show_domain_mesh_boundaries : bool
        On top of the floor itself, also overlay a black wireframe along its element edges (every
        triangle's 3 edges, so a shared interior edge between two same-material elements is still
        drawn -- consistent with `show_element_boundaries` elsewhere in this module). Ignored
        unless `show_domain_mesh=True`.
    color_by_material, domain_mesh_colors, material_names : optional
        Forwarded to the floor's material colouring; see `plot_domain`'s parameters of the same
        name. Ignored unless `show_domain_mesh=True`.
    domain_mesh_opacity : float
        Opacity of the floor mesh. Ignored unless `show_domain_mesh=True`.
    angle : jnp.ndarray, optional
        If given (a direction `[n_dim]`, as passed to e.g. `plot_domain`'s `graph_angle`), draws
        a small red "Angle" compass arrow (see `_add_angle_compass`) at the domain's corner,
        sitting at the same baseline height as the floor (whether or not `show_domain_mesh` is
        actually on).
    elevate : bool
        If True, plot `solution` as height (`z`) as well as colour, and switch to an angled 3D
        camera. If False (default), `z=0` for every point and the camera is a locked top-down
        view, i.e. a flat colour map.
    z_scale : float
        Multiplies `solution` before using it as `z`, when `elevate=True`. Vertical exaggeration
        for fields whose value range is small relative to the domain's spatial extent; ignored
        when `elevate=False`.
    zmin, zmax : float, optional
        Colour scale bounds (and, when `elevate=True`, the z-axis range before `z_scale`); left
        `None` (default), plotly scales to `solution`'s min/max.
    xaxis_title, yaxis_title, zaxis_title : str
        Axis labels; `zaxis_title` only shown when `elevate=True` (the z axis is hidden otherwise).
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly autosizes
        to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    assert solution_domain.dimension == 2, f"plot_2d_solution requires a 2D solution_domain, got dimension {solution_domain.dimension}"
    solution = onp.asarray(solution)
    assert solution.shape == solution_domain.elements_basis_shape.shape, \
        f"solution must be of shape {solution_domain.elements_basis_shape.shape} but got {solution.shape}"

    x, y, values, triangles = _dg_element_field_2d(solution_domain, solution, n_subdivisions)
    z = values * z_scale if elevate else onp.zeros_like(values)

    # Baseline height for anything that sits "under" the solution surface -- the floor (if drawn)
    # and the angle compass (if drawn), whether or not the floor itself is shown -- so the compass
    # stays at a consistent place regardless of `show_domain_mesh`. Two nearby levels: `floor_z`
    # for the floor's filled triangles, `floor_wire_z` (a touch closer to the surface) for its
    # wireframe, so the wireframe doesn't z-fight with the filled floor underneath it.
    if elevate:
        z_span       = float(onp.max(z) - onp.min(z))
        z_span       = z_span if z_span > 0 else 1.0
        floor_z      = float(onp.min(z)) - 0.08 * z_span
        floor_wire_z = float(onp.min(z)) - 0.078 * z_span
    else:
        domain_xy    = onp.asarray(solution_domain.domain.vertices)
        extent       = float(onp.max(domain_xy.max(axis=0) - domain_xy.min(axis=0)))
        extent       = extent if extent > 0 else 1.0
        floor_z      = -1e-3 * extent
        floor_wire_z = -0.5e-3 * extent

    fig = go.Figure()

    if show_domain_mesh:
        domain = solution_domain.domain
        floor_vertices = _pad_to_3d(onp.asarray(domain.vertices))
        floor_vertices[:, 2] = floor_z
        connectivity = onp.asarray(domain.connectivity)
        materials    = _material_indices(domain, color_by_material)
        for color_idx, material in enumerate(onp.unique(materials)):
            faces = connectivity[materials == material]
            fig.add_trace(go.Mesh3d(
                x=floor_vertices[:, 0], y=floor_vertices[:, 1], z=floor_vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=_color(domain_mesh_colors, color_idx), opacity=domain_mesh_opacity,
                flatshading=True, name=_material_label(material, material_names),
                showlegend=color_by_material,
            ))
        if show_domain_mesh_boundaries:
            floor_wire_vertices = floor_vertices.copy()
            floor_wire_vertices[:, 2] = floor_wire_z
            edge_x, edge_y, edge_z = _triangle_edges(floor_wire_vertices, connectivity)
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode="lines",
                line=dict(width=2, color="black"), showlegend=False, hoverinfo="skip",
            ))

    if angle is not None:
        _add_angle_compass(fig, 2, solution_domain.domain, angle, z=floor_z)

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        intensity=values, intensitymode="vertex", colorscale=colorscale, cmin=zmin, cmax=zmax,
        showscale=True, colorbar=dict(title=colorbar_title), flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), name="solution",
    ))

    if show_element_boundaries:
        # a coarse (element-corner-only, i.e. n_subdivisions=1) copy of the same field, used only
        # to draw the 3 edges of each element, nudged just off the surface to avoid z-fighting.
        corner_x, corner_y, corner_values, corner_triangles = _dg_element_field_2d(solution_domain, solution, 1)
        corner_z = corner_values * z_scale if elevate else onp.zeros_like(corner_values)
        corner_z = corner_z + 1e-3 * (float(onp.max(onp.abs(corner_z))) + 1.0)
        edge_x, edge_y, edge_z = _triangle_edges(
            onp.stack([corner_x, corner_y, corner_z], axis=1), corner_triangles)
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode="lines",
            line=dict(width=2, color="black"), showlegend=False, hoverinfo="skip",
        ))

    scene = dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    if elevate:
        scene |= dict(zaxis_title=zaxis_title, aspectmode="auto")
    else:
        scene |= dict(zaxis=dict(visible=False), aspectmode="data",
                       camera=dict(eye=dict(x=0, y=0, z=1.8), up=dict(x=0, y=1, z=0),
                                   projection=dict(type="orthographic")))

    fig.update_layout(scene=scene, margin=dict(l=0, r=0, t=40, b=40), width=width, height=height)
    return fig


_DISPATCH = {1: _plot_domain_1d, 2: _plot_domain_2d, 3: _plot_domain_3d}


def plot_domain(domain, *, color_by_material=True, show_element_boundaries=True,
                 show_material_interfaces=True, colors=None, material_names=None,
                 show_graph=False, graph_angle=None, show_levels=False, level_mode="order",
                 show_element_numbers=False, graph_color="darkred", mesh_opacity=1.0,
                 width=None, height=None):
    '''
    Visualize a `Domain`'s mesh with plotly, dispatching on `domain.vertices.shape[1]`.

    - 1D: a line with a marker at each vertex.
    - 2D: every triangle drawn as a filled polygon.
    - 3D: the tetrahedral mesh's boundary surface (exterior faces only, plus internal
      material-interface faces when `show_material_interfaces` is set) drawn with `Mesh3d`.

    Parameters
    ----------
    domain : jax_sn.domain.GeometricDomain
        The domain to visualize. Needs `cross_sections_indexed` when `color_by_material=True`.
    color_by_material : bool
        Colour elements by `domain.cross_sections_indexed.indices`, one legend entry (and
        plotly trace/trace-group) per distinct material. If False, the whole domain is drawn
        in a single colour and `show_material_interfaces` has no effect.
    show_element_boundaries : bool
        Overlay element boundaries: a vertical tick at every vertex in 1D, a black wireframe
        in 2D/3D.
    show_material_interfaces : bool
        3D only: in addition to the domain's exterior boundary, also draw the internal faces
        between two differently-tagged materials, so material regions are visible inside a
        volumetric mesh rather than only the outer box. Ignored outside 3D and when
        `color_by_material` is False.
    colors : Sequence[str], optional
        Colour palette to cycle through per material/group; defaults to plotly's qualitative
        `Plotly` palette.
    material_names : Mapping[int, str] or Sequence[str], optional
        Display name per material index, used as the legend label instead of
        `f"material {{index}}"` -- either a `{index: name}` mapping (sparse; unmapped indices
        still fall back to `f"material {{index}}"`) or a sequence indexed positionally by
        material index.
    show_graph : bool
        Overlay the element-adjacency graph: a `graph_color` circle at every element centroid,
        connected to its neighbours.
    graph_angle : jnp.ndarray, optional
        If given (a direction `[n_dim]`), `show_graph`'s edges are the *directed* sweep-
        dependency graph for that angle (upstream -> downstream, from
        `jax_sn.domain.graph_generation.compute_face_inflow`), with an arrowhead at each edge's
        midpoint and a small "Angle" legend entry -- a compass arrow just outside the domain
        showing the angle itself, separate from the per-edge arrowheads. If None (default),
        edges are the plain undirected `etoe` adjacency and no compass arrow is drawn. Also
        required (non-None) when `show_levels=True`, since levels are angle-dependent. Ignored
        when both `show_graph` and `show_levels` are False.
    show_levels : bool
        Label every element with a number from `level_mode`. Requires `graph_angle` to be given
        (both modes are angle-dependent).
    level_mode : {"levels", "order"}
        What `show_levels` labels each element with. `"levels"`: its sweep *level* -- the
        topological-sort group from
        `jax_sn.domain.graph_generation.compute_levels_domain_safe` for `graph_angle` (elements
        in the same level are mutually independent and solvable in parallel, so this repeats
        across a level). `"order"`: its own position in the topological sort (0 = computed
        first, ties within a level broken by element index) -- the actual sequential order a
        serial sweep computes elements in, matching how `BasicSweepData.levels_argsort` is built
        for a single angle (see `_element_topological_order`).
    show_element_numbers : bool
        Label every element with its own index (`0..n_elements-1`, matching row order in
        `domain.connectivity`), in the same white/`graph_color`-outlined circle style as
        `show_levels` -- but angle-independent, so it works with no `graph_angle` and needs no
        sweep computation. Independent of `show_graph`/`show_levels`; if more than one of these
        labelling options is set, `show_element_numbers` wins.
    graph_color : str
        Colour for everything drawn by `show_graph`/`show_levels`/`show_element_numbers` (edges,
        arrowheads, node outlines/labels) -- kept a single fixed accent, distinct from the mesh's
        black element boundaries, so a dense adjacency graph doesn't visually merge into the mesh
        underneath it.
    mesh_opacity : float
        Opacity (0-1) applied to the underlying mesh (fill and/or boundary lines/wireframe) only
        -- the graph/level overlay is unaffected. Lower it (e.g. 0.25-0.4) to fade a crowded mesh
        out of the way when `show_graph`/`show_levels`/`show_element_numbers` is the point of the
        plot.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly
        autosizes to its container, which -- combined with 2D/3D's fixed-aspect axes
        (`scaleanchor="y"` / `aspectmode="data"`) -- tends to letterbox a domain whose
        bounding box isn't near-square (wide flat mesh in a wide-but-short default figure,
        leaving dead vertical margin). Set both explicitly to match the domain's aspect ratio
        and reclaim that space.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    dim = onp.asarray(domain.vertices).shape[1]
    if dim not in _DISPATCH:
        raise ValueError(f"plot_domain only supports 1D/2D/3D domains, got dimension {dim}")
    if show_levels and graph_angle is None:
        raise ValueError("show_levels=True requires graph_angle (levels are angle-dependent).")
    if level_mode not in ("levels", "order"):
        raise ValueError(f'level_mode must be "levels" or "order", got {level_mode!r}')

    kwargs = dict(color_by_material=color_by_material, level_mode=level_mode,
                  show_element_boundaries=show_element_boundaries, colors=colors,
                  material_names=material_names, show_graph=show_graph, graph_angle=graph_angle,
                  show_levels=show_levels, show_element_numbers=show_element_numbers,
                  graph_color=graph_color, mesh_opacity=mesh_opacity)
    if dim == 3:
        kwargs["show_material_interfaces"] = show_material_interfaces
    fig = _DISPATCH[dim](domain, **kwargs)
    fig.update_layout(width=width, height=height)
    return fig


def _element_basis_values(element, local_coordinates):
    '''`local_coordinates` [n_points, dim] -> basis function values `[n_points, n_basis]`.'''
    return onp.asarray(element.tabulate_basis_functions_first_derivative(onp.asarray(local_coordinates))[0])


def _basis_function_slider_steps(n_basis, n_extra_always_visible=0):
    '''One slider step per basis function, toggling `visible` on its (curve/surface, nodes) trace
    pair -- traces are added in two same-length blocks (all curves/surfaces, then all node marker
    sets), so basis function `i` owns trace `i` and trace `n_basis + i`. Any further traces (e.g.
    the 1D zero-line/guide decoration, appended after both blocks) are covered by
    `n_extra_always_visible` and stay visible in every step.'''
    steps = []
    for i in range(n_basis):
        visible = [False] * (2 * n_basis) + [True] * n_extra_always_visible
        visible[i], visible[n_basis + i] = True, True
        steps.append(dict(method="update", args=[{"visible": visible}], label=str(i)))
    return steps


def _plot_element_basis_functions_1d(element, nodes, n_points, mode, colors, node_color, active_node_color):
    n_basis = element.n_basis
    x           = onp.linspace(0.0, 1.0, n_points)
    values      = _element_basis_values(element, x[:, None])
    node_values = _element_basis_values(element, nodes)

    y_pad   = 0.1 * max(float(values.max() - values.min()), 1e-9)
    y_range = [float(values.min()) - y_pad, float(values.max()) + y_pad]

    fig = go.Figure()
    if mode == "all":
        for i in range(n_basis):
            fig.add_trace(go.Scatter(
                x=x, y=values[:, i], mode="lines", line=dict(width=3, color=_color(colors, i)),
                name=f"H_{i}",
            ))
            fig.add_trace(go.Scatter(
                x=[nodes[i, 0]], y=[node_values[i, i]], mode="markers",
                marker=dict(size=9, color=_color(colors, i)), showlegend=False, hoverinfo="skip",
            ))
    else:
        for i in range(n_basis):
            fig.add_trace(go.Scatter(
                x=x, y=values[:, i], mode="lines", line=dict(width=3, color=_color(colors, i)),
                name=f"H_{i}", showlegend=False, visible=(i == 0),
            ))
        for i in range(n_basis):
            marker_colors = [active_node_color if j == i else node_color for j in range(n_basis)]
            sizes         = [12 if j == i else 7 for j in range(n_basis)]
            fig.add_trace(go.Scatter(
                x=nodes[:, 0], y=node_values[:, i], mode="markers",
                marker=dict(size=sizes, color=marker_colors), showlegend=False, hoverinfo="skip",
                visible=(i == 0),
            ))

    # Reference guide, always visible regardless of mode/slider step: a dashed vertical line from
    # the zero line up to 1.0 at every node, with a black marker on the zero line itself -- makes
    # the Lagrangian "1 at own node, 0 elsewhere" property visually obvious (each curve's peak lands
    # exactly on its own guide line's top).
    guide_x, guide_y = [], []
    for node_x in nodes[:, 0]:
        guide_x += [float(node_x), float(node_x), None]
        guide_y += [0.0, 1.0, None]
    fig.add_trace(go.Scatter(
        x=guide_x, y=guide_y, mode="lines", line=dict(width=1, color="black", dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=nodes[:, 0], y=onp.zeros(n_basis), mode="markers",
        marker=dict(size=7, color="black"), showlegend=False, hoverinfo="skip",
    ))

    if mode == "slider":
        fig.update_layout(sliders=[dict(active=0, currentvalue=dict(prefix="Basis function i = "),
                                         steps=_basis_function_slider_steps(n_basis, n_extra_always_visible=2))])

    fig.update_layout(xaxis_title="x (local coordinate)", yaxis_title="H_i(x)", yaxis=dict(range=y_range),
                       margin=dict(l=0, r=0, t=40, b=60))
    return fig


def _plot_element_basis_functions_2d(element, nodes, n_subdivisions, mode, colors, opacity, colorscale,
                                      colorbar_title, node_color, active_node_color, show_vertical_guides):
    n_basis = element.n_basis
    local_coordinates, triangles = _reference_triangle_subdivision(n_subdivisions)
    values      = _element_basis_values(element, local_coordinates)
    node_values = _element_basis_values(element, nodes)
    x, y        = local_coordinates[:, 0], local_coordinates[:, 1]

    fig = go.Figure()
    if mode == "all":
        for i in range(n_basis):
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=values[:, i], i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                color=_color(colors, i), opacity=opacity, flatshading=True, name=f"H_{i}",
                showlegend=True,
            ))
            fig.add_trace(go.Scatter3d(
                x=[nodes[i, 0]], y=[nodes[i, 1]], z=[node_values[i, i]], mode="markers",
                marker=dict(size=6, color=_color(colors, i)), showlegend=False, hoverinfo="skip",
            ))
    else:
        cmin, cmax = float(values.min()), float(values.max())
        for i in range(n_basis):
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=values[:, i], i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                intensity=values[:, i], intensitymode="vertex", colorscale=colorscale, cmin=cmin, cmax=cmax,
                showscale=True, colorbar=dict(title=colorbar_title), flatshading=False,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), name=f"H_{i}", visible=(i == 0),
            ))
        for i in range(n_basis):
            marker_colors = [active_node_color if j == i else node_color for j in range(n_basis)]
            sizes         = [8 if j == i else 5 for j in range(n_basis)]
            fig.add_trace(go.Scatter3d(
                x=nodes[:, 0], y=nodes[:, 1], z=node_values[:, i], mode="markers",
                marker=dict(size=sizes, color=marker_colors), showlegend=False, hoverinfo="skip",
                visible=(i == 0),
            ))

    # Reference-cell outline + black node dots on the z=0 plane, always visible -- the 2D
    # counterpart of the 1D zero-line guide, tying each node's (x, y) position back to the cell.
    corners = onp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    fig.add_trace(go.Scatter3d(
        x=corners[:, 0], y=corners[:, 1], z=onp.zeros(4), mode="lines",
        line=dict(width=3, color="black"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=nodes[:, 0], y=nodes[:, 1], z=onp.zeros(n_basis), mode="markers",
        marker=dict(size=5, color="black"), showlegend=False, hoverinfo="skip",
    ))
    n_extra = 2

    if show_vertical_guides:
        # a dashed vertical line from each node's zero-plane dot up to z=1.0, exactly where that
        # node's own basis function peaks -- makes the "1 at own node, 0 elsewhere" property
        # visually obvious the same way the 1D guide does.
        guide_x, guide_y, guide_z = [], [], []
        for nx, ny in nodes:
            guide_x += [float(nx), float(nx), None]
            guide_y += [float(ny), float(ny), None]
            guide_z += [0.0, 1.0, None]
        fig.add_trace(go.Scatter3d(
            x=guide_x, y=guide_y, z=guide_z, mode="lines",
            line=dict(width=2, color="black", dash="dash"), showlegend=False, hoverinfo="skip",
        ))
        n_extra += 1

    if mode == "slider":
        fig.update_layout(sliders=[dict(active=0, currentvalue=dict(prefix="Basis function i = "),
                                         steps=_basis_function_slider_steps(n_basis, n_extra_always_visible=n_extra))])

    fig.update_layout(
        scene=dict(xaxis_title="x (local)", yaxis_title="y (local)", zaxis_title="H_i(x, y)",
                   aspectmode="auto"),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    return fig


def plot_element_basis_functions(element, *, mode="auto", n_subdivisions=10, n_points_1d=200,
                                  colors=None, opacity=0.55, colorscale="Viridis", colorbar_title="H_i",
                                  node_color="black", active_node_color="red", show_vertical_guides=True,
                                  width=None, height=None):
    '''
    Visualize a Lagrangian element's basis functions H_i and their nodes x_i on the reference cell.

    1D: each H_i is drawn as a curve over the reference interval [0, 1]. 2D: as an elevated surface
    over the reference triangle (mirroring `plot_2d_solution`'s `elevate=True` style, reusing the
    same `_reference_triangle_subdivision`). In both cases every node is drawn as a marker at its own
    H_i value (1 for the active node, ~0 for the rest -- taken from the same tabulation as the
    curve/surface, not hard-coded).

    Parameters
    ----------
    element : jax_sn.solution_domain.basix_fem.BasixLagrangianSimplex
        A 1D or 2D Lagrangian simplex element (`element.dimension in (1, 2)`).
    mode : {"auto", "all", "slider"}
        "all": every basis function drawn simultaneously, each its own colour/legend entry with a
        marker at its own peak node -- clear in 1D, where curves rarely overlap confusingly. "slider":
        one basis function shown at a time (stepped through with a slider), with every node drawn
        (the active one highlighted in `active_node_color`) -- usually clearer in 2D, where several
        overlaid coloured/translucent elevated surfaces get visually busy fast, especially past a few
        basis functions. "auto" (default) picks "all" for a 1D element, "slider" for a 2D one.
    n_subdivisions : int
        2D only: sub-triangles per element per side used to sample the surface, see
        `_reference_triangle_subdivision`; higher values resolve the shape between nodes more
        smoothly for higher-degree elements.
    n_points_1d : int
        1D only: number of points densely sampled over [0, 1] to draw each curve.
    colors : Sequence[str], optional
        `mode="all"` only: colour palette to cycle through per basis function; defaults to plotly's
        qualitative `Plotly` palette.
    opacity : float
        2D `mode="all"` only: opacity of each overlaid surface.
    colorscale : str or list
        2D `mode="slider"` only: Plotly colourscale for the basis function's value.
    colorbar_title : str
        2D `mode="slider"` only: colour bar title.
    node_color, active_node_color : str
        `mode="slider"` only: marker colour for inactive nodes and for the currently selected basis
        function's own node.
    show_vertical_guides : bool
        2D only: also draw a dashed vertical line from each node's zero-plane dot up to z=1.0 (the
        2D counterpart of the 1D guide, always drawn there). The reference cell's outline and a
        black dot per node on the z=0 plane are always drawn in 2D regardless of this flag.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly autosizes
        to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    dim = element.dimension
    if dim not in (1, 2):
        raise ValueError(f"plot_element_basis_functions only supports 1D/2D elements, got dimension {dim}")
    if mode == "auto":
        mode = "all" if dim == 1 else "slider"
    if mode not in ("all", "slider"):
        raise ValueError(f'mode must be "auto", "all", or "slider", got {mode!r}')

    nodes = onp.asarray(element.element.points)
    if dim == 1:
        fig = _plot_element_basis_functions_1d(element, nodes, n_points_1d, mode, colors, node_color,
                                                active_node_color)
    else:
        fig = _plot_element_basis_functions_2d(element, nodes, n_subdivisions, mode, colors, opacity,
                                                colorscale, colorbar_title, node_color, active_node_color,
                                                show_vertical_guides)
    fig.update_layout(width=width, height=height)
    return fig


def plot_quadrature_set(quadrature_set, *, values=None, colorbar_title=None, color_by_weight=True,
                         colorscale="Viridis", cmin=None, cmax=None, marker_size=6,
                         ray_color="lightgray", show_unit_sphere=False, width=None, height=None):
    '''
    Plot a `QuadratureSetBase`'s angles as points on the unit sphere, each joined to the origin
    by a line so it reads as a direction/ray rather than a bare point.

    Parameters
    ----------
    quadrature_set : jax_sn.quadrature_set.QuadratureSetBase
        Any quadrature set; only `.angles` ([n_angles, 3]) and (unless `values` is given)
        `.weights` ([n_angles]) are used.
    values : jnp.ndarray [n_angles], optional
        Colour each point by these values instead of the quadrature weights -- e.g. some function
        evaluated at the quadrature's angles. Overrides `color_by_weight`.
    colorbar_title : str, optional
        Colour bar title; defaults to "w" for weights, or "f" when `values` is given.
    color_by_weight : bool
        Colour each point by its quadrature weight (with a colour bar) instead of a single flat
        colour. Ignored when `values` is given (then always coloured).
    colorscale : str or list
        Plotly colourscale name (or explicit scale) for the weights/`values`; ignored when
        neither is used for colouring.
    cmin, cmax : float, optional
        Colour scale bounds; left `None` (default), plotly scales to the weights/`values`'
        own min/max. Set both explicitly (e.g. to a shared, zero-centred range) to make colours
        comparable against another plot's -- e.g. `plot_moments_spheres`/`plot_spherical_field`'s.
    marker_size : int
        Marker size in pixels.
    ray_color : str
        Colour of the origin-to-angle lines.
    show_unit_sphere : bool
        Overlay a faint unit sphere for reference.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly autosizes
        to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    angles = onp.asarray(quadrature_set.angles)

    use_values   = values is not None
    color_values = onp.asarray(values) if use_values else onp.asarray(quadrature_set.weights)
    label        = "f" if use_values else "w"
    use_color    = use_values or color_by_weight

    fig = go.Figure()

    if show_unit_sphere:
        u = onp.linspace(0, onp.pi, 30)
        v = onp.linspace(0, 2 * onp.pi, 30)
        fig.add_trace(go.Surface(
            x=onp.outer(onp.sin(u), onp.cos(v)), y=onp.outer(onp.sin(u), onp.sin(v)),
            z=onp.outer(onp.cos(u), onp.ones_like(v)), opacity=0.15, showscale=False,
            colorscale=[[0, "lightblue"], [1, "lightblue"]], hoverinfo="skip", name="unit sphere",
        ))

    ray_x, ray_y, ray_z = [], [], []
    for x, y, z in angles:
        ray_x += [0.0, x, None]
        ray_y += [0.0, y, None]
        ray_z += [0.0, z, None]
    fig.add_trace(go.Scatter3d(
        x=ray_x, y=ray_y, z=ray_z, mode="lines", line=dict(width=2, color=ray_color),
        showlegend=False, hoverinfo="skip", name="rays",
    ))

    marker = dict(size=marker_size)
    if use_color:
        marker |= dict(color=color_values, colorscale=colorscale, cmin=cmin, cmax=cmax,
                        showscale=True, colorbar=dict(title=colorbar_title or label))
    else:
        marker |= dict(color=_color(None, 0))

    fig.add_trace(go.Scatter3d(
        x=angles[:, 0], y=angles[:, 1], z=angles[:, 2], mode="markers", marker=marker,
        text=[f"{label}={v:.4g}" for v in color_values], hoverinfo="text", name="angles",
    ))

    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                       margin=dict(l=0, r=0, t=40, b=40), width=width, height=height)
    return fig


def plot_moments_bar(moments, l_max, *, colors=None, width=None, height=None):
    r'''
    Bar chart of real-spherical-harmonic moment coefficients :math:`\phi_{lm}` against the
    flattened :math:`(l,m)` index, shaded by degree :math:`l`.

    Parameters
    ----------
    moments : jnp.ndarray [(l_max + 1)**2]
        Moment coefficients in the same :math:`(l,m)` ordering as
        `jax_sn.quadrature_set.QuadratureSetBase.wk_ylm_real`/`ylm_real`
        (see `jax_sn.quadrature_set.quadrature_set_base._lm_real_expansion`), e.g. from
        contracting a discrete field against `QuadratureSetBase.wk_ylm_real(l_max)`.
    l_max : int
        Maximum expansion degree `moments` was computed with.
    colors : Sequence[str], optional
        Colour palette to cycle through per degree l; defaults to plotly's qualitative `Plotly`
        palette.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly autosizes
        to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
    l, m    = (onp.asarray(x) for x in _lm_real_expansion(l_max))
    moments = onp.asarray(moments)
    assert moments.shape[0] == l.shape[0], \
        f"moments must have shape [{l.shape[0]}] for l_max={l_max}, got {moments.shape}"

    fig = go.Figure()
    for degree in range(l_max + 1):
        mask = l == degree
        fig.add_trace(go.Bar(
            x=[f"({degree},{mi})" for mi in m[mask]], y=moments[mask],
            marker_color=_color(colors, degree), name=f"l={degree}",
        ))
    fig.update_layout(xaxis_title="(l, m)", yaxis_title="φ_lm", bargap=0.15, showlegend=False,
                       margin=dict(l=0, r=0, t=40, b=40), width=width, height=height)
    return fig


def plot_spherical_field(l_max, moments, *, n_theta=60, n_phi=120, colorscale="Viridis",
                          colorbar_title="Σ φ_lm·Y_lm(Ω)", zmin=None, zmax=None, width=None,
                          height=None):
    '''
    Plot a real-spherical-harmonic expansion as a continuous field on the unit sphere, sampled on
    a fine (theta, phi) grid independent of any quadrature set -- the continuous, "re-interpolated"
    counterpart to `plot_quadrature_set`'s discrete points.

    Parameters
    ----------
    l_max : int
        Maximum expansion degree; `moments` must have `(l_max + 1)**2` entries.
    moments : jnp.ndarray [(l_max + 1)**2]
        Real-spherical-harmonic expansion coefficients, in the same ordering as
        `QuadratureSetBase.wk_ylm_real`/`ylm_real` (see `_lm_real_expansion`).
    n_theta, n_phi : int
        Grid resolution in the polar/azimuthal angle.
    colorscale : str or list
        Plotly colourscale for the reconstructed field.
    colorbar_title : str
        Colour bar title.
    zmin, zmax : float, optional
        Colour scale bounds; left `None` (default), plotly scales to the field's min/max.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly autosizes
        to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion, _ylm_real
    moments = jnp.asarray(moments)
    l, m    = _lm_real_expansion(l_max)
    assert moments.shape[0] == l.shape[0], \
        f"moments must have shape [{l.shape[0]}] for l_max={l_max}, got {moments.shape}"

    theta = jnp.linspace(0.0, jnp.pi, n_theta)
    phi   = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")

    yklm  = _ylm_real(l, m, theta_grid.reshape(-1), phi_grid.reshape(-1), l_max)
    field = onp.asarray((yklm @ moments).reshape(n_theta, n_phi))

    theta_grid, phi_grid = onp.asarray(theta_grid), onp.asarray(phi_grid)
    x = onp.sin(theta_grid) * onp.cos(phi_grid)
    y = onp.sin(theta_grid) * onp.sin(phi_grid)
    z = onp.cos(theta_grid)

    fig = go.Figure(go.Surface(
        x=x, y=y, z=z, surfacecolor=field, colorscale=colorscale, cmin=zmin, cmax=zmax,
        showscale=True, colorbar=dict(title=colorbar_title),
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    ))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                       margin=dict(l=0, r=0, t=40, b=40), width=width, height=height)
    return fig


def plot_moments_spheres(moments, l_max, *, n_theta=16, n_phi=32, colorscale="RdBu_r", clim=None,
                          radius=0.85, spacing=2.2, colorbar_title="φ_lm·Y_lm(Ω)", width=None,
                          height=None):
    r'''
    Visualize real-spherical-harmonic moments :math:`\phi_{lm}` as a pyramid of small spheres --
    row :math:`l`, column :math:`m` -- each coloured by that term's own contribution
    :math:`\phi_{lm} Y_{lm}(\Omega)` on one shared, zero-centred colour scale. That shared scale is
    what makes both shape *and* relative size legible at a glance: a near-zero moment reads as a
    flat, faint sphere, a large one shows its harmonic's lobed pattern vividly -- the small-multiples
    analogue of `plot_spherical_field`'s single reconstructed field, one term at a time instead of
    their sum.

    Each little sphere is still oriented in the same (x, y, z) convention as the discrete angles
    themselves (only translated to its grid slot), so the scene keeps (unticked, since the axis
    *values* are grid positions, not coordinates) x/y/z titles -- letting the reader relate a lobe's
    direction here back to the same direction in `plot_quadrature_set`/`plot_spherical_field`.

    Parameters
    ----------
    moments : jnp.ndarray [(l_max + 1)**2]
        Moment coefficients, in the same ordering as `plot_moments_bar`/`_lm_real_expansion`.
    l_max : int
        Maximum expansion degree.
    n_theta, n_phi : int
        Per-sphere grid resolution (kept low by default -- there are `(l_max + 1)**2` of these).
    colorscale : str or list
        Diverging Plotly colourscale; "RdBu" (default) makes positive/negative lobes easy to tell
        apart.
    clim : float, optional
        Symmetric colour scale bound (`cmin=-clim, cmax=clim`); left `None` (default), it's the
        largest `|φ_lm·Y_lm(Ω)|` over every term. Set explicitly to a shared value (e.g. matching
        `plot_quadrature_set`/`plot_spherical_field`'s own `cmin`/`cmax`) to make colours comparable
        across plots.
    radius : float
        Radius of each small sphere; less than half of `spacing` leaves a visible gap between
        neighbours.
    spacing : float
        Centre-to-centre distance between neighbouring (l, m) spheres.
    colorbar_title : str
        Colour bar title.
    width, height : int, optional
        Figure size in pixels; either/both may be given. Left `None` (default) plotly autosizes
        to its container.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion, _ylm_real
    moments = jnp.asarray(moments)
    l, m    = _lm_real_expansion(l_max)
    assert moments.shape[0] == l.shape[0], \
        f"moments must have shape [{l.shape[0]}] for l_max={l_max}, got {moments.shape}"

    theta = jnp.linspace(0.0, jnp.pi, n_theta)
    phi   = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")

    yklm  = _ylm_real(l, m, theta_grid.reshape(-1), phi_grid.reshape(-1), l_max)
    terms = onp.asarray(yklm * moments[None, :]).reshape(n_theta, n_phi, -1)  # each (l,m)'s own term

    if clim is None:
        clim = float(onp.max(onp.abs(terms)))
        clim = clim if clim > 0 else 1.0

    theta_grid, phi_grid = onp.asarray(theta_grid), onp.asarray(phi_grid)
    x_local = radius * onp.sin(theta_grid) * onp.cos(phi_grid)
    y_local = radius * onp.sin(theta_grid) * onp.sin(phi_grid)
    z_local = radius * onp.cos(theta_grid)

    l_np, m_np = onp.asarray(l), onp.asarray(m)

    fig = go.Figure()
    for idx, (li, mi) in enumerate(zip(l_np, m_np)):
        fig.add_trace(go.Surface(
            x=x_local + spacing * mi, y=y_local - spacing * li, z=z_local,
            surfacecolor=terms[..., idx], cmin=-clim, cmax=clim, colorscale=colorscale,
            showscale=(idx == 0), colorbar=dict(title=colorbar_title),
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), hoverinfo="skip",
        ))

    # one (l, m) tag per sphere -- m only, since the new row label below already carries l
    fig.add_trace(go.Scatter3d(
        x=spacing * m_np, y=-spacing * l_np - 1.3 * radius, z=onp.zeros_like(l_np, dtype=float),
        mode="text", text=[f"m={mi:+d}" for mi in m_np], textfont=dict(size=10),
        showlegend=False, hoverinfo="skip",
    ))

    # one "l=..." row label to the left of each row, past its leftmost (m = -l) sphere
    row_l = onp.arange(l_max + 1)
    fig.add_trace(go.Scatter3d(
        x=-spacing * (row_l + 1.4), y=-spacing * row_l, z=onp.zeros_like(row_l, dtype=float),
        mode="text", text=[f"l={li}" for li in row_l], textfont=dict(size=14, color="black"),
        showlegend=False, hoverinfo="skip",
    ))

    axis = dict(visible=True, showticklabels=False, showgrid=False, zeroline=False,
                showbackground=False)
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=dict(axis, title="x"), yaxis=dict(axis, title="y"),
                   zaxis=dict(axis, title="z"),
                   camera=dict(eye=dict(x=0, y=0, z=3.2), up=dict(x=0, y=1, z=0),
                               projection=dict(type="orthographic"))),
        showlegend=False, margin=dict(l=0, r=0, t=40, b=0), width=width, height=height)
    return fig


def add_axis_triad(fig, origin=(0.0, 0.0, 0.0), length=1.0, colors=("red", "green", "blue"),
                    labels=("x", "y", "z"), row=None, col=None):
    '''
    Draw a small PyVista/VTK-style XYZ orientation triad in a 3D scene -- three colour-coded arrows
    from `origin` along the local +x/+y/+z directions, each tipped with a cone and text label.
    Meant as a direct replacement for Plotly's own scene axes when those are hidden (e.g. because
    the axis *values* aren't physically meaningful, as in `plot_moments_spheres`'s grid layout, or
    just to keep several scenes visually consistent) -- a literal, always-legible direction marker
    instead of relying on axis lines/ticks that are easy to hide by accident and don't stand out
    much even when shown.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Modified in place (new traces appended).
    origin : Sequence[float] [3]
        Where the triad's three arrows start.
    length : float
        Arrow length (shaft + cone).
    colors : Sequence[str] [3]
        Colour per axis, in (x, y, z) order; default matches the common red/green/blue convention.
    labels : Sequence[str] [3]
        Text label per axis, in (x, y, z) order.
    row, col : int, optional
        Subplot cell to add the triad to (as for `Figure.add_trace`); omit for a non-subplot figure.
    '''
    origin = onp.asarray(origin, dtype=float)
    kwargs = dict(row=row, col=col) if row is not None else {}
    for direction, color, label in zip(onp.eye(3), colors, labels):
        tip = origin + direction * length
        fig.add_trace(go.Scatter3d(
            x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]], mode="lines",
            line=dict(width=6, color=color), showlegend=False, hoverinfo="skip",
        ), **kwargs)
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[direction[0]], v=[direction[1]], w=[direction[2]],
            sizemode="absolute", sizeref=0.3 * length, anchor="tail", showscale=False,
            colorscale=[[0, color], [1, color]], showlegend=False, hoverinfo="skip",
        ), **kwargs)
        label_pos = origin + direction * (length * 1.3)
        fig.add_trace(go.Scatter3d(
            x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]], mode="text", text=[label],
            textfont=dict(size=14, color=color), showlegend=False, hoverinfo="skip",
        ), **kwargs)


# ==================================================================================================
#                                       Derivation graphs
# ==================================================================================================
# A `jax_sn.tools.parametrisation.ParametrisedModule` subclass *is* a graph -- a derivation's parameter names
# are its dependencies -- and its storage policy is a colouring of that graph. Both come straight off
# `cls.graph()`; nothing here inspects the class any further.

_DERIVATION_ROLE_COLORS = {
    "free primitive"     : "#2ca02c",
    "retained primitive" : "#bcbd22",
    "frozen primitive"   : "#8c8c8c",
    "stored"             : "#1f77b4",
    "projected"          : "#9467bd",
    "derived"            : "#ff7f0e",
}

_DERIVATION_ROLE_DESCRIPTIONS = {
    "free primitive"     : "primitive, may be replaced by .updated()",
    "retained primitive" : "primitive, replaceable while every retention on it holds",
    "frozen primitive"   : "primitive, pinned: a stored quantity was built from it",
    "stored"             : "held as a field, read directly",
    "projected"          : "read back out of a stored quantity",
    "derived"            : "recomputed on every access",
}

_DERIVATION_PALE = "#e3e3e3"        # a node/edge greyed out by a `highlight` selection


def _derivation_roles(graph):
    '''Each node's role under this class's storage policy; the keys of `_DERIVATION_ROLE_COLORS`.'''
    roles = {n: ("retained primitive" if n in graph.conditionally_free else
                 "frozen primitive"   if n in graph.frozen else "free primitive")
             for n in graph.primitives}
    roles |= {n: ("stored"    if n in graph.stored    else
                  "projected" if n in graph.projected else "derived") for n in graph.quantities}
    return roles


def _derivation_layers(deps, nodes):
    '''
    `nodes` split into layers by longest path from a primitive, so every dependency of a node sits
    in a strictly earlier layer (primitives, having none, are layer 0). Longest rather than shortest
    path: a shortest-path layering would place a node next to its *nearest* dependency and leave the
    others reaching forward past it.
    '''
    depth = {}

    def resolve(name):
        if name in depth:
            return depth[name]
        depth[name] = 0             # guards the recursion; the registry is already known acyclic
        depth[name] = 1 + max((resolve(p) for p in deps.get(name, ())), default=-1)
        return depth[name]

    for name in nodes:
        resolve(name)
    layers = [[] for _ in range(max(depth.values()) + 1)]
    for name in sorted(nodes):      # alphabetical, so the ordering sweep starts from something fixed
        layers[depth[name]].append(name)
    return layers, depth


def _route_long_edges(layers, edges, depth):
    '''
    Make the layering *proper*: an edge spanning several layers gets a dummy node in each layer it
    crosses, and is drawn as a polyline through them.

    Without this, a long edge (``element`` feeds both ``fem_matrices`` one layer along and
    ``precomputed_geometric_data`` four layers along) is a single straight line across the whole
    drawing -- it crosses nearly everything, it passes *through* unrelated node markers so that two
    nodes it merely flies past look connected, and the ordering heuristic below cannot see it at all,
    since it only ever compares adjacent layers. Routed, the edge occupies a real slot in every layer
    it crosses, so it is ordered and separated like any other node.

    `layers` is extended in place. Returns the routes, plus the adjacency of the proper graph.
    '''
    routes = {}
    for src, dst in edges:
        route = [("dummy", src, dst, d) for d in range(depth[src] + 1, depth[dst])]
        for node in route:
            layers[node[3]].append(node)
        routes[(src, dst)] = route

    predecessors = {n: [] for layer in layers for n in layer}
    successors   = {n: [] for layer in layers for n in layer}
    for (src, dst), route in routes.items():
        chain = [src, *route, dst]
        for a, b in zip(chain, chain[1:]):
            predecessors[b].append(a)
            successors[a].append(b)
    return routes, predecessors, successors


def _count_crossings(layers, predecessors, order):
    '''
    Edge crossings between every pair of adjacent layers: two edges into the same layer cross when
    their endpoints are in opposite order. Pairwise, which is quadratic in the edge count -- fine for
    a registry of this size, and it lets the sweeps below keep the best ordering they ever saw rather
    than whatever the last sweep happened to leave behind.
    '''
    total = 0
    for layer in layers:
        pairs = [(order[p], order[n]) for n in layer for p in predecessors[n]]
        total += sum((a[0] - b[0]) * (a[1] - b[1]) < 0
                     for i, a in enumerate(pairs) for b in pairs[i + 1:])
    return total


def _ordering_sweeps(layers, predecessors, successors, statistic, n_sweeps=16):
    '''
    Reorder within each layer to cut crossings, by the classic heuristic: repeatedly place a node at
    the barycentre (or median) of its neighbours in the layer just ordered, alternating direction.
    Purely cosmetic; the layering is what carries the meaning.
    '''
    layers = [list(layer) for layer in layers]
    order  = {n: i for layer in layers for i, n in enumerate(layer)}
    best, best_crossings = [list(l) for l in layers], _count_crossings(layers, predecessors, order)

    for sweep in range(n_sweeps):
        neighbours = predecessors if sweep % 2 == 0 else successors
        for layer in (layers if sweep % 2 == 0 else layers[::-1]):
            layer.sort(key=lambda n: statistic([order[m] for m in neighbours[n]] or [order[n]]))
            order.update({n: i for i, n in enumerate(layer)})       # the next layer reads this one
        crossings = _count_crossings(layers, predecessors, order)
        if crossings < best_crossings:
            best, best_crossings = [list(l) for l in layers], crossings
    return best, best_crossings


def _isotonic(desired, weights, offsets):
    '''
    The closest sequence to `desired` (weighted least squares) that still respects the layer's node
    order and keeps every neighbouring pair at least `offsets`' spacing apart. Subtracting the
    (cumulative) offsets turns "separated and increasing" into plain "non-decreasing", which is
    textbook isotonic regression, solved here by pool-adjacent-violators.
    '''
    blocks = []          # [weighted sum, weight, count], each already >= the block before it
    for value, weight, offset in zip(desired, weights, offsets):
        blocks.append([(value - offset) * weight, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] > blocks[-1][0] * blocks[-2][1]:
            top, below = blocks.pop(), blocks.pop()
            blocks.append([below[0] + top[0], below[1] + top[1], below[2] + top[2]])
    flat = [total / weight for total, weight, count in blocks for _ in range(count)]
    return [value + offset for value, offset in zip(flat, offsets)]


def _layer_offsets(layer, dummy_half, node_claim):
    '''
    Where each node in a layer sits if the layer is packed as tightly as it may be. A real node
    claims `node_claim` on each side -- half a unit in total, so two of them end up the usual 1
    apart, but split unevenly to leave room on the side its label is drawn -- while a dummy node is
    just a line passing through and claims `dummy_half`, so routing a long edge costs a sliver of
    space rather than a whole row.
    '''
    claims  = [(dummy_half, dummy_half) if isinstance(n, tuple) else node_claim for n in layer]
    offsets = [0.0]
    for near, far in zip(claims, claims[1:]):
        offsets.append(offsets[-1] + near[1] + far[0])
    return offsets


def _straighten(layers, predecessors, successors, *, node_claim=(0.5, 0.5), n_iterations=12,
                 dummy_half=0.14, dummy_weight=6.0):
    '''
    Coordinates within a layer, given the ordering: pull every node toward the mean of its
    neighbours, then push the layer back apart just enough to restore the order and spacing
    (`_isotonic`). Dummy nodes pull much harder, which is what straightens a routed long edge into a
    line instead of a staircase.
    '''
    offsets    = [_layer_offsets(layer, dummy_half, node_claim) for layer in layers]
    coordinate = {n: offset for layer, layer_offsets in zip(layers, offsets)
                  for n, offset in zip(layer, layer_offsets)}
    for iteration in range(n_iterations):
        neighbours = predecessors if iteration % 2 == 0 else successors
        order      = range(len(layers)) if iteration % 2 == 0 else range(len(layers) - 1, -1, -1)
        for index in order:
            layer   = layers[index]
            desired = [float(onp.mean([coordinate[m] for m in neighbours[n]] or [coordinate[n]]))
                       for n in layer]
            weights = [dummy_weight if isinstance(n, tuple) else 1.0 for n in layer]
            coordinate.update(zip(layer, _isotonic(desired, weights, offsets[index])))
    return coordinate


def _derivation_positions(layers, coordinate, orientation, layer_gap):
    '''Node -> (x, y). Layers advance along x (left to right) or down y.'''
    high = max(coordinate.values())
    return {n: ((depth * layer_gap, high - coordinate[n]) if orientation == "horizontal"
                else (coordinate[n], -depth * layer_gap))
            for depth, layer in enumerate(layers) for n in layer}


def _wrap_derivation_label(name, max_chars):
    '''Break a `snake_case` quantity name across lines at underscores, keeping lines short enough
    that adjacent nodes' labels don't run into each other.'''
    lines, current = [], ""
    for word in name.split("_"):
        candidate = f"{current}_{word}" if current else word
        if current and len(candidate) > max_chars:
            lines.append(current + "_")
            current = word
        else:
            current = candidate
    lines.append(current)
    return "<br>".join(lines)


def _derivation_hover(name, cls, graph, roles):
    role = roles[name]
    lines = [f"<b>{name}</b>", f"<i>{role}</i> -- {_DERIVATION_ROLE_DESCRIPTIONS[role]}"]
    if name in graph.quantities:
        lines.append(f"depends on: {', '.join(sorted(graph.deps[name])) or '(nothing)'}")
        lines.append(f"follows from primitives: {', '.join(sorted(graph.primitive_deps[name]))}")
        if name in graph.projections:
            lines.append(f"projected out of: {graph.projections[name][0]}")
        if name in cls.NON_DIFFERENTIABLE:
            lines.append("non-differentiable (stop_gradient)")
        if name in cls.RETAINED_UNDER:
            lines.append(f"retained under: {', '.join(sorted(cls.RETAINED_UNDER[name]))}")
    elif role == "retained primitive":
        gating = sorted(q for q, sources in cls.RETAINED_UNDER.items() if name in sources)
        lines.append(f"kept across .updated() by the retention on: {', '.join(gating)}")
    used_by = sorted(q for q in graph.quantities if name in graph.deps[q])
    lines.append(f"used by: {', '.join(used_by) or '(nothing)'}")
    return "<br>".join(lines)


def _derivation_stale(graph, primitive, retentions):
    '''What changing `primitive` invalidates: itself, plus every quantity it feeds into -- except
    where a retention on a held value cuts the chain (the static form: a retention is declared, not
    that it holds for a particular new value).'''
    from jax_sn.tools.parametrisation import _insensitive
    held = graph.stored | graph.projected
    cut  = lambda n : n in held and primitive in retentions.get(n, ())
    return {primitive} | {q for q in graph.quantities
                          if not _insensitive(graph, q, primitive, cut, {})}


def _derivation_node_colors(names, roles, colors, stale=None):
    '''Marker colour per node: its role's colour, or pale grey if a selection excludes it.'''
    return [colors[roles[n]] if (stale is None or n in stale) else _DERIVATION_PALE for n in names]


def _bowed(start, end, bow):
    '''A backward edge drawn straight would cut across the layers it skips over; bowing it out
    sideways (one control point off the midpoint, rounded by the spline) keeps it clear of them.'''
    start, end = onp.asarray(start, dtype=float), onp.asarray(end, dtype=float)
    delta      = end - start
    normal     = onp.array([-delta[1], delta[0]])
    norm       = onp.linalg.norm(normal)
    if norm == 0:
        return [tuple(start), tuple(end)]
    return [tuple(start), tuple((start + end) / 2 + bow * normal / norm), tuple(end)]


def _smooth_path(points, samples_per_segment=14, tension=0.5):
    '''
    A Catmull-Rom curve through `points`, sampled densely.

    Rounding the corners of a routed edge is plotly's `line_shape="spline"`'s job, but a spline is
    drawn *near* its control points rather than through the straight segments between them, so an
    arrowhead placed on those segments visibly floats off the line. Sampling the curve here instead
    means the trace is a plain polyline and every point on it -- arrowhead included -- is a point the
    line actually passes through.
    '''
    points = onp.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    padded = onp.vstack([points[0], points, points[-1]])     # clamped ends: no overshoot
    pieces = []
    for i in range(len(points) - 1):
        start, end = padded[i + 1], padded[i + 2]
        start_tangent, end_tangent = tension * (end - padded[i]), tension * (padded[i + 3] - start)
        t = onp.linspace(0.0, 1.0, samples_per_segment, endpoint=(i == len(points) - 2))[:, None]
        pieces.append((2 * t**3 - 3 * t**2 + 1) * start + (t**3 - 2 * t**2 + t) * start_tangent
                      + (-2 * t**3 + 3 * t**2) * end + (t**3 - t**2) * end_tangent)
    return onp.vstack(pieces)


def _along_path(curve, distance_from_end):
    '''The point `distance_from_end` back along `curve` from its head, and the curve's direction
    there -- so an arrowhead sits *on* the line and points the way it is actually going.'''
    steps     = onp.diff(curve, axis=0)
    lengths   = onp.linalg.norm(steps, axis=1)
    remaining = distance_from_end
    for i in range(len(lengths) - 1, -1, -1):
        if remaining <= lengths[i] or i == 0:
            fraction = 1.0 - remaining / lengths[i] if lengths[i] > 0 else 0.0
            return curve[i] + onp.clip(fraction, 0.0, 1.0) * steps[i], steps[i]
        remaining -= lengths[i]
    return curve[-1], steps[-1]


def _add_derivation_edges(fig, edges, positions, routes=None, *, color, dash, width, name, arrows,
                           bow=0.0, arrow_at=0.72):
    '''One `Scatter` line trace for a whole edge set (None-separated, each edge a smoothed polyline
    through whatever dummy nodes `routes` gave it), plus a rotated-triangle arrowhead near the head
    of each edge -- the same trick `_add_graph_and_levels` uses for the sweep graph, so the two graph
    plots read the same way.'''
    if not edges:
        return
    routes = routes or {}
    paths  = [[positions[n] for n in (src, *routes.get((src, dst), ()), dst)] for src, dst in edges]
    if bow:
        paths = [path if len(path) > 2 else _bowed(path[0], path[-1], bow) for path in paths]
    curves = [_smooth_path(path) for path in paths]

    x, y = [], []
    for curve in curves:
        x += [*curve[:, 0], None]
        y += [*curve[:, 1], None]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, showlegend=False, hoverinfo="skip",
                             line=dict(width=width, color=color, dash=dash)))
    if not arrows:
        return
    # placed by distance back from the head, so a routed edge is marked where it arrives rather than
    # at the midpoint of a chain of segments.
    heads = [_along_path(curve, (1.0 - arrow_at) * float(onp.linalg.norm(
        onp.asarray(path[-1], dtype=float) - onp.asarray(path[-2], dtype=float))))
        for curve, path in zip(curves, paths)]
    tip       = onp.array([point for point, _ in heads])
    direction = onp.array([step for _, step in heads])
    fig.add_trace(go.Scatter(
        x=tip[:, 0], y=tip[:, 1], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=8, color=color, symbol="triangle-up", angle=_bearing_degrees(direction)),
    ))


def plot_derivation_graph(module, *, orientation="horizontal", highlight=None, stale_selector=False,
                           show_projections=True, fade_shielded=True, colors=None, layer_gap=None,
                           label_chars=14, font_size=10, node_size=15, width=800, height=700):
    '''
    Visualize the dependency graph a `jax_sn.tools.parametrisation.ParametrisedModule` declares -- e.g. any
    rung of `jax_sn.operator_creation.builders` -- with the class's storage policy as its colouring.

    Every node is a quantity or a primitive; every solid edge points from a dependency to what is
    built from it (a derivation's parameter names *are* its dependencies). Nodes are laid out left
    to right (or top to bottom) in layers by longest path from a primitive, so an edge never points
    backwards -- except a projection edge, which may, since a projection reads a quantity back out of
    a more-derived one that already holds it, and is bowed aside rather than drawn through the layers
    it skips. An edge spanning several layers is routed through the gaps between the layers it
    crosses instead of flying straight over them (see `_route_long_edges`), so no line passes through
    a node it has nothing to do with.

    The colouring is what the class's `graph()` says about storage: which primitives are still free
    to `.updated()`, which are pinned because something stored was built from them, and which
    quantities are stored as fields / projected out of a stored one / recomputed on every access.
    Since a stored or projected quantity is read rather than computed, the edges *into* it are inert
    here, and are faded out (`fade_shielded`) -- what is left in full colour is exactly the work a
    fresh access actually does. Calling this on an abstract rung (which stores nothing) therefore
    shows the bare structure; calling it on `Rung.varying(...)` / `.freezing(...)` shows what that
    policy bought.

    Parameters
    ----------
    module : type or ParametrisedModule
        A `ParametrisedModule` subclass or an instance of one (an instance is read as its class).
    orientation : {"horizontal", "vertical"}
        "horizontal" (default) lays the layers out left to right, "vertical" top to bottom -- easier
        to read in a notebook for a deep graph, since it scrolls rather than shrinks.
    highlight : str or Sequence[str], optional
        One or more primitives to trace forward: they and everything downstream keep their colour,
        everything else greys out -- i.e. exactly what `.updated(**{primitive: ...})` invalidates and
        rebuilds. A quantity name is rejected; only primitives can be updated.
    stale_selector : bool
        Add a dropdown doing the same thing interactively, one entry per primitive plus "all". Costs
        nothing at render time (every colouring is precomputed), so it survives being saved to a
        static HTML page, e.g. in the docs.
    show_projections : bool
        Draw the `PROJECTIONS` edges (source -> projected quantity) as dotted arrows. Only visible
        when the class actually stores something to project out of.
    fade_shielded : bool
        Fade the edges into a stored/projected quantity, which are not traversed on this class (see
        above). With False, every dependency edge is drawn identically -- the plain registry.
    colors : Mapping[str, str], optional
        Override colours per role; keys as in the legend ("free primitive", "frozen primitive",
        "stored", "projected", "derived").
    layer_gap : float, optional
        Spacing between layers, relative to the spacing between nodes within a layer. Raise it to
        pull crowded layers apart. Defaults to 1.0 horizontally and 1.8 vertically, where a label
        (drawn under its node) sits between two layers rather than beside them.
    label_chars, font_size, node_size : int
        Label wrap width (in characters, broken at underscores), label font size, and marker size.
        Each label sits under its node in a white, black-bordered box drawn above every trace, so an
        edge passing behind it hides rather than garbling the text.
    width, height : int, optional
        Figure size in pixels; defaults are derived from the graph's shape (its number of layers and
        the size of its widest layer), which is usually a better starting point than plotly's
        autosize for a graph this tall or wide. The default width is capped at a notebook-friendly
        1150px, so a deep graph tightens rather than overflowing the page; raise it (and `font_size`
        with it) if the labels end up crowded.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    from jax_sn.tools.parametrisation import ParametrisedModule

    cls = module if isinstance(module, type) else type(module)
    if not issubclass(cls, ParametrisedModule):
        raise TypeError(f"plot_derivation_graph needs a ParametrisedModule (sub)class, got {cls.__name__}")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f'orientation must be "horizontal" or "vertical", got {orientation!r}')
    if layer_gap is None:
        layer_gap = 1.0 if orientation == "horizontal" else 1.8

    graph = cls.graph()
    if not graph.quantities:
        raise ValueError(f"{cls.__name__} has an empty DERIVATIONS registry: nothing to plot.")
    colors = {**_DERIVATION_ROLE_COLORS, **(colors or {})}

    nodes = set(graph.quantities) | set(graph.primitives)
    deps  = {n: set(graph.deps.get(n, ())) for n in nodes}
    edges = [(d, n) for n in sorted(nodes) for d in sorted(deps[n])]

    layers, depth = _derivation_layers(deps, nodes)
    routes, predecessors, successors = _route_long_edges(layers, edges, depth)
    layers, _ = min((_ordering_sweeps(layers, predecessors, successors, statistic)
                     for statistic in (onp.mean, onp.median)), key=lambda result: result[1])
    # a label is drawn under its node, which in a horizontal layout is *along* the layer -- so a real
    # node claims more of the layer on that side, keeping routed edges out of its text.
    coordinate = _straighten(layers, predecessors, successors,
                             node_claim=(0.36, 0.64) if orientation == "horizontal" else (0.5, 0.5))
    positions  = _derivation_positions(layers, coordinate, orientation, layer_gap)

    roles = _derivation_roles(graph)
    names = [n for layer in layers for n in layer if not isinstance(n, tuple)]

    shielded = graph.stored | graph.projected
    live     = [e for e in edges if not (fade_shielded and e[1] in shielded)]
    inert    = [e for e in edges if e not in live]

    fig = go.Figure()
    _add_derivation_edges(fig, inert, positions, routes, color=_DERIVATION_PALE, dash="dot",
                          width=1.2, name="not traversed here", arrows=False)
    _add_derivation_edges(fig, live, positions, routes, color="#9a9a9a", dash="solid", width=1.4,
                          name="dependency", arrows=True)
    if show_projections:
        # a projection points from a more-derived quantity back to one it contains, so it runs
        # against the layering and is bowed out of the way rather than drawn through it.
        _add_derivation_edges(fig, [(src, q) for q, (src, _) in graph.projections.items()], positions,
                              color=colors["projected"], dash="dot", width=1.6, name="projection",
                              arrows=True, bow=0.45 * layer_gap)

    if highlight is not None:
        highlight = [highlight] if isinstance(highlight, str) else list(highlight)
        unknown = [h for h in highlight if h not in graph.primitives]
        if unknown:
            raise ValueError(f"highlight must name primitives of {cls.__name__}; {sorted(unknown)} "
                             f"{'is' if len(unknown) == 1 else 'are'} not. "
                             f"Primitives: {sorted(graph.primitives)}.")
        stale = set().union(*(_derivation_stale(graph, h, cls.RETAINED_UNDER) for h in highlight))
    else:
        stale = None

    node_x, node_y = zip(*(positions[n] for n in names))
    fig.add_trace(go.Scatter(
        x=list(node_x), y=list(node_y), mode="markers",
        marker=dict(size=node_size, symbol="circle",
                    color=_derivation_node_colors(names, roles, colors, stale),
                    line=dict(width=[2 if n in cls.NON_DIFFERENTIABLE else 1 for n in names],
                              color=["black" if n in cls.NON_DIFFERENTIABLE else "white" for n in names])),
        hovertext=[_derivation_hover(n, cls, graph, roles) for n in names],
        hoverinfo="text", showlegend=False, name="quantities",
    ))
    node_trace = len(fig.data) - 1

    # Labels as annotations rather than trace text: only an annotation can carry a filled, bordered
    # box, and annotations are drawn above every trace -- so a label is legible even where an edge
    # runs behind it, instead of the text and the line competing for the same pixels.
    labels = [dict(
        x=positions[n][0], y=positions[n][1], text=_wrap_derivation_label(n, label_chars),
        xanchor="center", yanchor="top", yshift=-(node_size / 2 + 3), showarrow=False,
        font=dict(size=font_size, color="black"), align="center",
        bgcolor="white", bordercolor="black", borderwidth=1, borderpad=3,
    ) for n in names]

    for role in _DERIVATION_ROLE_COLORS:            # legend only: dummy traces carrying no points
        if any(roles[n] == role for n in names):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=role, hoverinfo="skip",
                marker=dict(size=node_size, color=colors[role]),
            ))

    if stale_selector:
        buttons = [dict(label="all", method="restyle",
                        args=[{"marker.color": [_derivation_node_colors(names, roles, colors)]},
                              [node_trace]])]
        for primitive in sorted(graph.primitives):
            pinned = (" (retained)" if primitive in graph.conditionally_free else " (frozen)") \
                     if primitive in graph.frozen else ""
            buttons.append(dict(
                label=f"{primitive}{pinned}", method="restyle",
                args=[{"marker.color": [_derivation_node_colors(
                    names, roles, colors, _derivation_stale(graph, primitive, cls.RETAINED_UNDER))]}, [node_trace]],
            ))
        fig.update_layout(updatemenus=[dict(
            buttons=buttons, active=0, direction="down", showactive=True,
            x=0.0, xanchor="left", y=1.12, yanchor="top", pad=dict(t=0, b=4),
        )])
        labels.append(dict(
            text="stale after updating:", x=0.0, xanchor="left", y=1.19, yanchor="bottom",
            xref="paper", yref="paper", showarrow=False, font=dict(size=11),
        ))

    # Explicit ranges rather than autorange: a label is drawn outside its marker's extent but does
    # not enlarge the axis, so the outermost row/column would be clipped. Extra room below every
    # node for its (up to a few lines of) label.
    # `span` from the coordinates, not the node count: a layer's dummies are packed much tighter
    # than its real nodes, so counting slots would badly overstate how much room a layer needs.
    n_layers      = len(layers)
    span          = max(coordinate.values()) - min(coordinate.values()) + 1.0
    along, across = (layer_gap, 1.0) if orientation == "horizontal" else (1.0, layer_gap)
    all_x, all_y   = zip(*positions.values())       # routed edges can bulge past the real nodes
    fig.update_xaxes(visible=False, range=[min(all_x) - 0.7 * along, max(all_x) + 0.7 * along])
    fig.update_yaxes(visible=False, range=[min(all_y) - 0.85 * across, max(all_y) + 0.4 * across])
    fig.update_layout(
        annotations=labels,
        width=width or min(1150, max(760, 190 * (n_layers if orientation == "horizontal" else span))),
        height=height or (max(420, 86 * span) if orientation == "horizontal"
                          else max(420, 86 * n_layers)),
        margin=dict(l=20, r=20, t=90 if stale_selector else 60, b=20),
        legend=dict(orientation="h", y=-0.02, x=0.0),
        plot_bgcolor="white",
    )
    return fig

