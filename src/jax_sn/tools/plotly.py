'''
Plotly functions for visualizing domains. 

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
                           level_mode):
    '''
    Overlay the element-adjacency graph and/or per-element sweep levels/order, all in
    `graph_color` so the overlay reads as one layer distinct from the (black-outlined) mesh
    underneath.

    Nodes: a plain dot if `show_levels` is False, otherwise a white/`graph_color`-outlined circle
    with a number inside it -- `show_levels` requires `graph_angle` (both are angle-dependent).
    `level_mode` picks what that number is: `"levels"` is the sweep level (topological-sort
    *group* -- elements in the same level are mutually independent and solved in parallel,
    from `compute_levels_domain_safe`), `"order"` is each element's own position in the
    topological sort (0 = computed first, see `_element_topological_order`) -- the actual
    per-element sequential order a serial sweep processes them in.

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

    if show_levels:
        values = (_element_levels(domain, graph_angle) if level_mode == "levels"
                  else _element_topological_order(domain, graph_angle))
        fig.add_trace(_trace(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], mode="markers+text",
            marker=dict(size=20, color="white", line=dict(color=graph_color, width=1.5), symbol="circle"),
            text=[str(v) for v in values], textposition="middle center",
            textfont=dict(size=10, color=graph_color), name=level_mode,
        ))
    elif show_graph:
        fig.add_trace(_trace(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], mode="markers",
            marker=dict(size=5, color=graph_color, symbol="circle"), name="graph nodes",
        ))


def _plot_domain_1d(domain, *, color_by_material, show_element_boundaries, colors, material_names,
                     show_graph, graph_angle, show_levels, graph_color, mesh_opacity, level_mode):
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
    if show_graph or show_levels:
        _add_graph_and_levels(fig, 1, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels, graph_color=graph_color, level_mode=level_mode)
    return fig


def _plot_domain_2d(domain, *, color_by_material, show_element_boundaries, colors, material_names,
                     show_graph, graph_angle, show_levels, graph_color, mesh_opacity, level_mode):
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
    if show_graph or show_levels:
        _add_graph_and_levels(fig, 2, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels, graph_color=graph_color, level_mode=level_mode)
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
                     mesh_opacity, level_mode):
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
    if show_graph or show_levels:
        _add_graph_and_levels(fig, 3, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels, graph_color=graph_color, level_mode=level_mode)
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
                 graph_color="darkred", mesh_opacity=1.0, width=None, height=None):
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
    graph_color : str
        Colour for everything drawn by `show_graph`/`show_levels` (edges, arrowheads, node
        outlines/labels) -- kept a single fixed accent, distinct from the mesh's black element
        boundaries, so a dense adjacency graph doesn't visually merge into the mesh underneath it.
    mesh_opacity : float
        Opacity (0-1) applied to the underlying mesh (fill and/or boundary lines/wireframe) only
        -- the graph/level overlay is unaffected. Lower it (e.g. 0.25-0.4) to fade a crowded mesh
        out of the way when `show_graph`/`show_levels` is the point of the plot.
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
                  show_levels=show_levels, graph_color=graph_color, mesh_opacity=mesh_opacity)
    if dim == 3:
        kwargs["show_material_interfaces"] = show_material_interfaces
    fig = _DISPATCH[dim](domain, **kwargs)
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

