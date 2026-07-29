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


def _bearing_degrees(delta):
    '''Compass bearing (0 = +y/"up", clockwise) matching `marker.angle`'s default `angleref="up"`,
    so a `symbol="triangle-up"` rotated by this angle points along `delta`.'''
    return (90.0 - onp.degrees(onp.arctan2(delta[:, 1], delta[:, 0]))) % 360.0


def _add_angle_compass(fig, dim, domain, angle):
    '''
    A small arrow, legend-labelled "Angle", showing the overall sweep direction `angle` itself
    (as opposed to the per-edge arrowheads, which show the resulting element dependencies).
    Placed just outside the domain's bounding box, scaled to its extent, as a real trace (so it
    gets a legend entry, unlike a paper-space annotation) -- in 3D this also means it rotates
    correctly with the scene camera.
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
    tip = tail + direction * 0.08 * extent

    angle_color = "red"  # matches the default plotly qualitative palette's first color

    if dim == 3:
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


def _add_graph_and_levels(fig, dim, domain, *, show_graph, graph_angle, show_levels):
    '''
    Overlay the element-adjacency graph and/or per-element sweep levels.

    Nodes: a plain black dot if `show_levels` is False, otherwise a white/black-outlined circle
    with the element's sweep level (topological-sort group, from `compute_levels_domain_safe`)
    as text inside it -- `show_levels` requires `graph_angle` (levels are angle-dependent).

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
                              line=dict(width=1.5, color="black"), name="graph edges"))

        if directed and len(edges) > 0:
            src, dst = centroids[edges[:, 0]], centroids[edges[:, 1]]
            mid = (src + dst) / 2
            if dim == 3:
                # Scatter3d markers have no rotation, so approximate the arrowhead with a dot
                # offset toward the downstream element instead of a true oriented triangle.
                tip = src * 0.25 + dst * 0.75
                fig.add_trace(_trace(tip[:, 0], tip[:, 1], tip[:, 2], mode="markers",
                                      marker=dict(size=4, color="black", symbol="circle"),
                                      name="sweep direction"))
            else:
                bearing = _bearing_degrees(dst - src)
                fig.add_trace(_trace(mid[:, 0], mid[:, 1], None, mode="markers",
                                      marker=dict(size=9, color="black", symbol="triangle-up",
                                                  angle=bearing),
                                      name="sweep direction"))
            _add_angle_compass(fig, dim, domain, graph_angle)

    if show_levels:
        levels = _element_levels(domain, graph_angle)
        fig.add_trace(_trace(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], mode="markers+text",
            marker=dict(size=20, color="white", line=dict(color="black", width=1.5), symbol="circle"),
            text=[str(level) for level in levels], textposition="middle center",
            textfont=dict(size=10, color="black"), name="levels",
        ))
    elif show_graph:
        fig.add_trace(_trace(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], mode="markers",
            marker=dict(size=5, color="black", symbol="circle"), name="graph nodes",
        ))


def _plot_domain_1d(domain, *, color_by_material, show_element_boundaries, colors, material_names,
                     show_graph, graph_angle, show_levels):
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
            marker=dict(size=6, color=_color(colors, color_idx)),
            name=_material_label(material, material_names), showlegend=color_by_material,
        ))

    if show_element_boundaries:
        # a vertical tick at every vertex, marking element boundaries.
        tick_x, tick_y = [], []
        for v in onp.unique(vertices):
            tick_x += [v, v, None]
            tick_y += [-0.5, 0.5, None]
        fig.add_trace(go.Scatter(
            x=tick_x, y=tick_y, mode="lines",
            line=dict(width=1, color="black"), showlegend=False, hoverinfo="skip",
        ))

    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(xaxis_title="x", margin=dict(l=0, r=0, t=40, b=40))
    if show_graph or show_levels:
        _add_graph_and_levels(fig, 1, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels)
    return fig


def _plot_domain_2d(domain, *, color_by_material, show_element_boundaries, colors, material_names,
                     show_graph, graph_angle, show_levels):
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
                x=x, y=y, mode="lines", fill="toself",
                line=dict(width=1 if show_element_boundaries else 0, color="black"),
                fillcolor=_color(colors, color_idx),
                name=_material_label(material, material_names), legendgroup=f"material-{material}",
                showlegend=color_by_material and k == 0,
            ))
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    if show_graph or show_levels:
        _add_graph_and_levels(fig, 2, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels)
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
                     colors, material_names, show_graph, graph_angle, show_levels):
    vertices = onp.asarray(domain.vertices)
    groups   = _extract_faces(domain, color_by_material=color_by_material,
                               show_material_interfaces=show_material_interfaces)

    fig = go.Figure()
    for color_idx, (material, faces) in enumerate(sorted(groups.items())):
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=_color(colors, color_idx), opacity=0.6, flatshading=True,
            name=_material_label(material, material_names), showlegend=color_by_material,
        ))
        if show_element_boundaries:
            edge_x, edge_y, edge_z = _triangle_edges(vertices, faces)
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode="lines",
                line=dict(width=2, color="black"), showlegend=False, hoverinfo="skip",
            ))
    fig.update_layout(scene=dict(aspectmode="data"))
    if show_graph or show_levels:
        _add_graph_and_levels(fig, 3, domain, show_graph=show_graph, graph_angle=graph_angle,
                               show_levels=show_levels)
    return fig


_DISPATCH = {1: _plot_domain_1d, 2: _plot_domain_2d, 3: _plot_domain_3d}


def plot_domain(domain, *, color_by_material=True, show_element_boundaries=True,
                 show_material_interfaces=True, colors=None, material_names=None,
                 show_graph=False, graph_angle=None, show_levels=False):
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
        Overlay the element-adjacency graph: a black circle at every element centroid,
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
        Label every element with its sweep level -- the topological-sort group from
        `jax_sn.domain.graph_generation.compute_levels_domain_safe` for `graph_angle` (elements
        in the same level are mutually independent and solvable in parallel). Requires
        `graph_angle` to be given.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    '''
    dim = onp.asarray(domain.vertices).shape[1]
    if dim not in _DISPATCH:
        raise ValueError(f"plot_domain only supports 1D/2D/3D domains, got dimension {dim}")
    if show_levels and graph_angle is None:
        raise ValueError("show_levels=True requires graph_angle (levels are angle-dependent).")

    kwargs = dict(color_by_material=color_by_material,
                  show_element_boundaries=show_element_boundaries, colors=colors,
                  material_names=material_names, show_graph=show_graph, graph_angle=graph_angle,
                  show_levels=show_levels)
    if dim == 3:
        kwargs["show_material_interfaces"] = show_material_interfaces
    return _DISPATCH[dim](domain, **kwargs)
