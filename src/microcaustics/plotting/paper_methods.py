"""Exact reproduction of the paper's source-scout/IPM method figure.

This module intentionally keeps the publication renderer separate from the
general plotting helpers. Tutorials can therefore reproduce the documented
Q2237 B schematic exactly, while the generic methods module remains available
for figures generated from arbitrary simulations.
"""

# This renderer is deliberately frozen in its publication form.  In particular,
# its compact one-line plotting calls are retained so notebook output cannot
# drift from the accepted manuscript asset during routine style refactors.
# ruff: noqa

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    ConnectionPatch,
    FancyArrowPatch,
    Patch,
    Polygon,
    Rectangle,
)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


def _valid_theta_einstein_uas(value):
    """Return a finite positive mean-mass Einstein radius when available."""

    try:
        value = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return np.nan
    return value if np.isfinite(value) and value > 0.0 else np.nan


def _star_sizes_from_mass(
    mass,
    base_size=0.55,
    mean_mass=0.3,
    min_size=0.15,
    max_size=5.5,
):
    """Return the marker areas used by the published microlens schematic."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    if mass.size == 0:
        return None
    mass = np.clip(mass, 1.0e-12, None)
    return np.clip(base_size * (mass / mean_mass), min_size, max_size)


def make_exact_ipm_schematic(tile_data_path, output_dir, refinement=2, virtual_refinement=4, *, native_pixel_grid=True, source_bins=1024, output_prefix='paper_tile_exact_ipm_schematic'):
    """Draw the tile-selected exact-IPM method in the paper schematic style.

    The left panel deliberately matches ``paper_tile_upsampling_schematic``.
    The right inset replaces the bilinear U-sample cloud with the actual
    production biquadratic reconstruction and illustrates the exact area
    deposited into source pixels by the analytic scanline backend.
    """
    refinement = max(1, int(refinement))
    virtual_refinement = max(refinement, int(virtual_refinement))
    native_pixel_grid = bool(native_pixel_grid)
    source_bins = max(1, int(source_bins))
    required = ('probe_x_edges_uas', 'probe_y_edges_uas', 'selection_stage', 'displayed_lens_quad_uas', 'displayed_mapped_quad_uas', 'displayed_refinement_source_grid_uas', 'displayed_fine_cell', 'fine_grid_nx', 'fine_grid_ny', 'displayed_scout_grid_ratio', 'source_extent_uas', 'stellar_aperture_radius_uas', 'stellar_aperture_center_uas', 'source_npz')
    with np.load(tile_data_path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data]
        if missing:
            print(f"[exact IPM schematic] missing tile-schematic fields. Skipping ({', '.join(missing)}).")
            return []
        probe_x = np.asarray(data['probe_x_edges_uas'], dtype=float)
        probe_y = np.asarray(data['probe_y_edges_uas'], dtype=float)
        selection_stage = np.asarray(data['selection_stage'], dtype=np.uint8)
        lens_quad = np.asarray(data['displayed_lens_quad_uas'], dtype=float)
        source_quad = np.asarray(data['displayed_mapped_quad_uas'], dtype=float)
        stored_source_grid = np.asarray(data['displayed_refinement_source_grid_uas'], dtype=float)
        fine_cell = np.asarray(data['displayed_fine_cell'], dtype=np.int32)
        fine_nx = int(np.asarray(data['fine_grid_nx']).item())
        fine_ny = int(np.asarray(data['fine_grid_ny']).item())
        scout_ratio = int(np.asarray(data['displayed_scout_grid_ratio']).item())
        source_extent = tuple(np.asarray(data['source_extent_uas'], dtype=float).reshape(4))
        theta_ein_uas = _valid_theta_einstein_uas(data.get('mean_mass_einstein_radius_uas', np.nan))
        stellar_aperture_radius = float(np.asarray(data['stellar_aperture_radius_uas']).item())
        stellar_aperture_center = np.asarray(data['stellar_aperture_center_uas'], dtype=float).reshape(2)
        source_npz = str(np.asarray(data['source_npz']).item())
        saved_boundary = np.asarray(data.get('displayed_selected_union_boundary_segments_uas', np.empty((0, 2, 2))), dtype=float).reshape(-1, 2, 2)
        saved_vertex_numbers = np.asarray(data.get('displayed_vertex_numbers', (4, 3, 2, 1)), dtype=np.int32).reshape(4)
    if os.name == 'nt' and source_npz.startswith('/mnt/'):
        path_parts = source_npz.split('/', 3)
        if len(path_parts) == 4 and len(path_parts[2]) == 1:
            source_npz = f'{path_parts[2].upper()}:/{path_parts[3]}'
    if not os.path.exists(source_npz):
        print(f'[exact IPM schematic] source benchmark archive is absent. Skipping ({source_npz}).')
        return []
    with np.load(source_npz, allow_pickle=True) as source_data:
        star_x = np.asarray(source_data['star_x'], dtype=float).reshape(-1)
        star_y = np.asarray(source_data['star_y'], dtype=float).reshape(-1)
        star_mass = np.asarray(source_data.get('star_mass', np.ones_like(star_x)), dtype=float).reshape(-1)
    expected_shape = (probe_x.size - 1, probe_y.size - 1)
    if selection_stage.shape != expected_shape:
        print(f'[exact IPM schematic] selection mask and probe edges disagree. Mask={selection_stage.shape}, edges={expected_shape}.')
        return []
    tile_color = '#1F77B4'
    tile_fill_color = '#D9EAF7'
    source_plane_color = '#FF7F0E'
    vertex_color = '#2CA02C'
    virtual_vertex_color = '#1F77B4'
    fine_grid_color = '0.53'
    pixel_grid_color = '#9EC3D8'
    overlap_color = '#E69F00'

    def _nice_scale_length(width):
        target = 0.2 * float(width)
        exponent = np.floor(np.log10(max(target, 1e-12)))
        base = 10.0 ** exponent
        choices = [value * base for value in (1.0, 2.0, 5.0) if value * base <= target]
        return max(choices) if choices else base

    def _add_scalebar(ax, extent, location='lower left', pad=0.28, borderpad=0.36, font_size=8.3, anchor_extent=None):
        x0, x1, y0, y1 = [float(value) for value in extent]
        if anchor_extent is None:
            anchor_extent = extent
        anchor_x0, anchor_x1, anchor_y0, anchor_y1 = [float(value) for value in anchor_extent]
        length = _nice_scale_length(x1 - x0)
        bar = AnchoredSizeBar(ax.transData, length, f'{length:g} µas', location, pad=pad, borderpad=borderpad, sep=3, color='black', frameon=False, size_vertical=max((y1 - y0) * 0.005, 1e-12), fontproperties=fm.FontProperties(size=font_size), bbox_to_anchor=(anchor_x0, anchor_y0, anchor_x1 - anchor_x0, anchor_y1 - anchor_y0), bbox_transform=ax.transData)
        bar.set_zorder(30)
        ax.add_artist(bar)

    def _finish_axis(ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_edgecolor('0.25')

    def _square_view(points, pad_fraction=0.24):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        center = np.nanmean(points, axis=0)
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-12)
        half = 0.5 * span * (1.0 + 2.0 * float(pad_fraction))
        return (center[0] - half, center[0] + half, center[1] - half, center[1] + half)

    def _resample_grid(grid, side):
        """Resample saved true-node geometry in its lens-cell coordinates."""
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 3 or grid.shape[2] != 2 or min(grid.shape[:2]) < 2:
            q00, q10, q11, q01 = source_quad
            uv = np.linspace(0.0, 1.0, side)
            uu, vv = np.meshgrid(uv, uv, indexing='ij')
            return ((1.0 - uu) * (1.0 - vv))[..., None] * q00 + (uu * (1.0 - vv))[..., None] * q10 + (uu * vv)[..., None] * q11 + ((1.0 - uu) * vv)[..., None] * q01
        old_x = np.linspace(0.0, 1.0, grid.shape[0])
        old_y = np.linspace(0.0, 1.0, grid.shape[1])
        new_axis = np.linspace(0.0, 1.0, side)
        result = np.empty((side, side, 2), dtype=float)
        for new_i, u in enumerate(new_axis):
            i1 = min(int(np.searchsorted(old_x, u, side='right')), len(old_x) - 1)
            i0 = max(0, i1 - 1)
            fu = 0.0 if i1 == i0 else (u - old_x[i0]) / (old_x[i1] - old_x[i0])
            for new_j, v in enumerate(new_axis):
                j1 = min(int(np.searchsorted(old_y, v, side='right')), len(old_y) - 1)
                j0 = max(0, j1 - 1)
                fv = 0.0 if j1 == j0 else (v - old_y[j0]) / (old_y[j1] - old_y[j0])
                result[new_i, new_j] = (1.0 - fu) * (1.0 - fv) * grid[i0, j0] + fu * (1.0 - fv) * grid[i1, j0] + fu * fv * grid[i1, j1] + (1.0 - fu) * fv * grid[i0, j1]
        return result

    def _clip_half_plane(poly, axis, value, keep_greater):
        if len(poly) == 0:
            return poly
        clipped = []
        previous = np.asarray(poly[-1], dtype=float)
        previous_inside = previous[axis] >= value if keep_greater else previous[axis] <= value
        for current in np.asarray(poly, dtype=float):
            current_inside = current[axis] >= value if keep_greater else current[axis] <= value
            if current_inside != previous_inside:
                delta = current - previous
                if abs(delta[axis]) > 1e-15:
                    fraction = (value - previous[axis]) / delta[axis]
                    clipped.append(previous + fraction * delta)
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        return np.asarray(clipped, dtype=float).reshape(-1, 2)

    def _clip_to_pixel(triangle, x0, x1, y0, y1):
        poly = np.asarray(triangle, dtype=float)
        for axis, value, keep_greater in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
            poly = _clip_half_plane(poly, axis, value, keep_greater)
            if len(poly) == 0:
                break
        return poly

    def _polygon_area(poly):
        if len(poly) < 3:
            return 0.0
        return 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    selected_mask = selection_stage > 0
    selected_indices = np.argwhere(selected_mask)
    if selected_indices.size:
        center_index = np.asarray(selection_stage.shape, dtype=float) / 2.0
        centrality = np.sum((selected_indices - center_index) ** 2, axis=1)
        zoom_ix, zoom_iy = selected_indices[int(np.argmin(centrality))]
    else:
        zoom_ix, zoom_iy = np.asarray(selection_stage.shape) // 2
    if fine_cell.size == 2 and lens_quad.shape == (4, 2):
        zoom_ix = int(np.searchsorted(probe_x, np.mean(lens_quad[:, 0])) - 1)
        zoom_iy = int(np.searchsorted(probe_y, np.mean(lens_quad[:, 1])) - 1)
    half_cells = 1
    zx0 = max(0, int(zoom_ix) - half_cells)
    zx1 = min(selection_stage.shape[0], int(zoom_ix) + half_cells + 1)
    zy0 = max(0, int(zoom_iy) - half_cells)
    zy1 = min(selection_stage.shape[1], int(zoom_iy) + half_cells + 1)
    zoom_extent = (float(probe_x[zx0]), float(probe_x[zx1]), float(probe_y[zy0]), float(probe_y[zy1]))
    field_view = (float(probe_x[0]), float(probe_x[-1]), float(probe_y[0]), float(probe_y[-1]))
    true_source_nodes = _resample_grid(stored_source_grid, refinement + 1)

    def _biquadratic_virtual_grid(grid, side):
        if tuple(grid.shape[:2]) != (3, 3):
            return _resample_grid(grid, side + 1)
        coordinate = np.linspace(0.0, 1.0, side + 1)
        basis = np.stack((2.0 * (coordinate - 0.5) * (coordinate - 1.0), -4.0 * coordinate * (coordinate - 1.0), 2.0 * coordinate * (coordinate - 0.5)), axis=1)
        return np.einsum('ui,ijc,vj->uvc', basis, grid, basis)
    source_nodes = _biquadratic_virtual_grid(true_source_nodes, virtual_refinement)
    true_lens_axis = np.linspace(0.0, 1.0, refinement + 1)
    virtual_lens_axis = np.linspace(0.0, 1.0, virtual_refinement + 1)
    q00, q10, q11, q01 = lens_quad
    true_lens_nodes = np.empty_like(true_source_nodes)
    for i, u in enumerate(true_lens_axis):
        for j, v in enumerate(true_lens_axis):
            true_lens_nodes[i, j] = (1.0 - u) * (1.0 - v) * q00 + u * (1.0 - v) * q10 + u * v * q11 + (1.0 - u) * v * q01
    lens_nodes = np.empty_like(source_nodes)
    for i, u in enumerate(virtual_lens_axis):
        for j, v in enumerate(virtual_lens_axis):
            lens_nodes[i, j] = (1.0 - u) * (1.0 - v) * q00 + u * (1.0 - v) * q10 + u * v * q11 + (1.0 - u) * v * q01
    triangles = []
    lens_triangles = []
    for i in range(virtual_refinement):
        for j in range(virtual_refinement):
            b00 = source_nodes[i, j]
            b10 = source_nodes[i + 1, j]
            b11 = source_nodes[i + 1, j + 1]
            b01 = source_nodes[i, j + 1]
            triangles.extend(((b00, b10, b11), (b00, b11, b01)))
            l00 = lens_nodes[i, j]
            l10 = lens_nodes[i + 1, j]
            l11 = lens_nodes[i + 1, j + 1]
            l01 = lens_nodes[i, j + 1]
            lens_triangles.extend(((l00, l10, l11), (l00, l11, l01)))
    triangles = np.asarray(triangles, dtype=float)
    lens_triangles = np.asarray(lens_triangles, dtype=float)
    fig, (ax_probe, ax_source) = plt.subplots(1, 2, figsize=(7.5, 3.95))
    ax_lens_zoom = ax_probe.inset_axes((0.54, 0.0, 0.46, 0.46))
    ax_overlap = ax_source.inset_axes((0.54, 0.0, 0.46, 0.46))
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.055, top=0.94, wspace=0.11)
    finite_stars = np.isfinite(star_x) & np.isfinite(star_y)
    if np.any(finite_stars):
        star_sizes = _star_sizes_from_mass(star_mass)
        if star_sizes is None or star_sizes.size != star_x.size:
            star_sizes = np.full(star_x.shape, 0.55, dtype=float)
        ax_probe.scatter(star_x[finite_stars], star_y[finite_stars], s=star_sizes[finite_stars], color='red', linewidths=0, alpha=0.65, rasterized=True, zorder=1)
    selected_rgba = np.zeros((selected_mask.shape[1], selected_mask.shape[0], 4), dtype=np.float32)
    selected_rgba[selected_mask.T] = mcolors.to_rgba(tile_color, 0.9)
    ax_probe.imshow(selected_rgba, origin='lower', extent=field_view, interpolation='nearest', aspect='equal', rasterized=True, zorder=3)
    if saved_boundary.size:
        ax_probe.add_collection(LineCollection(saved_boundary, colors='0.08', linewidths=0.34, alpha=0.94, rasterized=True, zorder=4))
    if np.isfinite(stellar_aperture_radius) and stellar_aperture_radius > 0.0:
        ax_probe.add_patch(Circle(tuple(stellar_aperture_center), radius=stellar_aperture_radius, fill=False, edgecolor='0.30', linewidth=1.05, linestyle=(0, (5, 3)), zorder=4))
    ax_probe.set_xlim(field_view[0], field_view[1])
    ax_probe.set_ylim(field_view[2], field_view[3])
    ax_probe.set_title('Lens-plane tile selection', pad=4, fontsize=9.2)
    _add_scalebar(ax_probe, field_view)
    ax_probe.text(0.04, 0.95, f'{100.0 * np.mean(selected_mask):.2g}% scout cells\nretained', transform=ax_probe.transAxes, ha='left', va='top', fontsize=7.2, color='0.20', bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=1.1), zorder=20)
    _finish_axis(ax_probe)
    ax_lens_zoom.set_facecolor('#FAFAFA')
    for cell_ix in range(zx0, zx1):
        for cell_iy in range(zy0, zy1):
            ax_lens_zoom.add_patch(Rectangle((probe_x[cell_ix], probe_y[cell_iy]), probe_x[cell_ix + 1] - probe_x[cell_ix], probe_y[cell_iy + 1] - probe_y[cell_iy], facecolor=tile_fill_color if selected_mask[cell_ix, cell_iy] else 'white', edgecolor='none', linewidth=0.0, zorder=2))
    if fine_nx > 1 and fine_ny > 1 and (fine_cell.size == 2):
        fine_dx = float(lens_quad[1, 0] - lens_quad[0, 0])
        fine_dy = float(lens_quad[3, 1] - lens_quad[0, 1])
        fine_x_origin = float(lens_quad[0, 0]) - int(fine_cell[0]) * fine_dx
        fine_y_origin = float(lens_quad[0, 1]) - int(fine_cell[1]) * fine_dy
        fine_ix0 = max(0, int(np.ceil((zoom_extent[0] - fine_x_origin) / fine_dx)))
        fine_ix1 = min(fine_nx - 1, int(np.floor((zoom_extent[1] - fine_x_origin) / fine_dx)))
        fine_iy0 = max(0, int(np.ceil((zoom_extent[2] - fine_y_origin) / fine_dy)))
        fine_iy1 = min(fine_ny - 1, int(np.floor((zoom_extent[3] - fine_y_origin) / fine_dy)))
        for fine_ix in range(fine_ix0, fine_ix1 + 1):
            value = fine_x_origin + fine_ix * fine_dx
            ax_lens_zoom.plot((value, value), (zoom_extent[2], zoom_extent[3]), color=fine_grid_color, linewidth=0.3, alpha=0.78, zorder=4)
        for fine_iy in range(fine_iy0, fine_iy1 + 1):
            value = fine_y_origin + fine_iy * fine_dy
            ax_lens_zoom.plot((zoom_extent[0], zoom_extent[1]), (value, value), color=fine_grid_color, linewidth=0.3, alpha=0.78, zorder=4)
    for scout_ix in range(zx0, zx1 + 1):
        value = probe_x[scout_ix]
        ax_lens_zoom.plot((value, value), (zoom_extent[2], zoom_extent[3]), color=tile_color, linewidth=0.62, alpha=0.94, zorder=4.5)
    for scout_iy in range(zy0, zy1 + 1):
        value = probe_y[scout_iy]
        ax_lens_zoom.plot((zoom_extent[0], zoom_extent[1]), (value, value), color=tile_color, linewidth=0.62, alpha=0.94, zorder=4.5)
    if saved_boundary.size:
        ax_lens_zoom.add_collection(LineCollection(saved_boundary, colors='0.08', linewidths=0.72, zorder=6))
    zoom_stars = finite_stars & (star_x >= zoom_extent[0]) & (star_x <= zoom_extent[1]) & (star_y >= zoom_extent[2]) & (star_y <= zoom_extent[3])
    if np.any(zoom_stars):
        zoom_sizes = _star_sizes_from_mass(star_mass, base_size=0.38, min_size=0.1, max_size=3.5)
        ax_lens_zoom.scatter(star_x[zoom_stars], star_y[zoom_stars], s=zoom_sizes[zoom_stars], color='red', linewidths=0, alpha=0.65, zorder=7)
    ax_lens_zoom.add_patch(Polygon(lens_quad, closed=True, facecolor='none', edgecolor='0.05', linewidth=1.0, zorder=11))
    lens_triangle_segments = []
    for triangle in lens_triangles:
        lens_triangle_segments.extend(((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])))
    ax_lens_zoom.add_collection(LineCollection(np.asarray(lens_triangle_segments), colors=tile_color, linewidths=0.48, alpha=0.94, zorder=11.5))
    flat_lens_nodes = lens_nodes.reshape(-1, 2)
    ax_lens_zoom.scatter(flat_lens_nodes[:, 0], flat_lens_nodes[:, 1], s=6.0, marker='o', facecolor='white', edgecolor=virtual_vertex_color, linewidth=0.28, zorder=11.8)
    flat_true_lens_nodes = true_lens_nodes.reshape(-1, 2)
    ax_lens_zoom.scatter(flat_true_lens_nodes[:, 0], flat_true_lens_nodes[:, 1], s=12.0, marker='o', facecolor=vertex_color, edgecolor='black', linewidth=0.3, zorder=12)
    lens_center = lens_quad.mean(axis=0)
    lens_width = max(float(np.ptp(lens_quad[:, 0])), float(np.ptp(lens_quad[:, 1])))
    for number, point in zip(saved_vertex_numbers, lens_quad):
        direction = point - lens_center
        label_point = point + 0.3 * lens_width * direction / max(np.linalg.norm(direction), 1e-12)
        ax_lens_zoom.text(label_point[0], label_point[1], str(number), ha='center', va='center', fontsize=6.8, color='0.15', zorder=13)
    ax_lens_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
    ax_lens_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
    ax_lens_zoom.text(0.5, 0.985, 'scout tiles and fine grid', transform=ax_lens_zoom.transAxes, ha='center', va='top', fontsize=7.0, bbox=dict(facecolor='white', edgecolor='none', alpha=0.84, pad=0.6), zorder=20)
    _add_scalebar(ax_lens_zoom, zoom_extent, pad=0.16, borderpad=0.2, font_size=6.4)
    _finish_axis(ax_lens_zoom)
    sx0, sx1, sy0, sy1 = source_extent
    source_view = (sx0, sx1, sy0, sy1)
    ax_source.set_facecolor('white')
    ax_source.add_patch(Polygon(source_quad, closed=True, facecolor=mcolors.to_rgba(tile_color, 0.08), edgecolor=tile_color, linewidth=1.35, zorder=3))
    flat_true_nodes = true_source_nodes.reshape(-1, 2)
    true_inside = (flat_true_nodes[:, 0] >= sx0) & (flat_true_nodes[:, 0] <= sx1) & (flat_true_nodes[:, 1] >= sy0) & (flat_true_nodes[:, 1] <= sy1)
    if np.any(true_inside):
        ax_source.scatter(flat_true_nodes[true_inside, 0], flat_true_nodes[true_inside, 1], s=2.8, marker='o', facecolor=vertex_color, edgecolor='black', linewidth=0.24, clip_on=False, zorder=5)
    if np.any(~true_inside):
        ax_source.scatter(flat_true_nodes[~true_inside, 0], flat_true_nodes[~true_inside, 1], s=4.5, marker='x', color='#D62728', linewidth=0.62, clip_on=False, zorder=6)
    ax_source.set_xlim(source_view[0], source_view[1])
    ax_source.set_ylim(source_view[2], source_view[3])
    ax_source.set_title('Exact IPM in the source plane', pad=5, fontsize=9.8)
    _add_scalebar(ax_source, source_extent, anchor_extent=source_extent)
    _finish_axis(ax_source)
    overlap_view = _square_view(source_nodes, pad_fraction=0.24)
    ax_overlap.set_facecolor(mcolors.to_rgba(source_plane_color, 0.055))
    node_span = max(float(np.ptp(source_nodes[..., 0])), float(np.ptp(source_nodes[..., 1])), 1e-12)
    if native_pixel_grid:
        pixel_size = (sx1 - sx0) / float(source_bins)
        x_start = sx0 + np.floor((overlap_view[0] - sx0) / pixel_size) * pixel_size
        y_start = sy0 + np.floor((overlap_view[2] - sy0) / pixel_size) * pixel_size
        pixel_grid_linewidth = 0.24
        pixel_grid_alpha = 0.55
        pixel_grid_zorder = 2.65
    else:
        pixel_size = node_span / 8.0
        x_start = np.floor(overlap_view[0] / pixel_size) * pixel_size
        y_start = np.floor(overlap_view[2] / pixel_size) * pixel_size
        pixel_grid_linewidth = 0.35
        pixel_grid_alpha = 0.75
        pixel_grid_zorder = 2.65
    pixel_x = np.arange(x_start, overlap_view[1] + pixel_size, pixel_size)
    pixel_y = np.arange(y_start, overlap_view[3] + pixel_size, pixel_size)
    for value in pixel_x:
        ax_overlap.plot((value, value), (overlap_view[2], overlap_view[3]), color=pixel_grid_color, linewidth=pixel_grid_linewidth, alpha=pixel_grid_alpha, zorder=pixel_grid_zorder)
    for value in pixel_y:
        ax_overlap.plot((overlap_view[0], overlap_view[1]), (value, value), color=pixel_grid_color, linewidth=pixel_grid_linewidth, alpha=pixel_grid_alpha, zorder=pixel_grid_zorder)
    triangle_areas = np.asarray([_polygon_area(triangle) for triangle in triangles])
    triangle_centers = triangles.mean(axis=1)
    source_center = source_nodes.reshape(-1, 2).mean(axis=0)
    score = np.linalg.norm(triangle_centers - source_center, axis=1) / np.sqrt(np.maximum(triangle_areas, 1e-30))
    highlighted_index = int(np.argmin(score))
    highlighted_triangle = triangles[highlighted_index]
    ax_overlap.add_patch(Polygon(highlighted_triangle, closed=True, facecolor='none', edgecolor='none', zorder=1.2))
    for x0, x1 in zip(pixel_x[:-1], pixel_x[1:]):
        for y0, y1 in zip(pixel_y[:-1], pixel_y[1:]):
            clipped = _clip_to_pixel(highlighted_triangle, x0, x1, y0, y1)
            area_fraction = _polygon_area(clipped) / (pixel_size * pixel_size)
            if area_fraction <= 1e-08:
                continue
            ax_overlap.add_patch(Polygon(clipped, closed=True, facecolor=mcolors.to_rgba(overlap_color, 0.5), edgecolor='none', linewidth=0.0, zorder=2.2))
    triangle_segments = []
    for triangle in triangles:
        triangle_segments.extend(((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])))
    ax_overlap.add_collection(LineCollection(np.asarray(triangle_segments), colors=tile_color, linewidths=0.55, alpha=0.94, zorder=3.5))
    ax_overlap.add_patch(Polygon(source_quad, closed=True, facecolor='none', edgecolor=tile_color, linewidth=0.9, zorder=4))
    flat_nodes = source_nodes.reshape(-1, 2)
    ax_overlap.scatter(flat_nodes[:, 0], flat_nodes[:, 1], s=1.5, marker='o', facecolor='white', edgecolor=virtual_vertex_color, linewidth=0.24, zorder=4.8)
    if np.any(true_inside):
        ax_overlap.scatter(flat_true_nodes[true_inside, 0], flat_true_nodes[true_inside, 1], s=3.2, marker='o', facecolor=vertex_color, edgecolor='black', linewidth=0.22, zorder=5)
    if np.any(~true_inside):
        ax_overlap.scatter(flat_true_nodes[~true_inside, 0], flat_true_nodes[~true_inside, 1], s=4.2, marker='x', color='#D62728', linewidth=0.5, zorder=5.2)
    source_center = source_quad.mean(axis=0)
    source_width = max(float(np.ptp(source_quad[:, 0])), float(np.ptp(source_quad[:, 1])))
    for number, point in zip(saved_vertex_numbers, source_quad):
        direction = point - source_center
        label_point = point - 0.11 * source_width * direction / max(np.linalg.norm(direction), 1e-12)
        ax_overlap.text(label_point[0], label_point[1], str(number), ha='center', va='center', fontsize=7.2, fontweight='bold', color='0.10', clip_on=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, pad=0.12), zorder=7)
    ax_overlap.text(0.5, 0.985, 'triangle-pixel overlap', transform=ax_overlap.transAxes, ha='center', va='top', fontsize=7.0, bbox=dict(facecolor='white', edgecolor='none', alpha=0.84, pad=0.6), zorder=20)
    ax_overlap.set_xlim(overlap_view[0], overlap_view[1])
    ax_overlap.set_ylim(overlap_view[2], overlap_view[3])
    _add_scalebar(ax_overlap, overlap_view, pad=0.16, borderpad=0.2, font_size=6.4)
    _finish_axis(ax_overlap)
    for parent_ax, inset_ax, inset_extent in ((ax_probe, ax_lens_zoom, zoom_extent), (ax_source, ax_overlap, overlap_view)):
        parent_ax.add_patch(Rectangle((inset_extent[0], inset_extent[2]), inset_extent[1] - inset_extent[0], inset_extent[3] - inset_extent[2], facecolor='none', edgecolor='0.25', linewidth=0.75, linestyle=(0, (3, 2)), zorder=15))
        for parent_x, inset_x in ((inset_extent[0], 0.0), (inset_extent[1], 1.0)):
            parent_ax.add_artist(ConnectionPatch(xyA=(parent_x, inset_extent[2]), coordsA=parent_ax.transData, xyB=(inset_x, 1.0), coordsB=inset_ax.transAxes, color='0.35', linewidth=0.72, linestyle=(0, (3, 2)), clip_on=False, zorder=14))
    legend_handles = [Line2D([], [], marker='o', linestyle='None', markerfacecolor='red', markeredgecolor='none', markersize=6.5, label='Microlens'), Line2D([], [], marker='s', linestyle='None', markerfacecolor=tile_fill_color, markeredgecolor='0.10', markeredgewidth=0.75, markersize=8.0, label='Retained scout tile'), Line2D([], [], color=fine_grid_color, linewidth=0.8, label=f'Fine integration grid ($k={scout_ratio}$)'), Line2D([], [], marker='o', linestyle='None', markerfacecolor=vertex_color, markeredgecolor='black', markersize=6.0, label='Ray-traced node inside source plane'), Line2D([], [], marker='x', linestyle='None', color='#D62728', markeredgewidth=1.0, markersize=6.0, label='Ray-traced node outside source plane'), Line2D([], [], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor=virtual_vertex_color, markersize=5.0, label='Biquadratic virtual node'), Line2D([], [], color=tile_color, linewidth=1.25, label=f'Mapped triangles ($r={refinement},\\ v={virtual_refinement}$)'), Line2D([], [], color=pixel_grid_color, linewidth=0.75, label='Source-pixel grid' if native_pixel_grid else 'Diagnostic source-pixel grid'), Patch(facecolor=mcolors.to_rgba(overlap_color, 0.7), edgecolor=overlap_color, label='Exact overlap area')]
    legend = ax_source.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.004, 0.996), bbox_transform=ax_source.transAxes, ncol=1, fontsize=6.6, labelspacing=0.62, handlelength=1.34, handletextpad=0.48, borderpad=0.52, frameon=True, facecolor='white', edgecolor='0.35', framealpha=0.94)
    legend.set_zorder(40)
    left_box = ax_probe.get_position()
    right_box = ax_source.get_position()
    fig.add_artist(FancyArrowPatch((left_box.x1 + 0.01, 0.5 * (left_box.y0 + left_box.y1)), (right_box.x0 - 0.01, 0.5 * (right_box.y0 + right_box.y1)), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=11, linewidth=0.9, color='0.28', zorder=20))
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'{output_prefix}.png')
    pdf_path = os.path.join(output_dir, f'{output_prefix}.pdf')
    data_path = os.path.join(output_dir, f'{output_prefix}_data.npz')
    fig.savefig(png_path, dpi=360, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(pdf_path, dpi=360, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    np.savez_compressed(data_path, lens_nodes_uas=lens_nodes.astype(np.float32, copy=False), true_lens_nodes_uas=true_lens_nodes.astype(np.float32, copy=False), source_nodes_uas=source_nodes.astype(np.float32, copy=False), true_source_nodes_uas=true_source_nodes.astype(np.float32, copy=False), source_triangles_uas=triangles.astype(np.float32, copy=False), highlighted_triangle_index=np.asarray(highlighted_index, dtype=np.int32), displayed_refinement=np.asarray(refinement, dtype=np.int32), displayed_virtual_refinement=np.asarray(virtual_refinement, dtype=np.int32), displayed_pixel_size_uas=np.asarray(pixel_size, dtype=np.float32), displayed_pixel_grid_enlarged=np.asarray(not native_pixel_grid, dtype=np.bool_), source_bins=np.asarray(source_bins, dtype=np.int32), source_tile_data=np.asarray(os.path.abspath(tile_data_path)))
    return [png_path, pdf_path, data_path]

def render_paper_ipm_schematic(
    tile_diagnostic_path,
    star_diagnostic_path,
    output_directory,
    *,
    refinement=2,
    virtual_refinement=4,
    source_bins=1024,
    output_prefix="paper_tile_exact_ipm_schematic",
):
    """Reproduce paper Figure 3 from its compact Q2237 B diagnostics.

    The plotting body is the exact manuscript renderer that produced the
    approved figure. This wrapper only makes archived companion paths portable.
    """

    tile_diagnostic_path = Path(tile_diagnostic_path)
    star_diagnostic_path = Path(star_diagnostic_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with np.load(star_diagnostic_path, allow_pickle=False) as stars:
        star_x = np.asarray(
            stars["star_x_uas"] if "star_x_uas" in stars else stars["star_x"]
        )
        star_y = np.asarray(
            stars["star_y_uas"] if "star_y_uas" in stars else stars["star_y"]
        )
        star_mass = np.asarray(stars["star_mass"])
    portable_stars = output_directory / f"{output_prefix}_stars.npz"
    np.savez_compressed(
        portable_stars,
        star_x=star_x,
        star_y=star_y,
        star_mass=star_mass,
    )

    with np.load(tile_diagnostic_path, allow_pickle=True) as tile:
        portable_payload = {key: tile[key] for key in tile.files}
    portable_payload["source_npz"] = np.asarray(str(portable_stars.resolve()))
    portable_tile = output_directory / f"{output_prefix}_tile_data.npz"
    np.savez_compressed(portable_tile, **portable_payload)

    paper_style = {
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 13,
        "axes.linewidth": 1.1,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
    with plt.rc_context(paper_style):
        generated = make_exact_ipm_schematic(
            portable_tile,
            output_directory,
            refinement=refinement,
            virtual_refinement=virtual_refinement,
            native_pixel_grid=True,
            source_bins=source_bins,
            output_prefix=output_prefix,
        )
    return [Path(path) for path in generated]


def render_live_paper_ipm_schematic(
    simulation,
    lens_region,
    source_grid,
    config,
    output_directory,
    *,
    time_days=0.0,
    output_prefix="paper_tile_exact_ipm_schematic",
):
    """Generate the paper IPM schematic from a live simulation.

    This is the data-generating counterpart to
    :func:`render_paper_ipm_schematic`. It evaluates the configured scout and
    ray traces one representative retained fine cell, then passes those live
    diagnostics through the unchanged manuscript renderer. The resulting
    layout, labels, limits, marker conventions, and typography therefore stay
    identical to the published schematic without relying on archived arrays.
    """

    import torch

    from ..solvers.far_field import TaylorFarFieldApproximation
    from ..solvers.ipm import _source_scout_cells

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    far_field = None
    if config.far_field_approx.enabled:
        far_field = TaylorFarFieldApproximation(
            simulation,
            lens_region,
            config.far_field_approx,
            time_days=float(time_days),
        )
    selected, fine_ny, fine_nx, _, _ = _source_scout_cells(
        simulation,
        far_field,
        lens_region,
        source_grid,
        config,
        time_days=float(time_days),
        _return_corners=True,
    )
    if selected.numel() == 0:
        raise RuntimeError("the live paper schematic scout selected no cells")

    xmin, xmax, ymin, ymax = (float(value) for value in lens_region.bounds_uas)
    fine_dx = (xmax - xmin) / int(fine_nx)
    fine_dy = (ymax - ymin) / int(fine_ny)
    rows = torch.div(selected, int(fine_nx), rounding_mode="floor")
    columns = selected - rows * int(fine_nx)
    center_x = xmin + (columns.to(simulation.runtime.dtype) + 0.5) * fine_dx
    center_y = ymin + (rows.to(simulation.runtime.dtype) + 0.5) * fine_dy
    if far_field is None:
        mapped_x, mapped_y, _ = simulation.raytrace_direct(
            center_x, center_y, time_days=float(time_days)
        )
    else:
        mapped_x, mapped_y = far_field.raytrace(center_x, center_y)
    source_center_x = float(source_grid.center_uas[1])
    source_center_y = float(source_grid.center_uas[0])
    center_distance_squared = (
        (mapped_x - source_center_x).square()
        + (mapped_y - source_center_y).square()
    )

    # A source position generally has many microimages. Choosing only the
    # closest mapped center can therefore select a nearly singular cell even
    # when several clearer examples reach the same central source region. For
    # the explanatory schematic, inspect a bounded pool of the closest cells
    # and prefer a well-conditioned mapped quadrilateral. This changes only
    # which already-retained cell is displayed; it does not affect scouting or
    # map construction.
    candidate_count = min(4096, int(selected.numel()))
    candidate_indices = torch.topk(
        center_distance_squared,
        candidate_count,
        largest=False,
    ).indices
    candidate_rows = rows[candidate_indices]
    candidate_columns = columns[candidate_indices]
    candidate_x0 = xmin + candidate_columns.to(simulation.runtime.dtype) * fine_dx
    candidate_y0 = ymin + candidate_rows.to(simulation.runtime.dtype) * fine_dy
    candidate_corner_x = torch.stack(
        (
            candidate_x0,
            candidate_x0 + fine_dx,
            candidate_x0 + fine_dx,
            candidate_x0,
        ),
        dim=1,
    )
    candidate_corner_y = torch.stack(
        (
            candidate_y0,
            candidate_y0,
            candidate_y0 + fine_dy,
            candidate_y0 + fine_dy,
        ),
        dim=1,
    )
    flat_corner_x = candidate_corner_x.reshape(-1)
    flat_corner_y = candidate_corner_y.reshape(-1)
    if far_field is None:
        candidate_source_x, candidate_source_y, _ = simulation.raytrace_direct(
            flat_corner_x,
            flat_corner_y,
            time_days=float(time_days),
        )
    else:
        candidate_source_x, candidate_source_y = far_field.raytrace(
            flat_corner_x,
            flat_corner_y,
        )
    candidate_source_x = candidate_source_x.reshape(candidate_count, 4)
    candidate_source_y = candidate_source_y.reshape(candidate_count, 4)
    edge_x = torch.stack(
        (
            candidate_source_x[:, 1] - candidate_source_x[:, 0],
            candidate_source_y[:, 1] - candidate_source_y[:, 0],
        ),
        dim=1,
    )
    edge_y = torch.stack(
        (
            candidate_source_x[:, 3] - candidate_source_x[:, 0],
            candidate_source_y[:, 3] - candidate_source_y[:, 0],
        ),
        dim=1,
    )
    affine_edges = torch.stack((edge_x, edge_y), dim=2)
    singular_values = torch.linalg.svdvals(affine_edges)
    condition_number = singular_values[:, 0] / singular_values[:, 1].clamp_min(
        torch.finfo(simulation.runtime.dtype).eps
    )
    edge_scale = torch.maximum(
        torch.linalg.vector_norm(edge_x, dim=1),
        torch.linalg.vector_norm(edge_y, dim=1),
    )
    source_xmin, source_xmax, source_ymin, source_ymax = (
        float(value) for value in source_grid.bounds_uas
    )
    source_scale = min(
        float(source_grid.field_of_view_uas[0]),
        float(source_grid.field_of_view_uas[1]),
    )
    central_radius = 0.05 * source_scale
    maximum_edge_scale = 0.15 * source_scale
    candidate_distance = torch.sqrt(center_distance_squared[candidate_indices])
    inside_source = (
        (candidate_source_x >= source_xmin).all(dim=1)
        & (candidate_source_x <= source_xmax).all(dim=1)
        & (candidate_source_y >= source_ymin).all(dim=1)
        & (candidate_source_y <= source_ymax).all(dim=1)
    )
    suitable = (
        inside_source
        & torch.isfinite(condition_number)
        & (candidate_distance <= central_radius)
        & (edge_scale <= maximum_edge_scale)
    )
    if bool(torch.any(suitable)):
        score = (
            torch.log(condition_number)
            + 0.15 * candidate_distance / max(central_radius, 1.0e-12)
            + 0.10 * edge_scale / max(maximum_edge_scale, 1.0e-12)
        )
        score = torch.where(
            suitable,
            score,
            torch.full_like(score, float("inf")),
        )
        representative = candidate_indices[torch.argmin(score)]
    else:
        representative = torch.argmin(center_distance_squared)
    row = int(rows[representative])
    column = int(columns[representative])
    cell_x0 = xmin + column * fine_dx
    cell_y0 = ymin + row * fine_dy

    refinement = int(config.refinement)
    lens_x = torch.linspace(
        cell_x0,
        cell_x0 + fine_dx,
        refinement + 1,
        device=simulation.runtime.device,
        dtype=simulation.runtime.dtype,
    )
    lens_y = torch.linspace(
        cell_y0,
        cell_y0 + fine_dy,
        refinement + 1,
        device=simulation.runtime.device,
        dtype=simulation.runtime.dtype,
    )
    mesh_y, mesh_x = torch.meshgrid(lens_y, lens_x, indexing="ij")
    if far_field is None:
        source_x, source_y, _ = simulation.raytrace_direct(
            mesh_x, mesh_y, time_days=float(time_days)
        )
    else:
        source_x, source_y = far_field.raytrace(mesh_x, mesh_y)
    source_nodes = torch.stack((source_x, source_y), dim=-1)

    lens_quad = np.asarray(
        (
            (cell_x0, cell_y0),
            (cell_x0 + fine_dx, cell_y0),
            (cell_x0 + fine_dx, cell_y0 + fine_dy),
            (cell_x0, cell_y0 + fine_dy),
        ),
        dtype=np.float64,
    )
    nodes_np = source_nodes.detach().cpu().numpy()
    source_quad = np.asarray(
        (nodes_np[0, 0], nodes_np[0, -1], nodes_np[-1, -1], nodes_np[-1, 0]),
        dtype=np.float64,
    )

    ratio = int(config.scout_ratio)
    fine_mask = np.zeros((int(fine_ny), int(fine_nx)), dtype=np.bool_)
    fine_mask[
        rows.detach().cpu().numpy().astype(int),
        columns.detach().cpu().numpy().astype(int),
    ] = True
    coarse_ny = int(fine_ny) // ratio
    coarse_nx = int(fine_nx) // ratio
    coarse_mask_yx = fine_mask.reshape(
        coarse_ny, ratio, coarse_nx, ratio
    ).any(axis=(1, 3))
    selection_stage = coarse_mask_yx.T.astype(np.uint8)
    probe_x = np.linspace(xmin, xmax, coarse_nx + 1)
    probe_y = np.linspace(ymin, ymax, coarse_ny + 1)

    field = simulation.lens_state(float(time_days))
    star_x = field.x_uas.detach().cpu().numpy()
    star_y = field.y_uas.detach().cpu().numpy()
    if field.mass_solar is None:
        star_mass = field.einstein_radius_uas.detach().cpu().numpy() ** 2
    else:
        star_mass = field.mass_solar.detach().cpu().numpy()
    stars_path = output_directory / f"{output_prefix}_live_stars.npz"
    np.savez_compressed(
        stars_path,
        star_x=star_x,
        star_y=star_y,
        star_mass=star_mass,
    )
    source_extent = np.asarray(source_grid.bounds_uas, dtype=np.float64)
    payload_path = output_directory / f"{output_prefix}_live_diagnostics.npz"
    np.savez_compressed(
        payload_path,
        probe_x_edges_uas=probe_x,
        probe_y_edges_uas=probe_y,
        selection_stage=selection_stage,
        displayed_lens_quad_uas=lens_quad,
        displayed_mapped_quad_uas=source_quad,
        displayed_refinement_source_grid_uas=nodes_np,
        displayed_fine_cell=np.asarray((column, row), dtype=np.int32),
        fine_grid_nx=np.asarray(int(fine_nx), dtype=np.int32),
        fine_grid_ny=np.asarray(int(fine_ny), dtype=np.int32),
        displayed_scout_grid_ratio=np.asarray(ratio, dtype=np.int32),
        source_extent_uas=source_extent,
        stellar_aperture_radius_uas=np.asarray(
            0.5 * min(xmax - xmin, ymax - ymin), dtype=np.float64
        ),
        stellar_aperture_center_uas=np.asarray(
            ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0), dtype=np.float64
        ),
        source_npz=np.asarray(str(stars_path.resolve())),
        displayed_vertex_numbers=np.asarray((1, 2, 3, 4), dtype=np.int32),
    )
    return render_paper_ipm_schematic(
        payload_path,
        stars_path,
        output_directory,
        refinement=refinement,
        virtual_refinement=int(config.virtual_refinement),
        source_bins=int(source_grid.shape[1]),
        output_prefix=output_prefix,
    )


def render_paper_anchor_gauge(
    diagnostic_path,
    output_directory,
    *,
    output_prefix="q2237b_anchor_gauge_method",
):
    """Reproduce the manuscript anchor/gauge panel from compact diagnostics.

    The compact archive contains the exact binary field, caustic segments,
    classified probes, and parity-colored ray pieces used by the paper.  This
    function intentionally preserves the manuscript marker sizes, colors,
    scale bar, framing, and external legend placement.
    """

    diagnostic_path = Path(diagnostic_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with np.load(diagnostic_path, allow_pickle=False) as data:
        source_extent = tuple(np.asarray(data["source_extent"], dtype=float))
        outer_extent = tuple(np.asarray(data["outer_extent"], dtype=float))
        binary_map = np.asarray(data["binary_map"], dtype=np.uint8)
        caustics = np.asarray(data["caustic_segments"], dtype=float)
        anchors = np.asarray(data["anchor_points"], dtype=float)
        anchor_labels = np.asarray(data["anchor_labels"], dtype=np.uint8) & 1
        gauges = np.asarray(data["gauge_points"], dtype=float)
        gauge_labels = np.asarray(data["gauge_labels"], dtype=np.uint8) & 1
        center = np.asarray(data["center_point"], dtype=float)
        center_label = int(np.asarray(data["center_label"]).reshape(-1)[0]) & 1
        ray_segments = {
            0: np.asarray(data["ray_segments_class_0"], dtype=float),
            1: np.asarray(data["ray_segments_class_1"], dtype=float),
        }

    class_colors = ("#D97706", "#0072B2")
    region_colors = ("#F9E4B7", "#C9E3F1")
    x0, x1, y0, y1 = source_extent
    source_half = 0.5 * max(abs(x1 - x0), abs(y1 - y0))
    label_y_offset = -0.010 * source_half

    with plt.rc_context({"font.size": 13, "legend.fontsize": 13}):
        fig, ax = plt.subplots(figsize=(8.8, 7.4))
        fig.subplots_adjust(left=0.020, right=0.790, bottom=0.025, top=0.975)
        binary_cmap = mcolors.ListedColormap(region_colors)
        binary_norm = mcolors.BoundaryNorm((-0.5, 0.5, 1.5), binary_cmap.N)
        ax.imshow(
            binary_map.T,
            origin="lower",
            extent=source_extent,
            interpolation="nearest",
            cmap=binary_cmap,
            norm=binary_norm,
            zorder=1,
        )
        for binary_class in (0, 1):
            segments = ray_segments[binary_class].reshape(-1, 2, 2)
            if segments.size:
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=class_colors[binary_class],
                        linewidths=1.25,
                        alpha=1.0,
                        capstyle="butt",
                        zorder=4,
                    )
                )
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="0.18",
                linewidth=1.15,
                zorder=7,
            )
        )
        finite = caustics[np.all(np.isfinite(caustics), axis=(1, 2))]
        if finite.size:
            ax.add_collection(
                LineCollection(finite, colors="black", linewidths=0.85, zorder=8)
            )

        def _plot_labeled_points(points, labels, marker, size, zorder):
            for point, label in zip(points, labels):
                label = int(label) & 1
                ax.scatter(
                    [point[0]],
                    [point[1]],
                    s=size,
                    marker=marker,
                    facecolor=class_colors[label],
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=zorder,
                )
                ax.text(
                    point[0],
                    point[1] + label_y_offset,
                    str(label),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=11.4,
                    fontweight="bold",
                    zorder=zorder + 1,
                )

        _plot_labeled_points(anchors, anchor_labels, "o", 250, 10)
        _plot_labeled_points(gauges, gauge_labels, "D", 150, 10)
        ax.scatter(
            [center[0]],
            [center[1]],
            s=460,
            marker="*",
            facecolor=class_colors[center_label],
            edgecolor="black",
            linewidth=1.0,
            zorder=12,
        )
        ax.text(
            center[0],
            center[1] + label_y_offset,
            str(center_label),
            ha="center",
            va="center",
            color="white",
            fontsize=11.4,
            fontweight="bold",
            zorder=13,
        )

        scale_bar = AnchoredSizeBar(
            ax.transData,
            1.0,
            "1 µas",
            "lower left",
            pad=0.35,
            borderpad=0.5,
            sep=4,
            color="black",
            frameon=False,
            size_vertical=float((y1 - y0) * 0.003),
            fontproperties=fm.FontProperties(size=11.0),
            bbox_to_anchor=(x0, y0, x1 - x0, y1 - y0),
            bbox_transform=ax.transData,
        )
        scale_bar.set_zorder(20)
        ax.add_artist(scale_bar)

        legend_handles = [
            Line2D([], [], color="black", linewidth=1.2, label="Caustic"),
            Patch(facecolor=region_colors[0], edgecolor="0.35", label="Binary region 0"),
            Patch(facecolor=region_colors[1], edgecolor="0.35", label="Binary region 1"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor=class_colors[0], markeredgecolor="black", markersize=12.6, label="Anchor offset 0"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor=class_colors[1], markeredgecolor="black", markersize=12.6, label="Anchor offset 1"),
            Line2D([], [], marker="D", linestyle="None", markerfacecolor=class_colors[0], markeredgecolor="black", markersize=9.8, label="Gauge class 0"),
            Line2D([], [], marker="D", linestyle="None", markerfacecolor=class_colors[1], markeredgecolor="black", markersize=9.8, label="Gauge class 1"),
            Line2D([], [], marker="*", linestyle="None", markerfacecolor=class_colors[center_label], markeredgecolor="black", markersize=16.0, label=f"Center class {center_label}"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.800, 0.500),
            ncol=1,
            fontsize=13.0,
            handlelength=1.7,
            handletextpad=0.70,
            borderpad=0.65,
            labelspacing=0.68,
            frameon=True,
            facecolor="white",
            edgecolor="0.35",
            framealpha=0.94,
        )
        ax.set_xlim(outer_extent[0], outer_extent[1])
        ax.set_ylim(outer_extent[2], outer_extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        output_path = output_directory / f"{output_prefix}.png"
        fig.savefig(output_path, dpi=360, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    return output_path


def render_live_paper_anchor_gauge(
    frame,
    output_directory,
    *,
    output_prefix="q2237b_anchor_gauge_method",
):
    """Generate the paper anchor/gauge panel from a live labeled frame.

    The production label map and probes are evaluated before this function is
    called. This adapter only converts those public result objects into the
    compact representation consumed by the unchanged manuscript renderer.
    """

    if frame.label_map is None:
        raise ValueError("a diagnostic label map is required")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    labels = frame.labels
    source_extent = np.asarray(frame.label_map.grid.bounds_uas, dtype=np.float64)
    width = source_extent[1] - source_extent[0]
    height = source_extent[3] - source_extent[2]
    pad = 0.035 * max(width, height)
    outer_extent = source_extent + np.asarray((-pad, pad, -pad, pad))
    anchors = labels.anchor_points_uas.detach().cpu().numpy()
    gauges = labels.gauge_points_uas.detach().cpu().numpy()
    anchor_labels = labels.anchor_offsets.detach().cpu().numpy().astype(np.uint8) & 1
    gauge_labels = labels.gauge_labels.detach().cpu().numpy().astype(np.uint8) & 1
    center = labels.metadata.get("source_center_uas")
    if center is None:
        center = tuple(gauges.mean(axis=0)) if gauges.size else (0.0, 0.0)
    center = np.asarray(center, dtype=np.float64)
    ray_segments = {0: [], 1: []}
    for point, label in zip(anchors, anchor_labels, strict=True):
        ray_segments[int(label)].append((point, center))
    for label in (0, 1):
        ray_segments[label] = np.asarray(ray_segments[label], dtype=np.float64).reshape(
            -1, 2, 2
        )
    payload_path = output_directory / f"{output_prefix}_live_diagnostics.npz"
    np.savez_compressed(
        payload_path,
        source_extent=source_extent,
        outer_extent=outer_extent,
        # The frozen renderer uses x-major storage and transposes for imshow.
        binary_map=frame.label_map.values.detach().cpu().numpy().T.astype(np.uint8),
        caustic_segments=frame.caustics.caustic_segments_uas.detach().cpu().numpy(),
        anchor_points=anchors,
        anchor_labels=anchor_labels,
        gauge_points=gauges,
        gauge_labels=gauge_labels,
        center_point=center,
        center_label=np.asarray(int(labels.center_label) & 1, dtype=np.uint8),
        ray_segments_class_0=ray_segments[0],
        ray_segments_class_1=ray_segments[1],
    )
    return render_paper_anchor_gauge(
        payload_path,
        output_directory,
        output_prefix=output_prefix,
    )


def render_paper_sim5_validation(
    case_directory,
    output_directory,
    *,
    output_prefix="sim5_gr_visual_validation",
):
    """Reproduce the manuscript's two-row GR comparison against SIM5.

    The saved validation arrays retain every raw residual used for the reported
    metrics.  For display only, the two screen rows adjacent to the projected
    symmetry axis are linearly repaired.  The independent solvers choose
    opposite limiting branches exactly on that axis, producing a one-pixel
    raster seam that is not a physical disk feature.
    """

    case_directory = Path(case_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with np.load(case_directory / "pytorch_analytic_raw.npz", allow_pickle=False) as data:
        pytorch_g = np.asarray(data["gfactor"], dtype=float)
    with np.load(case_directory / "sim5_raw.npz", allow_pickle=False) as data:
        sim5_g = np.asarray(data["gfactor"], dtype=float)
    with np.load(case_directory / "comparison_data.npz", allow_pickle=False) as data:
        intensity_pytorch = np.asarray(data["intensity_pytorch_analytic"], dtype=float)
        intensity_sim5 = np.asarray(data["intensity_sim5"], dtype=float)
        g_residual = np.asarray(data["gfactor_fractional"], dtype=float)
        intensity_residual = np.asarray(data["intensity_symmetric_fractional"], dtype=float)
        extent = (
            float(np.min(data["screen_x"])),
            float(np.max(data["screen_x"])),
            float(np.min(data["screen_y"])),
            float(np.max(data["screen_y"])),
        )

    orient = lambda array: np.flip(np.asarray(array), axis=(-2, -1))

    def _repair_projected_axis_for_display(array):
        repaired = np.asarray(array, dtype=float).copy()
        lower = repaired.shape[-2] // 2 - 1
        upper = repaired.shape[-2] // 2
        before = repaired[lower - 1]
        after = repaired[upper + 1]
        for row, weight in ((lower, 1.0 / 3.0), (upper, 2.0 / 3.0)):
            interpolated = (1.0 - weight) * before + weight * after
            valid = np.isfinite(interpolated)
            repaired[row, valid] = interpolated[valid]
        return repaired
    g_values = np.concatenate(
        (pytorch_g[np.isfinite(pytorch_g)], sim5_g[np.isfinite(sim5_g)])
    )
    g_delta = max(float(np.percentile(np.abs(g_values - 1.0), 99.5)), 0.05)
    g_norm = mcolors.TwoSlopeNorm(
        vmin=1.0 - g_delta, vcenter=1.0, vmax=1.0 + g_delta
    )
    intensity_reference = float(np.nanmax(intensity_sim5))
    if not np.isfinite(intensity_reference) or intensity_reference <= 0.0:
        intensity_reference = max(float(np.nanmax(intensity_pytorch)), 1.0)
    normalized_pytorch = intensity_pytorch / intensity_reference
    normalized_sim5 = intensity_sim5 / intensity_reference
    positive = np.concatenate(
        (
            normalized_pytorch[normalized_pytorch > 0.0],
            normalized_sim5[normalized_sim5 > 0.0],
        )
    )
    intensity_low = max(float(np.percentile(positive, 0.5)), 1.0e-6)
    intensity_high = max(float(np.percentile(positive, 99.8)), 1.0)

    def _nice_limit(values, minimum):
        finite = np.abs(np.asarray(values)[np.isfinite(values)])
        target = max(float(np.percentile(finite, 99.0)), float(minimum))
        exponent = np.floor(np.log10(target))
        base = 10.0**exponent
        for factor in (1.0, 2.0, 5.0, 10.0):
            if target <= factor * base:
                return factor * base
        return 10.0 * base

    width = abs(extent[1] - extent[0])
    scale_target = 0.20 * width
    scale_base = 10.0 ** np.floor(np.log10(max(scale_target, 1.0e-12)))
    scale_candidates = np.asarray((1.0, 2.0, 5.0, 10.0)) * scale_base
    valid_scales = scale_candidates[scale_candidates <= scale_target]
    scale_length = float(valid_scales[-1] if valid_scales.size else scale_candidates[0])

    top = (
        orient(pytorch_g),
        orient(sim5_g),
        100.0 * orient(_repair_projected_axis_for_display(g_residual)),
    )
    bottom = (
        orient(normalized_pytorch),
        orient(normalized_sim5),
        100.0 * orient(_repair_projected_axis_for_display(intensity_residual)),
    )
    g_limit = _nice_limit(top[2], 0.01)
    intensity_limit = _nice_limit(bottom[2], 0.1)

    with plt.rc_context(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.72), constrained_layout=True)
        fig.set_constrained_layout_pads(
            w_pad=0.01, h_pad=0.005, wspace=0.01, hspace=0.0
        )

        def _cmap(name):
            cmap = plt.colormaps[name].copy()
            cmap.set_bad("white")
            return cmap

        for column in range(3):
            if column < 2:
                artist = axes[0, column].imshow(
                    top[column], origin="lower", extent=extent,
                    cmap=_cmap("coolwarm"), norm=g_norm, interpolation="nearest",
                )
                label = r"$g=\nu_{\rm obs}/\nu_{\rm em}$"
            else:
                artist = axes[0, column].imshow(
                    top[column], origin="lower", extent=extent,
                    cmap=_cmap("coolwarm"),
                    norm=mcolors.TwoSlopeNorm(vmin=-g_limit, vcenter=0.0, vmax=g_limit),
                    interpolation="nearest",
                )
                label = "Frequency-shift residual [%]"
            axes[0, column].set_title(
                ("This work (analytic)", "SIM5", "Difference")[column], pad=4.0
            )
            colorbar = fig.colorbar(artist, ax=axes[0, column], fraction=0.046, pad=0.018)
            colorbar.set_label(label, fontsize=8.5, labelpad=3)
            colorbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.6)

            if column < 2:
                artist = axes[1, column].imshow(
                    np.ma.masked_less_equal(bottom[column], 0.0),
                    origin="lower", extent=extent, cmap=_cmap("magma"),
                    norm=mcolors.LogNorm(vmin=intensity_low, vmax=intensity_high),
                    interpolation="nearest",
                )
                label = r"$I/I_{\max,\,\rm SIM5}$"
            else:
                artist = axes[1, column].imshow(
                    bottom[column], origin="lower", extent=extent,
                    cmap=_cmap("coolwarm"),
                    norm=mcolors.TwoSlopeNorm(
                        vmin=-intensity_limit, vcenter=0.0, vmax=intensity_limit
                    ),
                    interpolation="nearest",
                )
                label = "Brightness residual [%]"
            colorbar = fig.colorbar(artist, ax=axes[1, column], fraction=0.046, pad=0.018)
            colorbar.set_label(label, fontsize=8.5, labelpad=3)
            colorbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.6)

        for axis in axes.flat:
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.75)
            bar = AnchoredSizeBar(
                axis.transData, scale_length, rf"${scale_length:g}\,r_{{\rm g}}$",
                "lower left", pad=0.2, borderpad=0.25, sep=3, color="black",
                frameon=False, size_vertical=max(0.003 * abs(extent[3] - extent[2]), 1.0e-12),
                fontproperties=fm.FontProperties(size=8.5),
            )
            axis.add_artist(bar)
        axes[0, 0].text(-0.17, 0.5, "Frequency shift", transform=axes[0, 0].transAxes, ha="center", va="center", rotation=90, fontsize=9.5)
        axes[1, 0].text(-0.17, 0.5, "Bolometric brightness", transform=axes[1, 0].transAxes, ha="center", va="center", rotation=90, fontsize=9.5)
        output_path = output_directory / f"{output_prefix}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.01, facecolor="white")
        plt.close(fig)
    return output_path
