"""Geometry resolution for high-level microlensing systems."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..geometry import PlaneGrid, PlaneRegion
from ..lens import LensingDistances, MacroLens, PointMassField

if TYPE_CHECKING:
    from ..sources import PixelatedSource


def _source_grid_from_model(
    source: PixelatedSource,
    distances: LensingDistances,
) -> PlaneGrid:
    """Convert a pixelated source's physical geometry to an angular grid."""

    geometry = source.geometry
    fov_m = (
        float(geometry.shape[0]) * float(geometry.pixel_scale_m[0]),
        float(geometry.shape[1]) * float(geometry.pixel_scale_m[1]),
    )
    fov_uas = distances.source_length_to_uas(fov_m, dtype=torch.float64)
    return PlaneGrid(
        shape=geometry.shape,
        field_of_view_uas=(float(fov_uas[0]), float(fov_uas[1])),
    )


def _direct_star_region(
    stars: PointMassField,
    macro: MacroLens,
    source_region: PlaneRegion,
) -> PlaneRegion:
    """Infer a conservative integration box for directly supplied stars."""

    angle = 2.0 * macro.shear_angle_rad
    gamma_1 = macro.shear * math.cos(angle)
    gamma_2 = macro.shear * math.sin(angle)
    a_xx = 1.0 - macro.convergence - gamma_1
    a_xy = -gamma_2
    a_yy = 1.0 - macro.convergence + gamma_1
    determinant = a_xx * a_yy - a_xy * a_xy
    if abs(determinant) <= 1.0e-12:
        raise ValueError("macro-lens matrix is too close to singular")
    inverse_xx = a_yy / determinant
    inverse_xy = -a_xy / determinant
    inverse_yy = a_xx / determinant

    source_fov_y, source_fov_x = source_region.field_of_view_uas
    source_center_y, source_center_x = source_region.center_uas
    half_source_x = 0.5 * source_fov_x
    half_source_y = 0.5 * source_fov_y
    center_x = inverse_xx * source_center_x + inverse_xy * source_center_y
    center_y = inverse_xy * source_center_x + inverse_yy * source_center_y
    half_x = abs(inverse_xx) * half_source_x + abs(inverse_xy) * half_source_y
    half_y = abs(inverse_xy) * half_source_x + abs(inverse_yy) * half_source_y
    xmin, xmax = center_x - half_x, center_x + half_x
    ymin, ymax = center_y - half_y, center_y + half_y
    if len(stars):
        star_x = stars.x_uas.detach().cpu()
        star_y = stars.y_uas.detach().cpu()
        xmin = min(xmin, float(star_x.min()))
        xmax = max(xmax, float(star_x.max()))
        ymin = min(ymin, float(star_y.min()))
        ymax = max(ymax, float(star_y.max()))
    padding = (
        2.0 * float(stars.einstein_radius_uas.detach().max().cpu())
        if len(stars)
        else 0.0
    )
    if xmax <= xmin:
        xmin -= max(padding, 0.5)
        xmax += max(padding, 0.5)
    if ymax <= ymin:
        ymin -= max(padding, 0.5)
        ymax += max(padding, 0.5)
    return PlaneRegion(
        (ymax - ymin + 2.0 * padding, xmax - xmin + 2.0 * padding),
        (0.5 * (ymin + ymax), 0.5 * (xmin + xmax)),
    )


def _lens_plane_region(
    lens_plane_uas: str | float | tuple[float, float],
) -> PlaneRegion | None:
    """Resolve a centered public lens-plane size or the automatic sentinel."""

    if isinstance(lens_plane_uas, str):
        if lens_plane_uas != "auto":
            raise ValueError("lens_plane_uas must be 'auto', a size, or two sizes")
        return None
    if isinstance(lens_plane_uas, int | float):
        size = float(lens_plane_uas)
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("lens_plane_uas must be positive and finite")
        return PlaneRegion((size, size))
    if len(lens_plane_uas) != 2:
        raise ValueError("lens_plane_uas must contain (height, width)")
    sizes = tuple(float(value) for value in lens_plane_uas)
    if any(not math.isfinite(value) or value <= 0.0 for value in sizes):
        raise ValueError("lens_plane_uas values must be positive and finite")
    return PlaneRegion(sizes)
