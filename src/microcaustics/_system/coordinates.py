"""Coordinate-frame and stellar-motion helpers for system realization."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from ..geometry import PlaneRegion
from ..lens import IncompleteKinematicsWarning, PointMassField

if TYPE_CHECKING:
    from ..trajectories import SourceTrajectory


def _warn_incomplete_explicit_motion(stars: PointMassField) -> None:
    """Warn when explicit dynamic velocities omit bulk or differential motion."""

    if len(stars) == 0:
        return
    if not stars.has_motion:
        warnings.warn(
            "Dynamic point-mass field contains no velocities. Stellar "
            "dispersion and bulk motion are omitted. Supply explicit "
            "observer-frame velocity arrays, or use a StellarPopulation with "
            "SkyProjectedKinematics.",
            IncompleteKinematicsWarning,
            stacklevel=4,
        )
        return
    assert stars.velocity_x_uas_per_day is not None
    assert stars.velocity_y_uas_per_day is not None
    velocity = torch.stack(
        (stars.velocity_x_uas_per_day, stars.velocity_y_uas_per_day),
        dim=1,
    )
    mean = velocity.mean(dim=0)
    centered = velocity - mean
    floating = torch.finfo(velocity.dtype)
    scale = max(float(velocity.abs().max()), floating.tiny)
    tolerance = 32.0 * floating.eps * scale
    missing = []
    if float(mean.abs().max()) <= tolerance:
        missing.append("bulk motion")
    if len(stars) < 2 or float(centered.abs().max()) <= tolerance:
        missing.append("stellar velocity dispersion")
    if missing:
        warnings.warn(
            "Dynamic explicit point-mass velocities omit "
            + " and ".join(missing)
            + ". Explicit arrays are interpreted as final observer-frame "
            "velocities. Include projected CMB, lens, and source motion in "
            "their common drift, plus independent stellar motion, or use a "
            "StellarPopulation with SkyProjectedKinematics.",
            IncompleteKinematicsWarning,
            stacklevel=4,
        )


def _stellar_motion_metadata(stars: PointMassField) -> dict[str, object]:
    """Summarize the realized observer-frame point-lens velocities."""

    if not stars.has_motion or len(stars) == 0:
        return {
            "has_motion": False,
            "coordinate_basis": "realization x/y",
        }
    assert stars.velocity_x_uas_per_day is not None
    assert stars.velocity_y_uas_per_day is not None
    velocity = torch.stack(
        (stars.velocity_x_uas_per_day, stars.velocity_y_uas_per_day),
        dim=1,
    )
    mean = velocity.mean(dim=0)
    centered = velocity - mean
    component_rms = torch.sqrt(torch.mean(centered.square(), dim=0))
    return {
        "has_motion": True,
        "coordinate_basis": "realization x/y",
        "mean_velocity_uas_per_day": [float(value) for value in mean],
        "component_rms_uas_per_day": [
            float(value) for value in component_rms
        ],
    }


def _rotate_cartesian_components(
    x: torch.Tensor,
    y: torch.Tensor,
    angle_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express Cartesian components in axes rotated by ``angle_deg``.

    This is a coordinate-basis change, not an interpolation or a rotation of
    a materialized image. A positive angle maps sky-frame components to the
    local frame through an active rotation by the negative of that angle.
    """

    angle = torch.as_tensor(
        math.radians(float(angle_deg)),
        device=x.device,
        dtype=x.dtype,
    )
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def _rotate_point_mass_field(
    stars: PointMassField,
    angle_deg: float,
) -> PointMassField:
    """Return the same physical point lenses in a rotated coordinate basis."""

    x, y = _rotate_cartesian_components(stars.x_uas, stars.y_uas, angle_deg)
    velocity_x = velocity_y = None
    if stars.has_motion:
        assert stars.velocity_x_uas_per_day is not None
        assert stars.velocity_y_uas_per_day is not None
        velocity_x, velocity_y = _rotate_cartesian_components(
            stars.velocity_x_uas_per_day,
            stars.velocity_y_uas_per_day,
            angle_deg,
        )
    return PointMassField._from_einstein_radii(
        x,
        y,
        stars.einstein_radius_uas,
        mass_solar=stars.mass_solar,
        velocity_x_uas_per_day=velocity_x,
        velocity_y_uas_per_day=velocity_y,
    )


@dataclass(frozen=True)
class _RotatedTrajectory:
    """Internal coordinate-basis view of an arbitrary source trajectory."""

    trajectory: SourceTrajectory
    angle_deg: float

    def position_uas(
        self,
        times_days,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        positions = self.trajectory.position_uas(
            times_days,
            device=device,
            dtype=dtype,
        )
        x, y = _rotate_cartesian_components(
            positions[..., 0], positions[..., 1], self.angle_deg
        )
        return torch.stack((x, y), dim=-1)


def _source_in_rotated_frame(source, angle_deg: float):
    """Return a physical source model expressed in a rotated basis.

    Built-in physical models expose their sky position angle explicitly, so
    changing coordinates only requires subtracting the basis angle before
    pixelization. Gaussian centers are transformed in the same operation.
    Materialized pixel arrays are deliberately not accepted because rotating
    those would introduce an interpolation into the scientific calculation.
    """

    updates: dict[str, object] = {}
    if hasattr(source, "position_angle_deg"):
        updates["position_angle_deg"] = float(source.position_angle_deg) - float(
            angle_deg
        )
    elif hasattr(source, "position_angle_rad"):
        updates["position_angle_rad"] = float(source.position_angle_rad) - math.radians(
            float(angle_deg)
        )
    else:
        raise TypeError(
            "automatic shear-frame alignment requires a physical source model "
            "with position_angle_deg or position_angle_rad"
        )
    for name in ("center_m", "center_uas"):
        if getattr(source, name, None) is not None:
            center = torch.as_tensor(getattr(source, name), dtype=torch.float64)
            x, y = _rotate_cartesian_components(center[0], center[1], angle_deg)
            updates[name] = (float(x), float(y))
    return replace(source, **updates)


def _rotate_region_center(region: PlaneRegion, angle_deg: float) -> PlaneRegion:
    """Rotate a region center while preserving a circular aperture's size."""

    center = torch.as_tensor(
        (region.center_uas[1], region.center_uas[0]), dtype=torch.float64
    )
    x, y = _rotate_cartesian_components(center[0], center[1], angle_deg)
    return PlaneRegion(region.field_of_view_uas, (float(y), float(x)))
