"""Reproducible point-mass population builders."""

from __future__ import annotations

import math

import torch

from ..geometry import PlaneRegion
from .mass_functions import MassFunction
from .models import LensingDistances, MacroLens, PointMassField


def _stochastic_deflection_margin_uas(
    macro_lens: MacroLens,
    distances: LensingDistances,
    mass_function: MassFunction,
    light_loss: float,
) -> float:
    """Return the conventional RMS point-mass deflection margin."""

    mean_mass = float(mass_function.mean_mass())
    second_moment = float(mass_function.second_moment())
    if mean_mass <= 0.0 or second_moment <= 0.0:
        raise ValueError("mass-function moments must be positive")
    moment_ratio = second_moment / mean_mass**2
    mean_einstein_radius = float(
        distances.einstein_radius_uas(mean_mass, dtype=torch.float64)
    )
    return mean_einstein_radius * math.sqrt(
        macro_lens.compact_convergence * moment_ratio / light_loss
    )


def rectangular_lens_region(
    macro_lens: MacroLens,
    source_region: PlaneRegion,
    distances: LensingDistances,
    mass_function: MassFunction,
    *,
    light_loss: float = 0.01,
    padding_factor: float = 1.0,
) -> PlaneRegion:
    r"""Construct the conventional light-loss rectangular lens aperture.

    The source rectangle is mapped through the two local macro-lens
    eigenvalues and enlarged by the stochastic point-mass deflection margin

    .. math::

       \theta_E(\langle M\rangle)
       \sqrt{\kappa_*\,\langle M^2\rangle /
       (\epsilon\,\langle M\rangle^2)},

    where ``epsilon`` is ``light_loss``.  This reproduces the rectangular
    field convention used by the manuscript benchmark.  Omitting this term
    and using only the macro preimage of the source box produces a rectangle
    that is much too small.

    The returned region is an axis-aligned bounding box in the caller's
    coordinate frame. Nonzero shear angles are handled by inverting the full
    local macro-lens matrix, so callers do not need to rotate coordinates.
    ``padding_factor`` is an optional additional multiplicative margin. The
    manuscript rectangle uses its default value of one.
    """

    loss = float(light_loss)
    padding = float(padding_factor)
    if not 0.0 < loss < 1.0:
        raise ValueError("light_loss must lie strictly between zero and one")
    if padding < 1.0:
        raise ValueError("padding_factor must be at least one")
    stochastic_margin = _stochastic_deflection_margin_uas(
        macro_lens,
        distances,
        mass_function,
        loss,
    )

    angle = 2.0 * macro_lens.shear_angle_rad
    gamma_1 = macro_lens.shear * math.cos(angle)
    gamma_2 = macro_lens.shear * math.sin(angle)
    a_xx = 1.0 - macro_lens.convergence - gamma_1
    a_xy = -gamma_2
    a_yy = 1.0 - macro_lens.convergence + gamma_1
    determinant = a_xx * a_yy - a_xy * a_xy
    if abs(determinant) <= 1.0e-12:
        raise ValueError("macro-lens matrix is too close to singular")
    inverse_xx = a_yy / determinant
    inverse_xy = -a_xy / determinant
    inverse_yy = a_xx / determinant

    source_fov_y, source_fov_x = source_region.field_of_view_uas
    source_center_y, source_center_x = source_region.center_uas
    half_source_x = 0.5 * source_fov_x + stochastic_margin
    half_source_y = 0.5 * source_fov_y + stochastic_margin
    center_x = inverse_xx * source_center_x + inverse_xy * source_center_y
    center_y = inverse_xy * source_center_x + inverse_yy * source_center_y
    half_x = (
        abs(inverse_xx) * half_source_x
        + abs(inverse_xy) * half_source_y
    )
    half_y = (
        abs(inverse_xy) * half_source_x
        + abs(inverse_yy) * half_source_y
    )
    return PlaneRegion(
        (2.0 * padding * half_y, 2.0 * padding * half_x),
        (center_y, center_x),
    )


def compact_convergence(
    field: PointMassField,
    region: PlaneRegion,
) -> float:
    r"""Return the realized compact convergence of a point-mass field.

    For angular Einstein radius :math:`\theta_E`, one point lens contributes
    :math:`\pi\theta_E^2` in convergence-weighted angular area. The result is
    therefore independent of a particular pixelization.
    """

    area = math.prod(region.field_of_view_uas)
    value = torch.pi * field.einstein_radius_uas.square().sum() / area
    return float(value.detach().cpu())


def sample_uniform_point_masses(
    region: PlaneRegion,
    compact_convergence_value: float,
    distances: LensingDistances,
    mass_function: MassFunction,
    *,
    count: int | None = None,
    velocity_dispersion_uas_per_day: float | tuple[float, float] | None = None,
    velocity_mean_uas_per_day: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PointMassField:
    """Draw a uniform rectangular point-mass population.

    Parameters
    ----------
    region:
        Angular lens-plane region in which centers are sampled uniformly.
        Any desired safety margin belongs in this explicit region.
    compact_convergence_value:
        Target convergence in point masses. If ``count`` is omitted, the
        nearest integer count is chosen from the mass function's analytic mean.
        The finite realization generally differs slightly from this target;
        use :func:`compact_convergence` to measure it.
    distances:
        Lens/source distances used to convert sampled masses to Einstein radii.
    mass_function:
        Any object implementing the public ``MassFunction`` protocol.
    count:
        Optional exact number of objects. This is useful for compact-object or
        injected-planet studies that should not infer a count from convergence.
    velocity_dispersion_uas_per_day:
        Optional independent zero-mean Gaussian dispersion. A scalar applies
        to both Cartesian components. A pair specifies ``(sigma_x, sigma_y)``.
    velocity_mean_uas_per_day:
        Mean Cartesian motion ``(vx, vy)`` added to every sampled object.
    seed, generator:
        Reproducibility controls. Supply at most one. Sampling is performed on
        CPU before the completed tensors are moved, giving deterministic inputs
        across CPU and accelerator runs for a fixed PyTorch version. An
        explicit ``seed`` reproduces the complete masses, positions and
        velocities. If neither argument is supplied, the evolving global CPU
        PyTorch random stream is used, so successive calls draw different
        realizations. A caller-owned ``generator`` is advanced in place.
    """

    target = float(compact_convergence_value)
    if target < 0:
        raise ValueError("compact_convergence_value must be non-negative")
    if seed is not None and generator is not None:
        raise ValueError("supply at most one of seed and generator")
    if generator is None and seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    elif generator is not None and generator.device.type != "cpu":
        raise ValueError("generator must be a CPU generator")

    if count is None:
        if target == 0:
            count = 0
        else:
            mean_radius = distances.einstein_radius_uas(
                mass_function.mean_mass(),
                dtype=torch.float64,
            )
            expected = target * math.prod(region.field_of_view_uas) / (
                math.pi * float(mean_radius.square())
            )
            count = max(1, int(round(expected)))
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    count = int(count)

    masses = mass_function.sample(
        count,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    xmin, xmax, ymin, ymax = region.bounds_uas
    uniform = torch.rand((2, count), generator=generator, dtype=torch.float64)
    x = xmin + (xmax - xmin) * uniform[0]
    y = ymin + (ymax - ymin) * uniform[1]

    mean_velocity = torch.as_tensor(
        velocity_mean_uas_per_day,
        dtype=torch.float64,
    ).reshape(-1)
    if mean_velocity.numel() != 2 or not bool(torch.isfinite(mean_velocity).all()):
        raise ValueError("velocity mean must contain two finite values")
    has_mean_motion = bool(torch.any(mean_velocity != 0.0))
    velocity_x = velocity_y = None
    if velocity_dispersion_uas_per_day is not None or has_mean_motion:
        velocity_x = torch.zeros(count, dtype=torch.float64)
        velocity_y = torch.zeros(count, dtype=torch.float64)
    if velocity_dispersion_uas_per_day is not None:
        dispersion = torch.as_tensor(
            velocity_dispersion_uas_per_day,
            dtype=torch.float64,
        ).reshape(-1)
        if dispersion.numel() == 1:
            dispersion = dispersion.expand(2)
        if dispersion.numel() != 2 or bool(torch.any(dispersion < 0)):
            raise ValueError(
                "velocity dispersion must be non-negative and scalar or length two"
            )
        velocity_x.add_(
            torch.randn(count, generator=generator, dtype=torch.float64)
            * dispersion[0]
        )
        velocity_y.add_(
            torch.randn(count, generator=generator, dtype=torch.float64)
            * dispersion[1]
        )
    if velocity_x is not None:
        velocity_x.add_(mean_velocity[0])
        velocity_y.add_(mean_velocity[1])

    return PointMassField._from_masses(
        x,
        y,
        masses,
        distances,
        velocity_x_uas_per_day=velocity_x,
        velocity_y_uas_per_day=velocity_y,
        device=device,
        dtype=dtype,
    )


def sample_uniform_circular_point_masses(
    radius_uas: float,
    compact_convergence_value: float,
    distances: LensingDistances,
    mass_function: MassFunction,
    *,
    center_uas: tuple[float, float] = (0.0, 0.0),
    count: int | None = None,
    velocity_dispersion_uas_per_day: float | tuple[float, float] | None = None,
    velocity_mean_uas_per_day: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PointMassField:
    """Draw a uniform point-mass population inside a circular aperture.

    The circle is the physical stellar aperture. Numerical scout, full-field,
    and rectangular integration regions may all trace this same realization.
    The target count follows directly from the circular convergence-weighted
    area, while an explicit ``count`` remains available for controlled tests.
    """

    radius = float(radius_uas)
    target = float(compact_convergence_value)
    if radius <= 0.0:
        raise ValueError("radius_uas must be positive")
    if target < 0.0:
        raise ValueError("compact_convergence_value must be non-negative")
    if len(center_uas) != 2:
        raise ValueError("center_uas must contain y and x")
    if seed is not None and generator is not None:
        raise ValueError("supply at most one of seed and generator")
    if generator is None and seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    elif generator is not None and generator.device.type != "cpu":
        raise ValueError("generator must be a CPU generator")

    if count is None:
        if target == 0.0:
            count = 0
        else:
            mean_radius = distances.einstein_radius_uas(
                mass_function.mean_mass(),
                dtype=torch.float64,
            )
            expected = target * radius**2 / float(mean_radius.square())
            count = max(1, int(round(expected)))
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    count = int(count)

    masses = mass_function.sample(
        count,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    uniform = torch.rand((2, count), generator=generator, dtype=torch.float64)
    radial = radius * torch.sqrt(uniform[0])
    azimuth = 2.0 * math.pi * uniform[1]
    center_y, center_x = (float(value) for value in center_uas)
    x = center_x + radial * torch.cos(azimuth)
    y = center_y + radial * torch.sin(azimuth)

    mean_velocity = torch.as_tensor(
        velocity_mean_uas_per_day,
        dtype=torch.float64,
    ).reshape(-1)
    if mean_velocity.numel() != 2 or not bool(torch.isfinite(mean_velocity).all()):
        raise ValueError("velocity mean must contain two finite values")
    has_mean_motion = bool(torch.any(mean_velocity != 0.0))
    velocity_x = velocity_y = None
    if velocity_dispersion_uas_per_day is not None or has_mean_motion:
        velocity_x = torch.zeros(count, dtype=torch.float64)
        velocity_y = torch.zeros(count, dtype=torch.float64)
    if velocity_dispersion_uas_per_day is not None:
        dispersion = torch.as_tensor(
            velocity_dispersion_uas_per_day,
            dtype=torch.float64,
        ).reshape(-1)
        if dispersion.numel() == 1:
            dispersion = dispersion.expand(2)
        if dispersion.numel() != 2 or bool(torch.any(dispersion < 0)):
            raise ValueError(
                "velocity dispersion must be non-negative and scalar or length two"
            )
        velocity_x.add_(
            torch.randn(count, generator=generator, dtype=torch.float64)
            * dispersion[0]
        )
        velocity_y.add_(
            torch.randn(count, generator=generator, dtype=torch.float64)
            * dispersion[1]
        )
    if velocity_x is not None:
        velocity_x.add_(mean_velocity[0])
        velocity_y.add_(mean_velocity[1])

    return PointMassField._from_masses(
        x,
        y,
        masses,
        distances,
        velocity_x_uas_per_day=velocity_x,
        velocity_y_uas_per_day=velocity_y,
        device=device,
        dtype=dtype,
    )
