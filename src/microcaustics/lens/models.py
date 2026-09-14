"""Validated, device-independent lens model containers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

_GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
_SPEED_OF_LIGHT_SI = 299_792_458.0
_SOLAR_MASS_KG = 1.988409870698051e30
_RADIANS_TO_MICROARCSECONDS = 180.0 / torch.pi * 3600.0 * 1.0e6
_MICROARCSECONDS_TO_RADIANS = 1.0 / _RADIANS_TO_MICROARCSECONDS
_MEGAPARSEC_M = 3.085677581491367e22
_SPEED_OF_LIGHT_KM_S = 299_792.458


class _SourceCoordinates:
    """Shared physical/angular conversions for source and lens geometries."""

    def source_length_to_uas(self, length_m, *, device=None, dtype=torch.float32):
        """Convert proper source-plane meters to microarcseconds."""
        return (
            torch.as_tensor(length_m, device=device, dtype=dtype)
            / self.source_m
            * float(_RADIANS_TO_MICROARCSECONDS)
        )

    def uas_to_source_length(self, angle_uas, *, device=None, dtype=torch.float32):
        """Convert source-plane microarcseconds to proper meters."""
        return (
            torch.as_tensor(angle_uas, device=device, dtype=dtype)
            * float(_MICROARCSECONDS_TO_RADIANS)
            * self.source_m
        )


@dataclass(frozen=True)
class _SourceDistances(_SourceCoordinates):
    """Source-only angular geometry used internally by standalone sources."""

    source_m: float
    source_redshift: float


def _source_distances_from_redshift(source_redshift, *, H0, Om0, device, dtype):
    """Resolve a flat cosmology without inventing an unused lens plane."""
    redshift = float(source_redshift)
    if not math.isfinite(redshift) or redshift <= 0:
        raise ValueError("source_redshift must be finite and positive")
    if not math.isfinite(H0) or H0 <= 0 or not math.isfinite(Om0) or not 0 <= Om0 <= 1:
        raise ValueError("H0 must be positive and Om0 must lie in [0, 1]")
    z = torch.linspace(0, redshift, 16_385, device=device, dtype=dtype)
    inverse_expansion = torch.rsqrt(Om0 * (1 + z).pow(3) + (1 - Om0))
    integral = (
        0.5 * redshift / 16_384 * (inverse_expansion[:-1] + inverse_expansion[1:])
    ).cumsum(0)[-1]
    return _SourceDistances(
        source_m=float(integral)
        * _SPEED_OF_LIGHT_KM_S
        / H0
        * _MEGAPARSEC_M
        / (1 + redshift),
        source_redshift=redshift,
    )


@dataclass(frozen=True)
class LensingDistances(_SourceCoordinates):
    """Angular-diameter distances required by a single lens plane.

    All three values are in meters. Keeping this lightweight container in the
    core package allows callers to use any cosmology implementation. The
    :meth:`from_redshifts` helper evaluates a flat cosmology directly with
    PyTorch, while still accepting an optional Astropy cosmology.
    """

    lens_m: float
    source_m: float
    lens_to_source_m: float
    lens_redshift: float | None = None
    source_redshift: float | None = None

    def __post_init__(self) -> None:
        if self.lens_m <= 0 or self.source_m <= 0 or self.lens_to_source_m <= 0:
            raise ValueError("all angular-diameter distances must be positive")
        if self.lens_redshift is not None and self.lens_redshift < 0:
            raise ValueError("lens_redshift must be non-negative")
        if self.source_redshift is not None:
            if self.lens_redshift is None:
                raise ValueError(
                    "source_redshift requires lens_redshift so their ordering is known"
                )
            if self.source_redshift <= self.lens_redshift:
                raise ValueError("source_redshift must exceed lens_redshift")

    @classmethod
    def from_redshifts(
        cls,
        lens_redshift: float,
        source_redshift: float,
        *,
        cosmology=None,
        H0: float = 67.66,
        Om0: float = 0.30966,
        device: torch.device | str | None = "auto",
        dtype: torch.dtype = torch.float32,
        integration_steps: int = 16_384,
    ) -> LensingDistances:
        """Calculate angular-diameter distances from two redshifts.

        With no external ``cosmology``, a flat matter-plus-cosmological-
        constant expansion history is integrated directly with PyTorch. The
        default calculation uses float32, like the rest of the production
        package, and can run on the selected accelerator. ``H0`` is expressed
        in km s^-1 Mpc^-1. ``Om0`` is the present matter density fraction.

        Passing an Astropy-compatible cosmology remains supported for
        arbitrary expansion histories and independent validation.
        """

        if lens_redshift < 0 or source_redshift <= lens_redshift:
            raise ValueError("source_redshift must exceed a non-negative lens_redshift")
        if cosmology is None:
            if not math.isfinite(float(H0)) or float(H0) <= 0.0:
                raise ValueError("H0 must be positive and finite")
            if not math.isfinite(float(Om0)) or not 0.0 <= float(Om0) <= 1.0:
                raise ValueError("Om0 must lie in [0, 1]")
            if int(integration_steps) < 256:
                raise ValueError("integration_steps must be at least 256")
            if not torch.empty((), dtype=dtype).is_floating_point():
                raise TypeError("cosmological distance dtype must be floating point")
            if device is None or str(device) == "auto":
                if torch.cuda.is_available():
                    resolved_device = torch.device("cuda")
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    resolved_device = torch.device("mps")
                else:
                    resolved_device = torch.device("cpu")
            else:
                resolved_device = torch.device(device)
            redshifts = torch.linspace(
                0.0,
                float(source_redshift),
                int(integration_steps) + 1,
                device=resolved_device,
                dtype=dtype,
            )
            inverse_expansion = torch.rsqrt(
                float(Om0) * (1.0 + redshifts).pow(3) + (1.0 - float(Om0))
            )
            dz = float(source_redshift) / int(integration_steps)
            increments = 0.5 * dz * (inverse_expansion[:-1] + inverse_expansion[1:])
            cumulative = torch.cat(
                (
                    torch.zeros(1, device=resolved_device, dtype=dtype),
                    increments.cumsum(0),
                )
            )
            lens_position = (
                float(lens_redshift) / float(source_redshift) * int(integration_steps)
            )
            lower = min(int(math.floor(lens_position)), int(integration_steps) - 1)
            fraction = lens_position - lower
            lens_integral = cumulative[lower] + fraction * (
                cumulative[lower + 1] - cumulative[lower]
            )
            source_integral = cumulative[-1]
            hubble_distance_m = _SPEED_OF_LIGHT_KM_S / float(H0) * _MEGAPARSEC_M
            lens_comoving_m = float(lens_integral) * hubble_distance_m
            source_comoving_m = float(source_integral) * hubble_distance_m
            return cls(
                lens_m=lens_comoving_m / (1.0 + float(lens_redshift)),
                source_m=source_comoving_m / (1.0 + float(source_redshift)),
                lens_to_source_m=(source_comoving_m - lens_comoving_m)
                / (1.0 + float(source_redshift)),
                lens_redshift=float(lens_redshift),
                source_redshift=float(source_redshift),
            )

        try:
            import astropy.units as units
        except ImportError as exc:
            raise ImportError(
                "an external Astropy cosmology requires the 'science' extra"
            ) from exc
        try:
            lens_to_source = cosmology.angular_diameter_distance(
                lens_redshift,
                source_redshift,
            )
        except TypeError:
            # Astropy < 7 exposed the two-redshift calculation under this
            # dedicated name. Keep the supported older science extra working
            # while avoiding its deprecation on current Astropy releases.
            lens_to_source = cosmology.angular_diameter_distance_z1z2(
                lens_redshift,
                source_redshift,
            )
        return cls(
            lens_m=float(
                cosmology.angular_diameter_distance(lens_redshift).to_value(units.m)
            ),
            source_m=float(
                cosmology.angular_diameter_distance(source_redshift).to_value(units.m)
            ),
            lens_to_source_m=float(lens_to_source.to_value(units.m)),
            lens_redshift=float(lens_redshift),
            source_redshift=float(source_redshift),
        )

    def einstein_radius_uas(
        self,
        mass_solar,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert point-lens masses in solar units to angular Einstein radii."""

        mass = torch.as_tensor(mass_solar, device=device, dtype=dtype)
        if bool(torch.any(mass <= 0)):
            raise ValueError("point-lens masses must be positive")
        distance_factor = self.lens_to_source_m / (self.lens_m * self.source_m)
        radius_rad = torch.sqrt(
            (4.0 * _GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG / _SPEED_OF_LIGHT_SI**2)
            * mass
            * distance_factor
        )
        return radius_rad * float(_RADIANS_TO_MICROARCSECONDS)


def _distances_from_arguments(
    *,
    lens_redshift: float | None,
    source_redshift: float | None,
    distances: LensingDistances | None,
    H0: float,
    Om0: float,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> LensingDistances:
    if distances is not None:
        if lens_redshift is not None or source_redshift is not None:
            raise ValueError("supply redshifts or distances, not both")
        return distances
    if lens_redshift is None or source_redshift is None:
        raise ValueError("lens_redshift and source_redshift are required")
    return LensingDistances.from_redshifts(
        lens_redshift,
        source_redshift,
        H0=H0,
        Om0=Om0,
        device=device,
        dtype=dtype,
    )


def einstein_radius_uas(
    mean_mass_solar,
    *,
    lens_redshift: float | None = None,
    source_redshift: float | None = None,
    distances: LensingDistances | None = None,
    H0: float = 67.66,
    Om0: float = 0.30966,
    device: torch.device | str | None = "auto",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the angular Einstein radius for masses in solar units.

    Redshifts use the same default flat cosmology as
    :class:`LensingDistances`. Advanced callers may instead pass an existing
    distance object.
    """

    resolved = _distances_from_arguments(
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        distances=distances,
        H0=H0,
        Om0=Om0,
        device=device,
        dtype=dtype,
    )
    resolved_device = None if device is None or str(device) == "auto" else device
    return resolved.einstein_radius_uas(
        mean_mass_solar,
        device=resolved_device,
        dtype=dtype,
    )


def einstein_units_to_uas(
    values,
    *,
    mean_mass_solar,
    lens_redshift: float | None = None,
    source_redshift: float | None = None,
    distances: LensingDistances | None = None,
    H0: float = 67.66,
    Om0: float = 0.30966,
    device: torch.device | str | None = "auto",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert coordinates in mean-microlens Einstein units to microarcseconds."""

    radius = einstein_radius_uas(
        mean_mass_solar,
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        distances=distances,
        H0=H0,
        Om0=Om0,
        device=device,
        dtype=dtype,
    )
    return torch.as_tensor(values, device=radius.device, dtype=radius.dtype) * radius


def uas_to_einstein_units(
    values,
    *,
    mean_mass_solar,
    lens_redshift: float | None = None,
    source_redshift: float | None = None,
    distances: LensingDistances | None = None,
    H0: float = 67.66,
    Om0: float = 0.30966,
    device: torch.device | str | None = "auto",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert angular coordinates to mean-microlens Einstein units."""

    radius = einstein_radius_uas(
        mean_mass_solar,
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        distances=distances,
        H0=H0,
        Om0=Om0,
        device=device,
        dtype=dtype,
    )
    return torch.as_tensor(values, device=radius.device, dtype=radius.dtype) / radius


@dataclass(frozen=True)
class MacroLens:
    """Local convergence and shear at one macroimage.

    Parameters are dimensionless except ``shear_angle_deg``. The smooth matter
    fraction is the fraction of total convergence represented by a continuous
    sheet rather than the supplied point-mass population.
    """

    convergence: float
    shear: float
    shear_angle_deg: float = 0.0
    smooth_matter_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "convergence", float(self.convergence))
        object.__setattr__(self, "shear", float(self.shear))
        object.__setattr__(self, "shear_angle_deg", float(self.shear_angle_deg))
        object.__setattr__(
            self,
            "smooth_matter_fraction",
            float(self.smooth_matter_fraction),
        )
        if self.convergence < 0:
            raise ValueError("convergence must be non-negative")
        if self.shear < 0:
            raise ValueError("shear must be non-negative")
        if not math.isfinite(self.shear_angle_deg):
            raise ValueError("shear_angle_deg must be finite")
        if not 0.0 <= self.smooth_matter_fraction <= 1.0:
            raise ValueError("smooth_matter_fraction must lie in [0, 1]")

    @property
    def shear_angle_rad(self) -> float:
        """Return the shear position angle in radians for numerical kernels."""

        return math.radians(self.shear_angle_deg)

    @property
    def smooth_convergence(self) -> float:
        """Convergence assigned to the continuous matter sheet."""

        return self.convergence * self.smooth_matter_fraction

    @property
    def compact_convergence(self) -> float:
        """Convergence assigned to compact objects."""

        return self.convergence - self.smooth_convergence


@dataclass(frozen=True, init=False)
class PointMassField:
    """Point lenses specified by physical masses and angular positions.

    Public construction uses ``mass_solar``. Positions are in microarcseconds
    and optional velocities are in microarcseconds per day. A
    :class:`~microcaustics.MicrolensingSystem` converts the masses to angular
    Einstein radii from its cosmological distances before ray tracing.

    Package internals keep the resolved angular strengths separate from this
    physical constructor.
    """

    x_uas: torch.Tensor
    y_uas: torch.Tensor
    mass_solar: torch.Tensor | None
    velocity_x_uas_per_day: torch.Tensor | None = None
    velocity_y_uas_per_day: torch.Tensor | None = None
    _einstein_radius_uas: torch.Tensor | None = field(default=None, repr=False)

    def __init__(
        self,
        x_uas,
        y_uas,
        mass_solar,
        *,
        velocity_x_uas_per_day=None,
        velocity_y_uas_per_day=None,
    ) -> None:
        object.__setattr__(self, "x_uas", x_uas)
        object.__setattr__(self, "y_uas", y_uas)
        object.__setattr__(self, "mass_solar", mass_solar)
        object.__setattr__(self, "velocity_x_uas_per_day", velocity_x_uas_per_day)
        object.__setattr__(self, "velocity_y_uas_per_day", velocity_y_uas_per_day)
        object.__setattr__(self, "_einstein_radius_uas", None)
        self.__post_init__()

    @classmethod
    def _from_einstein_radii(
        cls,
        x_uas,
        y_uas,
        einstein_radius_uas,
        *,
        mass_solar=None,
        velocity_x_uas_per_day=None,
        velocity_y_uas_per_day=None,
    ) -> PointMassField:
        """Construct an already resolved field for package internals."""

        instance = object.__new__(cls)
        object.__setattr__(instance, "x_uas", x_uas)
        object.__setattr__(instance, "y_uas", y_uas)
        object.__setattr__(instance, "mass_solar", mass_solar)
        object.__setattr__(instance, "velocity_x_uas_per_day", velocity_x_uas_per_day)
        object.__setattr__(instance, "velocity_y_uas_per_day", velocity_y_uas_per_day)
        object.__setattr__(instance, "_einstein_radius_uas", einstein_radius_uas)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        x = torch.as_tensor(self.x_uas)
        if not x.is_floating_point():
            x = x.to(torch.float32)
        y = torch.as_tensor(self.y_uas, device=x.device, dtype=x.dtype)
        if x.ndim != 1 or y.shape != x.shape:
            raise ValueError("point-lens positions must be 1D arrays of equal length")
        mass = self.mass_solar
        if mass is not None:
            mass = torch.as_tensor(mass, device=x.device, dtype=x.dtype)
            if mass.shape != x.shape or bool(torch.any(mass <= 0)):
                raise ValueError("mass_solar must be positive and match the positions")
        radius = self._einstein_radius_uas
        if radius is not None:
            radius = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
            if radius.shape != x.shape or bool(torch.any(radius <= 0)):
                raise ValueError(
                    "resolved Einstein radii must be positive and match the positions"
                )
        if mass is None and radius is None and x.numel() != 0:
            raise ValueError("mass_solar is required for a non-empty point-mass field")
        vx, vy = self.velocity_x_uas_per_day, self.velocity_y_uas_per_day
        if (vx is None) != (vy is None):
            raise ValueError("both velocity components must be supplied together")
        if vx is not None:
            vx = torch.as_tensor(vx, device=x.device, dtype=x.dtype)
            vy = torch.as_tensor(vy, device=x.device, dtype=x.dtype)
            if vx.shape != x.shape or vy.shape != x.shape:
                raise ValueError("velocity arrays must match the position shape")
        object.__setattr__(self, "x_uas", x)
        object.__setattr__(self, "y_uas", y)
        object.__setattr__(self, "_einstein_radius_uas", radius)
        object.__setattr__(self, "velocity_x_uas_per_day", vx)
        object.__setattr__(self, "velocity_y_uas_per_day", vy)
        object.__setattr__(self, "mass_solar", mass)

    @property
    def einstein_radius_uas(self) -> torch.Tensor:
        """Return resolved angular Einstein radii.

        Directly constructed physical fields are resolved automatically by a
        high-level system. Access before that resolution is an error because
        the answer depends on lens and source distances.
        """

        if self._einstein_radius_uas is None:
            raise RuntimeError(
                "Einstein radii have not been resolved. Add this field to a "
                "MicrolensingSystem with lens and source redshifts first"
            )
        return self._einstein_radius_uas

    def resolve(self, distances: LensingDistances) -> PointMassField:
        """Return a numerical field with Einstein radii for ``distances``."""

        if self._einstein_radius_uas is not None:
            return self
        assert self.mass_solar is not None
        radius = distances.einstein_radius_uas(
            self.mass_solar,
            device=self.x_uas.device,
            dtype=self.x_uas.dtype,
        )
        return PointMassField._from_einstein_radii(
            self.x_uas,
            self.y_uas,
            radius,
            mass_solar=self.mass_solar,
            velocity_x_uas_per_day=self.velocity_x_uas_per_day,
            velocity_y_uas_per_day=self.velocity_y_uas_per_day,
        )

    @classmethod
    def _from_masses(
        cls,
        x_uas,
        y_uas,
        mass_solar,
        distances: LensingDistances,
        *,
        velocity_x_uas_per_day=None,
        velocity_y_uas_per_day=None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PointMassField:
        """Build a resolved field for internal population samplers."""

        mass = torch.as_tensor(mass_solar, device=device, dtype=dtype)
        return cls._from_einstein_radii(
            torch.as_tensor(x_uas, device=device, dtype=dtype),
            torch.as_tensor(y_uas, device=device, dtype=dtype),
            distances.einstein_radius_uas(mass, device=device, dtype=dtype),
            mass_solar=mass,
            velocity_x_uas_per_day=velocity_x_uas_per_day,
            velocity_y_uas_per_day=velocity_y_uas_per_day,
        )

    @classmethod
    def sample_uniform(
        cls,
        region,
        macro_lens: MacroLens,
        distances: LensingDistances,
        mass_function,
        **kwargs,
    ) -> PointMassField:
        """Draw a uniform population matching the macro lens's compact matter.

        This convenience constructor delegates to
        :func:`microcaustics.lens.sample_uniform_point_masses`. Direct array
        construction remains available when locations or masses are supplied
        by an external population model.
        """

        from .populations import sample_uniform_point_masses

        return sample_uniform_point_masses(
            region,
            macro_lens.compact_convergence,
            distances,
            mass_function,
            **kwargs,
        )

    def __len__(self) -> int:
        return int(self.x_uas.numel())

    @property
    def has_motion(self) -> bool:
        """Whether explicit point-lens velocities are available."""

        return self.velocity_x_uas_per_day is not None

    def at_time(self, time_days: float | torch.Tensor) -> PointMassField:
        """Return the point-mass field after linear motion for ``time_days``."""

        if not self.has_motion:
            return self
        time = torch.as_tensor(
            time_days, device=self.x_uas.device, dtype=self.x_uas.dtype
        )
        if time.numel() != 1:
            raise ValueError("at_time expects one scalar time")
        assert self.velocity_x_uas_per_day is not None
        assert self.velocity_y_uas_per_day is not None
        moved_x = self.x_uas + time * self.velocity_x_uas_per_day
        moved_y = self.y_uas + time * self.velocity_y_uas_per_day
        # Motion changes only the two position arrays. Re-running ``__post_init__``
        # here would validate the same masses, Einstein radii, and velocities at
        # every epoch; on CUDA, each positivity check would also force a device
        # synchronization. The parent field has already established all shape,
        # dtype, device, and positivity invariants, and the two tensor operations
        # above preserve the position invariants.
        instance = object.__new__(PointMassField)
        object.__setattr__(instance, "x_uas", moved_x)
        object.__setattr__(instance, "y_uas", moved_y)
        object.__setattr__(instance, "mass_solar", self.mass_solar)
        object.__setattr__(
            instance,
            "velocity_x_uas_per_day",
            self.velocity_x_uas_per_day,
        )
        object.__setattr__(
            instance,
            "velocity_y_uas_per_day",
            self.velocity_y_uas_per_day,
        )
        object.__setattr__(
            instance,
            "_einstein_radius_uas",
            self._einstein_radius_uas,
        )
        return instance

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PointMassField:
        """Return a copy on another device or with another floating dtype."""

        kwargs = {"device": device, "dtype": dtype}
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        mass = None if self.mass_solar is None else self.mass_solar.to(**kwargs)
        velocity_x = (
            None
            if self.velocity_x_uas_per_day is None
            else self.velocity_x_uas_per_day.to(**kwargs)
        )
        velocity_y = (
            None
            if self.velocity_y_uas_per_day is None
            else self.velocity_y_uas_per_day.to(**kwargs)
        )
        if self._einstein_radius_uas is None:
            return PointMassField(
                self.x_uas.to(**kwargs),
                self.y_uas.to(**kwargs),
                mass,
                velocity_x_uas_per_day=velocity_x,
                velocity_y_uas_per_day=velocity_y,
            )
        return PointMassField._from_einstein_radii(
            self.x_uas.to(**kwargs),
            self.y_uas.to(**kwargs),
            self._einstein_radius_uas.to(**kwargs),
            mass_solar=mass,
            velocity_x_uas_per_day=velocity_x,
            velocity_y_uas_per_day=velocity_y,
        )
