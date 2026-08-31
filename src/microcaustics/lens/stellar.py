"""Physical stellar-population specifications and circular apertures."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Protocol, runtime_checkable

from ..geometry import PlaneRegion
from .mass_functions import MassFunction, salpeter_mass_function
from .models import LensingDistances, MacroLens, PointMassField
from .populations import (
    _stochastic_deflection_margin_uas,
    rectangular_lens_region,
    sample_uniform_circular_point_masses,
)

_SECONDS_PER_DAY = 86_400.0
_RADIANS_TO_MICROARCSECONDS = 180.0 / math.pi * 3600.0 * 1.0e6


@runtime_checkable
class StellarKinematics(Protocol):
    """Physical prescription for independent stellar velocities."""

    def component_dispersion_uas_per_day(
        self,
        distances: LensingDistances,
    ) -> float:
        """Return one Cartesian observer-frame angular dispersion."""

        ...


@dataclass(frozen=True)
class StaticKinematics:
    """A stationary stellar realization."""

    def component_dispersion_uas_per_day(
        self,
        distances: LensingDistances,
    ) -> float:
        """Return zero angular dispersion."""

        del distances
        return 0.0


@dataclass(frozen=True)
class IsotropicKinematics:
    """An isotropic Gaussian stellar velocity distribution.

    ``dispersion_km_s`` is the one-dimensional proper velocity dispersion in
    the lens rest frame. Observer-frame angular motion includes cosmological
    time dilation when the lens redshift is available from
    :meth:`LensingDistances.from_redshifts`.
    """

    dispersion_km_s: float
    bulk_velocity_km_s: tuple[float, float] = (0.0, 0.0)
    lens_redshift: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.dispersion_km_s)):
            raise ValueError("dispersion_km_s must be finite")
        if self.dispersion_km_s < 0.0:
            raise ValueError("dispersion_km_s must be non-negative")
        if len(self.bulk_velocity_km_s) != 2 or any(
            not math.isfinite(float(value)) for value in self.bulk_velocity_km_s
        ):
            raise ValueError("bulk_velocity_km_s must contain two finite values")
        if self.lens_redshift is not None and self.lens_redshift < 0.0:
            raise ValueError("lens_redshift must be non-negative")

    def component_dispersion_uas_per_day(
        self,
        distances: LensingDistances,
    ) -> float:
        """Convert the proper velocity dispersion to observer angular units."""

        redshift = (
            distances.lens_redshift
            if self.lens_redshift is None
            else self.lens_redshift
        )
        if redshift is None:
            raise ValueError(
                "IsotropicKinematics requires a lens redshift. Construct "
                "distances with LensingDistances.from_redshifts or supply "
                "lens_redshift explicitly"
            )
        velocity_m_s = float(self.dispersion_km_s) * 1_000.0
        radians_per_second = velocity_m_s / float(distances.lens_m)
        return (
            radians_per_second
            * _RADIANS_TO_MICROARCSECONDS
            * _SECONDS_PER_DAY
            / (1.0 + float(redshift))
        )

    def mean_velocity_uas_per_day(
        self,
        distances: LensingDistances,
    ) -> tuple[float, float]:
        """Convert the proper bulk velocity to observer angular units."""

        redshift = (
            distances.lens_redshift
            if self.lens_redshift is None
            else self.lens_redshift
        )
        if redshift is None:
            raise ValueError(
                "IsotropicKinematics requires a lens redshift. Construct "
                "distances with LensingDistances.from_redshifts or supply "
                "lens_redshift explicitly"
            )
        conversion = (
            1_000.0
            / float(distances.lens_m)
            * _RADIANS_TO_MICROARCSECONDS
            * _SECONDS_PER_DAY
            / (1.0 + float(redshift))
        )
        return tuple(float(value) * conversion for value in self.bulk_velocity_km_s)


@dataclass(frozen=True)
class SkyProjectedKinematics:
    """Observer, lens, source, and stellar transverse velocities.

    Cartesian velocity pairs follow the local east/north axes, equivalent to
    increasing ICRS right ascension and declination. Lens and source peculiar
    velocities are proper velocities in their respective rest frames. The
    CMB dipole is projected at ``ra_deg`` and ``dec_deg`` and included by
    default. Set ``peculiar_velocity_dispersion_km_s`` to draw reproducible
    lens and source peculiar velocities after the system supplies redshifts.
    """

    ra_deg: float
    dec_deg: float
    stellar_dispersion_km_s: float = 170.0
    lens_peculiar_velocity_km_s: tuple[float, float] | None = None
    source_peculiar_velocity_km_s: tuple[float, float] | None = None
    peculiar_velocity_dispersion_km_s: float | None = None
    omega_matter: float = 0.3
    omega_lambda: float = 0.7
    seed: int | None = None
    include_cmb_dipole: bool = True
    cmb_speed_km_s: float = 369.82
    cmb_galactic_longitude_deg: float = 264.021
    cmb_galactic_latitude_deg: float = 48.253
    lens_redshift: float | None = None
    source_redshift: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.ra_deg)) or not 0.0 <= self.ra_deg < 360.0:
            raise ValueError("ra_deg must lie in [0, 360)")
        if not math.isfinite(float(self.dec_deg)) or not -90.0 <= self.dec_deg <= 90.0:
            raise ValueError("dec_deg must lie in [-90, 90]")
        for name in (
            "stellar_dispersion_km_s",
            "cmb_speed_km_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        velocity_names = (
            "lens_peculiar_velocity_km_s",
            "source_peculiar_velocity_km_s",
        )
        if self.peculiar_velocity_dispersion_km_s is None and all(
            getattr(self, name) is None for name in velocity_names
        ):
            object.__setattr__(self, "peculiar_velocity_dispersion_km_s", 235.0)
        sampled = self.peculiar_velocity_dispersion_km_s is not None
        if sampled:
            if any(getattr(self, name) is not None for name in velocity_names):
                raise ValueError(
                    "sampled sky kinematics cannot also supply explicit peculiar velocities"
                )
            dispersion = float(self.peculiar_velocity_dispersion_km_s)
            if not math.isfinite(dispersion) or dispersion < 0.0:
                raise ValueError(
                    "peculiar_velocity_dispersion_km_s must be finite and non-negative"
                )
            if self.seed is None:
                import torch

                object.__setattr__(self, "seed", int(torch.seed()))
        else:
            for name in velocity_names:
                values = getattr(self, name)
                if values is None or len(values) != 2 or any(
                    not math.isfinite(float(v)) for v in values
                ):
                    raise ValueError(f"{name} must contain two finite values")
        if not 0.0 <= self.omega_matter <= 1.0 or not 0.0 <= self.omega_lambda <= 2.0:
            raise ValueError("omega_matter and omega_lambda are outside supported bounds")
        for name in ("lens_redshift", "source_redshift"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @staticmethod
    def _growth_rate(redshift: float, omega_matter: float, omega_lambda: float) -> float:
        expansion2 = (
            omega_matter * (1.0 + redshift) ** 3
            + (1.0 - omega_matter - omega_lambda) * (1.0 + redshift) ** 2
            + omega_lambda
        )
        matter = omega_matter * (1.0 + redshift) ** 3 / expansion2
        dark_energy = omega_lambda / expansion2
        return matter ** (4.0 / 7.0) + dark_energy * (1.0 + matter / 2.0) / 70.0

    def _peculiar_velocities(
        self,
        distances: LensingDistances,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.peculiar_velocity_dispersion_km_s is None:
            assert self.lens_peculiar_velocity_km_s is not None
            assert self.source_peculiar_velocity_km_s is not None
            return (
                self.lens_peculiar_velocity_km_s,
                self.source_peculiar_velocity_km_s,
            )
        lens_redshift, source_redshift = self._redshifts(distances)
        import torch

        generator = torch.Generator(device="cpu")
        assert self.seed is not None
        generator.manual_seed(int(self.seed))
        f0 = self._growth_rate(0.0, self.omega_matter, self.omega_lambda)
        lens_sigma = (
            self.peculiar_velocity_dispersion_km_s
            / math.sqrt(1.0 + lens_redshift)
            * self._growth_rate(lens_redshift, self.omega_matter, self.omega_lambda)
            / f0
        )
        source_sigma = (
            self.peculiar_velocity_dispersion_km_s
            / math.sqrt(1.0 + source_redshift)
            * self._growth_rate(source_redshift, self.omega_matter, self.omega_lambda)
            / f0
        )
        lens = tuple(
            float(v)
            for v in torch.randn(2, generator=generator, dtype=torch.float64)
            * lens_sigma
        )
        source = tuple(
            float(v)
            for v in torch.randn(2, generator=generator, dtype=torch.float64)
            * source_sigma
        )
        return lens, source

    def _redshifts(self, distances: LensingDistances) -> tuple[float, float]:
        lens = distances.lens_redshift if self.lens_redshift is None else self.lens_redshift
        source = (
            distances.source_redshift
            if self.source_redshift is None
            else self.source_redshift
        )
        if lens is None or source is None:
            raise ValueError(
                "SkyProjectedKinematics requires lens and source redshifts"
            )
        return float(lens), float(source)

    def component_dispersion_uas_per_day(self, distances: LensingDistances) -> float:
        """Return the observer-frame stellar component dispersion."""

        lens_redshift, _ = self._redshifts(distances)
        return (
            self.stellar_dispersion_km_s
            * 1_000.0
            / distances.lens_m
            / (1.0 + lens_redshift)
            * _RADIANS_TO_MICROARCSECONDS
            * _SECONDS_PER_DAY
        )

    def _cmb_transverse_km_s(self) -> tuple[float, float]:
        if not self.include_cmb_dipole:
            return (0.0, 0.0)
        longitude = math.radians(self.cmb_galactic_longitude_deg)
        latitude = math.radians(self.cmb_galactic_latitude_deg)
        galactic = (
            self.cmb_speed_km_s * math.cos(latitude) * math.cos(longitude),
            self.cmb_speed_km_s * math.cos(latitude) * math.sin(longitude),
            self.cmb_speed_km_s * math.sin(latitude),
        )
        # Transpose of the standard ICRS-to-Galactic rotation matrix.
        rotation = (
            (-0.0548755604, 0.4941094279, -0.8676661490),
            (-0.8734370902, -0.4448296300, -0.1980763734),
            (-0.4838350155, 0.7469822445, 0.4559837762),
        )
        icrs = tuple(sum(row[j] * galactic[j] for j in range(3)) for row in rotation)
        ra = math.radians(self.ra_deg)
        dec = math.radians(self.dec_deg)
        east = (-math.sin(ra), math.cos(ra), 0.0)
        north = (-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec))
        return (
            sum(icrs[j] * east[j] for j in range(3)),
            sum(icrs[j] * north[j] for j in range(3)),
        )

    def mean_velocity_uas_per_day(
        self,
        distances: LensingDistances,
    ) -> tuple[float, float]:
        """Return the effective east/north angular drift of the star field."""

        lens_redshift, source_redshift = self._redshifts(distances)
        lens_velocity, source_velocity = self._peculiar_velocities(distances)
        cmb = self._cmb_transverse_km_s()
        result = []
        for axis in range(2):
            angular_per_second = (
                lens_velocity[axis]
                * 1_000.0
                / distances.lens_m
                / (1.0 + lens_redshift)
                - source_velocity[axis]
                * 1_000.0
                / distances.source_m
                / (1.0 + source_redshift)
                - cmb[axis]
                * 1_000.0
                * distances.lens_to_source_m
                / (
                    distances.lens_m
                    * distances.source_m
                    * (1.0 + lens_redshift)
                )
            )
            result.append(
                angular_per_second
                * _RADIANS_TO_MICROARCSECONDS
                * _SECONDS_PER_DAY
            )
        return (result[0], result[1])
@dataclass(frozen=True)
class StellarAperture:
    """The full circular lens-plane region populated by compact objects."""

    radius_uas: float
    center_uas: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.radius_uas)) or self.radius_uas <= 0.0:
            raise ValueError("radius_uas must be finite and positive")
        if len(self.center_uas) != 2 or any(
            not math.isfinite(float(value)) for value in self.center_uas
        ):
            raise ValueError("center_uas must contain two finite values")

    @property
    def bounding_region(self) -> PlaneRegion:
        """Return the square numerical field enclosing the full circle."""

        diameter = 2.0 * float(self.radius_uas)
        return PlaneRegion((diameter, diameter), self.center_uas)


@dataclass(frozen=True)
class StellarPopulation:
    """A physical compact-object population prior to random realization.

    The default spatial model is uniform within the complete circular stellar
    aperture derived for a microlensing system. Integration rectangles never
    truncate this population.
    """

    mass_function: MassFunction
    kinematics: StellarKinematics = StaticKinematics()
    count: int | None = None
    name: str = "stellar_population"

    def __post_init__(self) -> None:
        if not isinstance(self.mass_function, MassFunction):
            raise TypeError("mass_function must implement the MassFunction protocol")
        if not isinstance(self.kinematics, StellarKinematics):
            raise TypeError("kinematics must implement the StellarKinematics protocol")
        if self.count is not None and int(self.count) < 0:
            raise ValueError("count must be non-negative")
        if not self.name:
            raise ValueError("name must be non-empty")

    @classmethod
    def salpeter(
        cls,
        *,
        mean_mass_solar: float = 0.3,
        mass_ratio: float = 100.0,
        kinematics: StellarKinematics | None = None,
        count: int | None = None,
        name: str = "salpeter",
    ) -> StellarPopulation:
        """Construct a Salpeter population from its mean mass and mass ratio."""

        mean_mass = float(mean_mass_solar)
        ratio = float(mass_ratio)
        if mean_mass <= 0.0:
            raise ValueError("mean_mass_solar must be positive")
        if ratio <= 1.0:
            raise ValueError("mass_ratio must be greater than one")
        unit_distribution = salpeter_mass_function(1.0, ratio)
        minimum_mass = mean_mass / unit_distribution.mean_mass()
        return cls(
            mass_function=salpeter_mass_function(
                minimum_mass,
                ratio * minimum_mass,
            ),
            kinematics=StaticKinematics() if kinematics is None else kinematics,
            count=count,
            name=name,
        )

    def realize(
        self,
        aperture: StellarAperture,
        macro_lens: MacroLens,
        distances: LensingDistances,
        *,
        seed: int | None = None,
        device="cpu",
        dtype=None,
    ) -> PointMassField:
        """Sample one reproducible point-mass field inside ``aperture``."""

        import torch

        resolved_dtype = torch.float32 if dtype is None else dtype
        dispersion = self.kinematics.component_dispersion_uas_per_day(distances)
        velocity = None if dispersion == 0.0 else dispersion
        mean_velocity_method = getattr(
            self.kinematics,
            "mean_velocity_uas_per_day",
            None,
        )
        mean_velocity = (
            (0.0, 0.0)
            if mean_velocity_method is None
            else mean_velocity_method(distances)
        )
        return sample_uniform_circular_point_masses(
            aperture.radius_uas,
            macro_lens.compact_convergence,
            distances,
            self.mass_function,
            center_uas=aperture.center_uas,
            count=self.count,
            velocity_dispersion_uas_per_day=velocity,
            velocity_mean_uas_per_day=mean_velocity,
            seed=seed,
            device=device,
            dtype=resolved_dtype,
        )

    def metadata(self) -> dict[str, object]:
        """Return serializable mass-function and kinematic provenance."""

        mass_function = (
            asdict(self.mass_function)
            if is_dataclass(self.mass_function)
            else {"type": type(self.mass_function).__name__}
        )
        kinematics = (
            asdict(self.kinematics)
            if is_dataclass(self.kinematics)
            else {"type": type(self.kinematics).__name__}
        )
        return {
            "name": self.name,
            "count_override": self.count,
            "mass_function": {
                "type": type(self.mass_function).__name__,
                **mass_function,
            },
            "kinematics": {
                "type": type(self.kinematics).__name__,
                **kinematics,
            },
        }


def circular_stellar_aperture(
    macro_lens: MacroLens,
    source_region: PlaneRegion,
    distances: LensingDistances,
    population: StellarPopulation,
    *,
    light_loss: float = 0.01,
    safety_scale: float = 1.5,
    duration_days: float = 0.0,
    motion_sigma_margin: float = 5.0,
    source_support_radius_uas: float | None = None,
) -> StellarAperture:
    """Derive the complete circular stellar aperture for one system.

    A physical circular support is expanded by the worst macro-lens stretch
    and the stochastic deflection allowance. When no physical support is
    supplied, all four corners of the padded rectangular source grid are
    transformed explicitly. The result is enlarged by ``safety_scale`` and a
    conservative stellar-motion allowance.
    """

    if safety_scale < 1.0:
        raise ValueError("safety_scale must be at least one")
    if duration_days < 0.0:
        raise ValueError("duration_days must be non-negative")
    if motion_sigma_margin < 0.0:
        raise ValueError("motion_sigma_margin must be non-negative")
    if source_support_radius_uas is not None and source_support_radius_uas <= 0.0:
        raise ValueError("source_support_radius_uas must be positive")
    rectangle = rectangular_lens_region(
        macro_lens,
        source_region,
        distances,
        population.mass_function,
        light_loss=light_loss,
    )
    source_fov_y, source_fov_x = source_region.field_of_view_uas
    stochastic_margin = _stochastic_deflection_margin_uas(
        macro_lens,
        distances,
        population.mass_function,
        float(light_loss),
    )
    half_source_x = 0.5 * source_fov_x + stochastic_margin
    half_source_y = 0.5 * source_fov_y + stochastic_margin
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
    if source_support_radius_uas is None:
        base_radius = max(
            math.hypot(
                inverse_xx * (sign_x * half_source_x)
                + inverse_xy * (sign_y * half_source_y),
                inverse_xy * (sign_x * half_source_x)
                + inverse_yy * (sign_y * half_source_y),
            )
            for sign_x in (-1.0, 1.0)
            for sign_y in (-1.0, 1.0)
        )
    else:
        minimum_eigenvalue = min(
            abs(1.0 - macro_lens.convergence - macro_lens.shear),
            abs(1.0 - macro_lens.convergence + macro_lens.shear),
        )
        if minimum_eigenvalue <= 1.0e-12:
            raise ValueError("macro-lens eigenvalue is too close to zero")
        base_radius = (
            float(source_support_radius_uas) + stochastic_margin
        ) / minimum_eigenvalue
    dispersion = population.kinematics.component_dispersion_uas_per_day(distances)
    mean_velocity_method = getattr(
        population.kinematics,
        "mean_velocity_uas_per_day",
        None,
    )
    mean_velocity = (
        (0.0, 0.0)
        if mean_velocity_method is None
        else mean_velocity_method(distances)
    )
    bulk_speed = math.hypot(*mean_velocity)
    motion_padding = (
        float(motion_sigma_margin) * dispersion + bulk_speed
    ) * float(duration_days)
    return StellarAperture(
        radius_uas=float(safety_scale) * base_radius + motion_padding,
        center_uas=rectangle.center_uas,
    )
