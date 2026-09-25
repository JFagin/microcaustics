"""Empirical quasar spectra layered on physical disk continua."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Protocol

import numpy as np
import torch

from .bandpasses import BandpassSet
from .random import derive_seed

_PARSEC_M = 3.085677581491367e16
_AB_ZERO_POINT_JY = 3631.0
_ABSOLUTE_MAGNITUDE_REFERENCE_REDSHIFT = 2.0


class Distribution(Protocol):
    """A scalar population distribution sampled by NumPy's local generator."""

    def sample(self, generator: np.random.Generator) -> float:
        """Draw one scalar using the caller-owned random generator."""

        ...


@dataclass(frozen=True)
class Normal:
    """Normal scalar distribution."""

    mean: float
    std: float

    def sample(self, generator: np.random.Generator) -> float:
        """Draw one normally distributed scalar."""

        if self.std < 0.0:
            raise ValueError("normal standard deviation must be non-negative")
        return float(generator.normal(self.mean, self.std))


@dataclass(frozen=True)
class LogNormal:
    """Base-10 lognormal scalar distribution."""

    log10_mean: float
    log10_std: float

    def sample(self, generator: np.random.Generator) -> float:
        """Draw one scalar whose base-10 logarithm is normal."""

        if self.log10_std < 0.0:
            raise ValueError("lognormal standard deviation must be non-negative")
        return float(10.0 ** generator.normal(self.log10_mean, self.log10_std))


@dataclass(frozen=True)
class ClippedNormal:
    """Normal scalar distribution clipped to a finite interval."""

    mean: float
    std: float
    minimum: float
    maximum: float

    def sample(self, generator: np.random.Generator) -> float:
        """Draw and clip one normally distributed scalar."""

        if self.std < 0.0 or self.minimum > self.maximum:
            raise ValueError("invalid clipped-normal parameters")
        return float(
            np.clip(
                generator.normal(self.mean, self.std),
                self.minimum,
                self.maximum,
            )
        )


ScalarParameter = float | int | Distribution


def _sample(value: ScalarParameter, generator: np.random.Generator) -> float:
    method = getattr(value, "sample", None)
    return float(value) if method is None else float(method(generator))


@dataclass(frozen=True)
class QuasarSpectrum:
    """One deterministic realization of an empirical quasar spectrum.

    The supplied physical disk remains the continuum model. This object adds
    relative emission-line and host templates, intrinsic reddening, and
    intergalactic absorption. It contains no mutable or implicit random state.
    """

    ebv: float = 0.01
    emission_line_scale: float = 0.9936
    emission_line_type: float | None = None
    preserve_line_equivalent_width: bool = True
    halpha_scale: float = 1.0
    lya_scale: float = 1.0
    narrow_line_scale: float = 1.0
    host_fraction: float = 0.244
    host_luminosity_slope: float = 0.684
    absolute_i_magnitude: float | None = None
    absolute_i_magnitude_offset: float = 0.0
    baldwin_slope: float = 0.183
    baldwin_reference_magnitude: float = -27.0
    include_emission_lines: bool = True
    include_host: bool = True
    include_igm_absorption: bool = True
    lyman_limit_angstrom: float = 912.0
    global_magnitude_offset: float = 0.0
    color_tilt_magnitude: float = 0.0
    band_magnitude_scatter: float = 0.0
    amplitude_color_scatter: float = 0.0
    amplitude_band_scatter: float = 0.0
    scatter_seed: int | None = None
    population_seed: int | None = None

    def __post_init__(self) -> None:
        finite = {
            "ebv": self.ebv,
            "emission_line_scale": self.emission_line_scale,
            "halpha_scale": self.halpha_scale,
            "lya_scale": self.lya_scale,
            "narrow_line_scale": self.narrow_line_scale,
            "host_fraction": self.host_fraction,
            "host_luminosity_slope": self.host_luminosity_slope,
            "absolute_i_magnitude_offset": self.absolute_i_magnitude_offset,
            "baldwin_slope": self.baldwin_slope,
            "baldwin_reference_magnitude": self.baldwin_reference_magnitude,
            "lyman_limit_angstrom": self.lyman_limit_angstrom,
            "global_magnitude_offset": self.global_magnitude_offset,
            "color_tilt_magnitude": self.color_tilt_magnitude,
            "band_magnitude_scatter": self.band_magnitude_scatter,
            "amplitude_color_scatter": self.amplitude_color_scatter,
            "amplitude_band_scatter": self.amplitude_band_scatter,
        }
        if any(not math.isfinite(float(value)) for value in finite.values()):
            raise ValueError("quasar spectrum parameters must be finite")
        for name, value in (
            ("emission_line_type", self.emission_line_type),
            ("absolute_i_magnitude", self.absolute_i_magnitude),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if min(
            self.ebv,
            self.emission_line_scale,
            self.halpha_scale,
            self.lya_scale,
            self.narrow_line_scale,
        ) < 0.0:
            raise ValueError("reddening and emission-line scales must be non-negative")
        if not 0.0 <= self.host_fraction < 1.0:
            raise ValueError("host_fraction must lie in [0, 1)")
        if min(
            self.band_magnitude_scatter,
            self.amplitude_band_scatter,
        ) < 0.0:
            raise ValueError("scatter standard deviations must be non-negative")
        if self.lyman_limit_angstrom <= 0.0:
            raise ValueError("lyman_limit_angstrom must be positive")

    def required_wavelengths(self, source_redshift: float) -> tuple[float, ...]:
        """Observed continuum nodes needed by line and host normalization."""

        parts = []
        if self.include_host:
            parts.append(
                np.linspace(4000.0, 5000.0, 32, dtype=np.float64)
                * (1.0 + float(source_redshift))
            )
        needs_physical_magnitude = self.absolute_i_magnitude is None and (
            self.include_host
            or (self.include_emission_lines and self.emission_line_type is None)
        )
        if needs_physical_magnitude:
            i_band = BandpassSet.lsst().bandpasses[3]
            z2_nodes, _ = i_band.quadrature(32)
            parts.append(
                z2_nodes
                / (1.0 + _ABSOLUTE_MAGNITUDE_REFERENCE_REDSHIFT)
                * (1.0 + float(source_redshift))
            )
        if not parts:
            return ()
        return tuple(np.unique(np.concatenate(parts)).tolist())

    def infer_absolute_i_magnitude(
        self,
        wavelengths_angstrom,
        mean_continuum_fnu,
        *,
        source_redshift: float,
        luminosity_distance_m: float,
    ) -> float:
        """Infer the continuum's conventional ``M_i(z=2)`` in AB magnitudes.

        The same intrinsic continuum is placed at redshift two, integrated
        through the bundled LSST ``i`` response, and referred to 10 pc. This
        reproduces the luminosity normalization used by the empirical line
        and host prescriptions without another disk evaluation.
        """

        wavelength = torch.as_tensor(wavelengths_angstrom)
        continuum = torch.as_tensor(
            mean_continuum_fnu,
            device=wavelength.device,
            dtype=wavelength.dtype,
        )
        if wavelength.ndim != 1 or continuum.shape[-1] != wavelength.numel():
            raise ValueError("continuum final axis must match the wavelength grid")
        if not math.isfinite(luminosity_distance_m) or luminosity_distance_m <= 0:
            raise ValueError("luminosity_distance_m must be finite and positive")
        i_band = BandpassSet.lsst().bandpasses[3]
        z2_nodes, weights = i_band.quadrature(32)
        z2_nodes = wavelength.new_tensor(z2_nodes)
        current_nodes = z2_nodes * (
            (1.0 + float(source_redshift))
            / (1.0 + _ABSOLUTE_MAGNITUDE_REFERENCE_REDSHIFT)
        )
        sampled = _log_flux_interpolate(
            wavelength, continuum, current_nodes, check_range=True
        )
        # For f_nu at fixed rest wavelength, moving the source from its true
        # redshift to z=2 contributes the inverse (1+z) factor. Multiplying by
        # the z=2 distance modulus then cancels the arbitrary z=2 distance.
        ten_parsec_flux = sampled * (
            (float(luminosity_distance_m) / (10.0 * _PARSEC_M)) ** 2
            * (1.0 + _ABSOLUTE_MAGNITUDE_REFERENCE_REDSHIFT)
            / (1.0 + float(source_redshift))
        )
        integrated = ten_parsec_flux @ wavelength.new_tensor(weights)
        tiny = torch.finfo(integrated.dtype).tiny
        return float(-2.5 * torch.log10(integrated.clamp_min(tiny) / _AB_ZERO_POINT_JY))

    def band_offsets(
        self,
        band_names: tuple[str, ...],
        wavelengths_angstrom: torch.Tensor,
        *,
        amplitude: bool = False,
        include_smooth: bool = True,
    ) -> torch.Tensor:
        """Return reproducible phenomenological magnitude offsets by band."""

        wavelength = torch.as_tensor(wavelengths_angstrom)
        if wavelength.ndim != 1 or wavelength.numel() != len(band_names):
            raise ValueError("wavelengths must contain one value per band name")
        if not bool(torch.all(torch.isfinite(wavelength) & (wavelength > 0.0))):
            raise ValueError("band wavelengths must be finite and positive")
        pivot = wavelength.new_tensor(6000.0)
        if amplitude:
            tilt = self.amplitude_color_scatter
            std = self.amplitude_band_scatter
            component = "quasar_spectrum_amplitude_bands"
            base = 0.0
        else:
            tilt = self.color_tilt_magnitude
            std = self.band_magnitude_scatter
            component = "quasar_spectrum_brightness_bands"
            base = self.global_magnitude_offset
        offsets = (
            base + tilt * torch.log(wavelength / pivot)
            if include_smooth
            else torch.zeros_like(wavelength)
        )
        if std > 0.0:
            generator = np.random.default_rng(derive_seed(self.scatter_seed, component))
            residual = torch.as_tensor(
                generator.normal(0.0, std, len(band_names)),
                device=wavelength.device,
                dtype=wavelength.dtype,
            )
            offsets = offsets + residual
        return offsets

    def components(
        self,
        wavelengths_angstrom,
        mean_continuum_fnu,
        *,
        source_redshift: float,
    ) -> QuasarSpectrumComponents:
        """Evaluate attenuated continuum, line, and host flux densities.

        ``mean_continuum_fnu`` may have any units. Returned components preserve
        those units. The wavelength axis must be last and is interpreted in
        the observer frame.
        """

        wavelength = torch.as_tensor(wavelengths_angstrom)
        continuum = torch.as_tensor(
            mean_continuum_fnu,
            device=wavelength.device,
            dtype=wavelength.dtype,
        )
        if wavelength.ndim != 1 or continuum.shape[-1] != wavelength.numel():
            raise ValueError("continuum final axis must match the wavelength grid")
        rest = wavelength / (1.0 + float(source_redshift))
        template = _templates_at(rest)

        # Relative f_lambda is sufficient: the common f_nu-to-f_lambda
        # constant cancels when components are converted back to f_nu.
        continuum_lambda = continuum / wavelength.square()
        line_lambda = torch.zeros_like(continuum_lambda)
        if self.include_emission_lines:
            line_type = self.emission_line_type
            if line_type is None:
                line_type = (
                    self._absolute_magnitude(source_redshift)
                    - self.baldwin_reference_magnitude
                ) * self.baldwin_slope
            median = template["median"]
            if line_type > 0.0:
                amount = min(float(line_type), 3.0)
                lines = amount * template["peaky"] + (1.0 - amount) * median
            elif line_type < 0.0:
                amount = min(abs(float(line_type)), 2.0)
                lines = amount * template["windy"] + (1.0 - amount) * median
            else:
                lines = median
            lines = lines + (self.narrow_line_scale - 1.0) * template["narrow"]
            lines = torch.where(
                ((rest > 4930.0) & (rest < 5030.0) & (lines < 0.0))
                | ((rest > 1150.0) & (rest < 1200.0) & (lines < 0.0)),
                torch.zeros_like(lines),
                lines,
            )
            scaling = torch.full_like(rest, self.emission_line_scale)
            scaling = torch.where(
                (rest >= 6000.0) & (rest <= 7000.0),
                scaling * self.halpha_scale,
                scaling,
            )
            scaling = torch.where(rest < 1350.0, scaling * self.lya_scale, scaling)
            if self.preserve_line_equivalent_width:
                ratio = lines / template["reference_continuum"].clamp_min(
                    torch.finfo(rest.dtype).tiny
                )
                line_lambda = scaling * ratio * continuum_lambda
            else:
                reference = _nearest_value(
                    rest,
                    continuum_lambda,
                    5500.0,
                )
                template_reference = _nearest_value(
                    rest,
                    template["reference_continuum"],
                    5500.0,
                ).clamp_min(torch.finfo(rest.dtype).tiny)
                line_lambda = scaling * lines * reference / template_reference
            # The empirical templates contain genuine negative residuals. Keep
            # them while enforcing the original model's non-negative total
            # quasar spectrum rather than clipping the line component itself.
            line_lambda = torch.maximum(line_lambda, -continuum_lambda)

        dust = _dust_transmission(rest, self.ebv, template["reddening"])
        igm = (
            _igm_transmission(
                rest,
                float(source_redshift),
                self.lyman_limit_angstrom,
            )
            if self.include_igm_absorption
            else torch.ones_like(rest)
        )
        continuum_observed = continuum_lambda * dust * igm
        line_observed = line_lambda * dust * igm

        host_lambda = torch.zeros_like(continuum_lambda)
        if self.include_host and self.host_fraction > 0.0:
            normalization = (rest >= 4000.0) & (rest <= 5000.0)
            if int(normalization.sum()) < 2:
                raise ValueError(
                    "host emission requires continuum samples spanning rest-frame "
                    "4000-5000 Angstrom. Include spectrum.required_wavelengths(z)"
                )
            quasar_reference = torch.trapezoid(
                (continuum_lambda + line_lambda)[normalization],
                rest[normalization],
            )
            host_reference = torch.trapezoid(
                template["host"][normalization],
                rest[normalization],
            ).clamp_min(torch.finfo(rest.dtype).tiny)
            luminosity = 10.0 ** (
                -0.4 * (self._absolute_magnitude(source_redshift) + 23.0)
            )
            luminosity_scale = luminosity ** (self.host_luminosity_slope - 1.0)
            fraction_scale = self.host_fraction / (1.0 - self.host_fraction)
            host_lambda = (
                template["host"]
                * quasar_reference
                / host_reference
                * fraction_scale
                * luminosity_scale
                * igm
            )

        # Undo the relative f_nu-to-f_lambda conversion.
        return QuasarSpectrumComponents(
            continuum_fnu=continuum_observed * wavelength.square(),
            emission_line_fnu=line_observed * wavelength.square(),
            host_fnu=host_lambda * wavelength.square(),
        )

    def _absolute_magnitude(self, source_redshift: float) -> float:
        if self.absolute_i_magnitude is not None:
            return float(self.absolute_i_magnitude) + self.absolute_i_magnitude_offset
        redshift = np.asarray(
            [0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3, 3.7, 4.13, 4.5]
        )
        magnitude = np.asarray(
            [-21.76, -22.9, -24.1, -25.4, -26.0, -26.6, -27.1, -27.6,
             -27.9, -28.1, -28.4, -28.6, -28.9]
        )
        return float(np.interp(source_redshift, redshift, magnitude)) + (
            self.absolute_i_magnitude_offset
        )


@dataclass(frozen=True)
class QuasarSpectrumPopulation:
    """Configurable independent distributions for quasar-spectrum parameters."""

    ebv: ScalarParameter = LogNormal(-2.0, 0.5)
    emission_line_scale: ScalarParameter = LogNormal(math.log10(0.9936), 0.075)
    emission_line_type: ScalarParameter | None = None
    halpha_scale: ScalarParameter = LogNormal(0.0, 0.1)
    lya_scale: ScalarParameter = LogNormal(0.0, 0.1)
    narrow_line_scale: ScalarParameter = LogNormal(0.0, 0.1)
    preserve_line_equivalent_width: bool = True
    host_fraction: ScalarParameter = ClippedNormal(0.244, 0.075, 0.0, 0.7)
    host_luminosity_slope: ScalarParameter = 0.684
    absolute_i_magnitude: ScalarParameter | None = None
    absolute_i_magnitude_scatter: ScalarParameter = Normal(0.0, 0.1)
    baldwin_slope: ScalarParameter = 0.183
    baldwin_reference_magnitude: ScalarParameter = -27.0
    lyman_limit_angstrom: ScalarParameter = 912.0
    global_magnitude_scatter: ScalarParameter = Normal(0.0, 0.08)
    color_tilt_scatter: ScalarParameter = Normal(0.0, 0.06)
    band_magnitude_scatter: float = 0.02
    amplitude_color_scatter: ScalarParameter = Normal(0.0, 0.04)
    amplitude_band_scatter: float = 0.02
    include_emission_lines: bool = True
    include_host: bool = True
    include_igm_absorption: bool = True

    def sample(
        self,
        size: int | None = None,
        *,
        seed: int | None = None,
    ) -> QuasarSpectrum | tuple[QuasarSpectrum, ...]:
        """Draw one or several reproducible, mutually independent spectra."""

        if size is None:
            return self._sample_one(seed)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer or None")
        return tuple(
            self._sample_one(derive_seed(seed, f"quasar_spectrum:{index}"))
            for index in range(size)
        )

    def _sample_one(self, seed: int | None) -> QuasarSpectrum:
        generator = np.random.default_rng(derive_seed(seed, "quasar_spectrum"))
        scatter_seed = (
            int(generator.integers(0, 2**63 - 1))
            if seed is None
            else derive_seed(seed, "quasar_spectrum_band_scatter")
        )
        return QuasarSpectrum(
            ebv=_sample(self.ebv, generator),
            emission_line_scale=_sample(self.emission_line_scale, generator),
            emission_line_type=(
                None
                if self.emission_line_type is None
                else _sample(self.emission_line_type, generator)
            ),
            halpha_scale=_sample(self.halpha_scale, generator),
            lya_scale=_sample(self.lya_scale, generator),
            narrow_line_scale=_sample(self.narrow_line_scale, generator),
            preserve_line_equivalent_width=bool(
                self.preserve_line_equivalent_width
            ),
            host_fraction=_sample(self.host_fraction, generator),
            host_luminosity_slope=_sample(self.host_luminosity_slope, generator),
            absolute_i_magnitude=(
                None
                if self.absolute_i_magnitude is None
                else _sample(self.absolute_i_magnitude, generator)
            ),
            absolute_i_magnitude_offset=_sample(
                self.absolute_i_magnitude_scatter, generator
            ),
            baldwin_slope=_sample(self.baldwin_slope, generator),
            baldwin_reference_magnitude=_sample(
                self.baldwin_reference_magnitude, generator
            ),
            lyman_limit_angstrom=_sample(self.lyman_limit_angstrom, generator),
            global_magnitude_offset=_sample(
                self.global_magnitude_scatter, generator
            ),
            color_tilt_magnitude=_sample(self.color_tilt_scatter, generator),
            band_magnitude_scatter=float(self.band_magnitude_scatter),
            amplitude_color_scatter=_sample(
                self.amplitude_color_scatter, generator
            ),
            amplitude_band_scatter=float(self.amplitude_band_scatter),
            include_emission_lines=bool(self.include_emission_lines),
            include_host=bool(self.include_host),
            include_igm_absorption=bool(self.include_igm_absorption),
            scatter_seed=scatter_seed,
            population_seed=seed,
        )


@dataclass(frozen=True)
class QuasarSpectrumComponents:
    """Mean spectral components evaluated on one wavelength grid."""

    continuum_fnu: torch.Tensor
    emission_line_fnu: torch.Tensor
    host_fnu: torch.Tensor

    @property
    def total_fnu(self) -> torch.Tensor:
        """Return the sum of continuum, line, and host flux density."""

        return self.continuum_fnu + self.emission_line_fnu + self.host_fnu


def _log_flux_interpolate(
    x: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    *,
    check_range: bool = False,
) -> torch.Tensor:
    """Interpolate positive continuum in log wavelength and log flux.

    Intervals touching zero fall back to linear flux interpolation, preserving
    exactly dark channels without inventing a positive floor.
    """

    if check_range:
        # The same required endpoint can round to adjacent float32 values
        # when the sampling plan and normalization use different arithmetic.
        tolerance = 4 * torch.finfo(x.dtype).eps * torch.maximum(x[0], x[-1])
        if query[0] < x[0] - tolerance or query[-1] > x[-1] + tolerance:
            raise ValueError(
                "continuum wavelengths do not span the M_i(z=2) normalization grid; "
                "include spectrum.required_wavelengths(source_redshift)"
            )
        query = query.clamp(min=x[0], max=x[-1])
    upper = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    lower = upper - 1
    y0 = values.index_select(-1, lower)
    y1 = values.index_select(-1, upper)
    log_x = torch.log(x)
    fraction = (torch.log(query) - log_x.index_select(0, lower)) / (
        log_x.index_select(0, upper) - log_x.index_select(0, lower)
    )
    tiny = torch.finfo(values.dtype).tiny
    log_y0 = torch.log(y0.clamp_min(tiny))
    logged = torch.exp(
        log_y0 + fraction * (torch.log(y1.clamp_min(tiny)) - log_y0)
    )
    linear_fraction = (query - x.index_select(0, lower)) / (
        x.index_select(0, upper) - x.index_select(0, lower)
    )
    return torch.where((y0 > 0) & (y1 > 0), logged, y0 + linear_fraction * (y1 - y0))


def _nearest_value(x: torch.Tensor, values: torch.Tensor, target: float) -> torch.Tensor:
    index = torch.argmin(torch.abs(x - target))
    return values[..., index]


def _dust_transmission(
    rest_wavelength: torch.Tensor,
    ebv: float,
    reddening_curve: torch.Tensor,
) -> torch.Tensor:
    if ebv == 0.0:
        return torch.ones_like(rest_wavelength)
    extinction = float(ebv) * (reddening_curve + 3.1)
    return torch.pow(rest_wavelength.new_tensor(10.0), -extinction / 2.5)


def _tau_eff(redshift: torch.Tensor) -> torch.Tensor:
    value = 0.751 * ((1.0 + redshift) / 4.5) ** 2.90 - 0.132
    return torch.clamp(value, min=0.0)


def _igm_transmission(
    rest_wavelength: torch.Tensor,
    source_redshift: float,
    lyman_limit: float,
) -> torch.Tensor:
    transmission = torch.ones_like(rest_wavelength)
    if _tau_eff(rest_wavelength.new_tensor(source_redshift)) > 0.0:
        for limit, strength in ((972.0, 0.056), (1026.0, 0.16), (1216.0, 1.0)):
            mask = rest_wavelength < limit
            absorption_redshift = (
                (1.0 + float(source_redshift)) * rest_wavelength / limit - 1.0
            )
            optical_depth = torch.where(
                mask,
                strength * _tau_eff(absorption_redshift),
                torch.zeros_like(rest_wavelength),
            )
            transmission = transmission * torch.exp(-optical_depth)
    return torch.where(
        rest_wavelength < float(lyman_limit),
        torch.zeros_like(transmission),
        transmission,
    )


@lru_cache(maxsize=16)
def _load_template_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = files("microcaustics.data.quasar_spectrum")
    with root.joinpath("qsosed_emlines_20210625.dat").open("rb") as handle:
        emission = np.loadtxt(handle)
    with root.joinpath("S0_template_norm.sed").open("rb") as handle:
        host = np.loadtxt(handle)
    with root.joinpath("pl_ext_comp_03.sph").open("rb") as handle:
        reddening = np.loadtxt(handle)
    return emission, host, reddening


@lru_cache(maxsize=16)
def _template_tensors(
    device_string: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep immutable template tables resident on each requested device."""

    emission, host, reddening = _load_template_tables()
    device = torch.device(device_string)
    return (
        torch.as_tensor(emission[:, 0].copy(), device=device, dtype=dtype),
        torch.as_tensor(
            emission[:, 1:].T.copy(), device=device, dtype=dtype
        ),
        torch.as_tensor(host[:, 0].copy(), device=device, dtype=dtype),
        torch.as_tensor(host[:, 1].copy(), device=device, dtype=dtype),
        torch.as_tensor(reddening[:, 0].copy(), device=device, dtype=dtype),
        torch.as_tensor(reddening[:, 1].copy(), device=device, dtype=dtype),
    )


def _interpolate_template(
    wavelength: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Match NumPy's constant-edge linear interpolation on one device."""

    clipped = query.clamp(min=wavelength[0], max=wavelength[-1])
    upper = torch.searchsorted(wavelength, clipped).clamp(1, wavelength.numel() - 1)
    lower = upper - 1
    x0 = wavelength.index_select(0, lower)
    x1 = wavelength.index_select(0, upper)
    y0 = values.index_select(-1, lower)
    y1 = values.index_select(-1, upper)
    return y0 + (clipped - x0) / (x1 - x0) * (y1 - y0)


def _templates_at(rest_wavelength: torch.Tensor) -> dict[str, torch.Tensor]:
    device_string = str(rest_wavelength.device)
    emission_x, emission_y, host_x, host_y, reddening_x, reddening_y = (
        _template_tensors(device_string, rest_wavelength.dtype)
    )
    emission = _interpolate_template(emission_x, emission_y, rest_wavelength)
    return {
        "median": emission[0],
        "reference_continuum": emission[1],
        "peaky": emission[2],
        "windy": emission[3],
        "narrow": emission[4],
        "host": _interpolate_template(host_x, host_y, rest_wavelength),
        "reddening": _interpolate_template(
            reddening_x, reddening_y, rest_wavelength
        ),
    }
