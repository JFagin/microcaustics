"""Composable intrinsic variability for arbitrary pixelated sources."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .base import PixelatedSource, SourceGeometry, _as_times


@dataclass(frozen=True)
class TimeShiftedSource:
    """Evaluate a source at ``observer_time - delay_days``.

    This lightweight adapter is useful for cosmological arrival-time delays in
    resolved multiply imaged sources. It changes source evolution only. Lens
    motion and source-plane trajectories continue to use observer time.
    """

    source: PixelatedSource
    delay_days: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.delay_days)):
            raise ValueError("delay_days must be finite")

    @property
    def geometry(self) -> SourceGeometry:
        """Return the wrapped source geometry unchanged."""

        return self.source.geometry

    @property
    def is_time_static(self) -> bool:
        """Return whether the wrapped source is time independent."""

        return bool(self.source.is_time_static)

    def brightness(
        self,
        times_days,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return brightness at the corresponding source-emission times."""

        times = _as_times(times_days)
        resolved_dtype = (
            dtype
            if dtype is not None
            else times.dtype
            if times.is_floating_point()
            else torch.get_default_dtype()
        )
        times = times.to(
            device=times.device if device is None else device,
            dtype=resolved_dtype,
        )
        shifted = times - float(self.delay_days)
        return self.source.brightness(shifted, dtype=dtype, device=device)

    def metadata(self) -> Mapping[str, object]:
        """Return the delay convention and wrapped source provenance."""

        return {
            "type": "time_shifted",
            "arrival_time_delay_days": float(self.delay_days),
            "source_time_convention": "observer_time_minus_arrival_delay",
            "source": dict(self.source.metadata()),
        }


@runtime_checkable
class DrivingSignal(Protocol):
    """Multiplicative band amplitudes evaluated at arbitrary source times."""

    def amplitudes(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        bands: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return non-negative values with shape ``[time, band]``."""

        ...

    def metadata(self) -> Mapping[str, object]:
        """Return serializable signal provenance."""

        ...


@dataclass(frozen=True)
class CallableDrivingSignal:
    """Adapt a user function returning one or one-per-band amplitude."""

    function: Callable[[torch.Tensor], torch.Tensor]
    name: str = "callable"
    user_metadata: Mapping[str, object] | None = None

    def amplitudes(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        bands: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Evaluate and validate the callable's multiplicative amplitudes."""

        times = _as_times(times_days).to(device=device, dtype=dtype)
        values = torch.as_tensor(self.function(times), device=device, dtype=dtype)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != times.numel() or values.shape[1] not in (1, bands):
            raise ValueError(
                "driving callable must return [time] or [time, band] amplitudes"
            )
        if values.shape[1] == 1:
            values = values.expand(-1, bands)
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values < 0)):
            raise ValueError("driving amplitudes must be finite and non-negative")
        return values

    def metadata(self) -> Mapping[str, object]:
        """Return serializable callable provenance."""

        return {
            "type": "callable",
            "name": self.name,
            **dict(self.user_metadata or {}),
        }


@dataclass(frozen=True)
class TabulatedDrivingSignal:
    """Linearly interpolate a sampled scalar or multiband driving signal."""

    times_days: torch.Tensor
    values: torch.Tensor
    extrapolation: str = "error"
    name: str = "tabulated"
    user_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.times_days)
        values = torch.as_tensor(self.values, device=times.device)
        if times.ndim != 1 or times.numel() < 1:
            raise ValueError("times_days must be a non-empty 1D array")
        if not times.is_floating_point() or not values.is_floating_point():
            raise TypeError("tabulated signal arrays must use floating dtypes")
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != times.numel():
            raise ValueError("values must have shape [time] or [time, band]")
        if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
            raise ValueError("times_days must be strictly increasing")
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values < 0)):
            raise ValueError("driving values must be finite and non-negative")
        if self.extrapolation not in {"error", "hold"}:
            raise ValueError("extrapolation must be 'error' or 'hold'")
        object.__setattr__(self, "times_days", times)
        object.__setattr__(self, "values", values)

    def amplitudes(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        bands: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Interpolate amplitudes, optionally holding endpoint values."""

        query = _as_times(times_days).to(device=device, dtype=dtype)
        times = self.times_days.to(device=device, dtype=dtype)
        values = self.values.to(device=device, dtype=dtype)
        if values.shape[1] not in (1, bands):
            raise ValueError("tabulated signal band count does not match the source")
        if self.extrapolation == "error" and bool(
            torch.any((query < times[0]) | (query > times[-1]))
        ):
            raise ValueError("requested time lies outside the tabulated signal")
        if times.numel() == 1:
            result = values[0].expand(query.numel(), -1)
        else:
            clipped = query.clamp(times[0], times[-1])
            right = torch.searchsorted(times, clipped, right=True).clamp(
                1,
                times.numel() - 1,
            )
            left = right - 1
            fraction = (clipped - times[left]) / (times[right] - times[left])
            result = values[left] + fraction[:, None] * (
                values[right] - values[left]
            )
        return result.expand(-1, bands) if result.shape[1] == 1 else result

    def metadata(self) -> Mapping[str, object]:
        """Return signal interpolation provenance without embedding samples."""

        return {
            "type": "tabulated",
            "name": self.name,
            "sample_count": int(self.times_days.numel()),
            "signal_bands": int(self.values.shape[1]),
            "extrapolation": self.extrapolation,
            **dict(self.user_metadata or {}),
        }


@dataclass(frozen=True)
class BrokenPowerLawPSD:
    """Smoothly broken power spectral density in inverse days.

    The convention is
    ``f**(-alpha_low) / (1 + (f/f_break)**(alpha_high-alpha_low))``.
    Absolute normalization is intentionally irrelevant because the synthesized
    realization is later assigned an explicit mean and standard deviation.
    """

    break_timescale_days: float = 200.0
    low_frequency_slope: float = 1.0
    high_frequency_slope: float = 3.0

    def __post_init__(self) -> None:
        if self.break_timescale_days <= 0 or not math.isfinite(
            self.break_timescale_days
        ):
            raise ValueError("break_timescale_days must be finite and positive")
        if not math.isfinite(self.low_frequency_slope) or not math.isfinite(
            self.high_frequency_slope
        ):
            raise ValueError("PSD slopes must be finite")
        if self.high_frequency_slope < self.low_frequency_slope:
            raise ValueError("high-frequency slope must not be shallower")

    def __call__(self, frequency_per_day: torch.Tensor) -> torch.Tensor:
        """Evaluate the unnormalized PSD at positive frequencies."""

        frequency = torch.as_tensor(frequency_per_day)
        if bool(torch.any(frequency <= 0)):
            raise ValueError("PSD frequencies must be positive")
        break_frequency = 1.0 / float(self.break_timescale_days)
        slope_change = self.high_frequency_slope - self.low_frequency_slope
        return frequency.pow(-self.low_frequency_slope) / (
            1.0 + (frequency / break_frequency).pow(slope_change)
        )

    def metadata(self) -> Mapping[str, object]:
        """Return the broken-power-law parameters."""

        return {
            "psd": "smooth_broken_power_law",
            "break_timescale_days": float(self.break_timescale_days),
            "low_frequency_slope": float(self.low_frequency_slope),
            "high_frequency_slope": float(self.high_frequency_slope),
        }


def _signal_band_values(
    value: float | Sequence[float] | torch.Tensor,
    bands: int,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast one scalar or validate one value per synthesized band."""

    result = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if result.numel() == 1:
        result = result.expand(bands)
    if result.numel() != bands or not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must be finite and scalar or one value per band")
    return result


def driving_signal_from_psd(
    times_days: torch.Tensor | Sequence[float],
    psd: Callable[[torch.Tensor], torch.Tensor],
    *,
    mean_amplitude: float | Sequence[float] | torch.Tensor = 1.0,
    standard_deviation: float | Sequence[float] | torch.Tensor = 0.3,
    padding_factor: int = 5,
    crop_start_samples: int | None = None,
    fourier_sampling: str = "gaussian",
    amplitude_transform: str = "lognormal",
    seed: int | None = None,
    extrapolation: str = "error",
    name: str = "psd_driving_signal",
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> TabulatedDrivingSignal:
    """Synthesize a positive driving signal from an arbitrary PSD callable.

    The requested time grid must be regular. ``psd`` receives the positive FFT
    frequencies in inverse days and returns either ``[frequency]`` or
    ``[band, frequency]`` non-negative power. ``gaussian`` sampling draws the
    real and imaginary Fourier coefficients. ``random_phase`` fixes their
    amplitudes to ``sqrt(PSD)`` and matches the paper simulator. Padding and
    cropping suppress the artificial periodic boundary.
    """

    resolved_device = torch.device(device)
    times = torch.as_tensor(times_days, device=resolved_device, dtype=dtype).reshape(-1)
    if times.numel() < 2 or not bool(torch.all(torch.isfinite(times))):
        raise ValueError("PSD synthesis requires at least two finite times")
    intervals = times[1:] - times[:-1]
    if bool(torch.any(intervals <= 0)) or not torch.allclose(
        intervals,
        intervals[0].expand_as(intervals),
        rtol=1.0e-6,
        atol=1.0e-10,
    ):
        raise ValueError("PSD synthesis requires a regular increasing time grid")
    if not isinstance(padding_factor, int) or padding_factor < 1:
        raise ValueError("padding_factor must be a positive integer")
    if fourier_sampling not in {"gaussian", "random_phase"}:
        raise ValueError("fourier_sampling must be 'gaussian' or 'random_phase'")
    if amplitude_transform not in {"lognormal", "linear"}:
        raise ValueError("amplitude_transform must be 'lognormal' or 'linear'")
    sample_count = int(times.numel())
    padded_count = padding_factor * sample_count
    crop_start = (
        (padded_count - sample_count) // 2
        if crop_start_samples is None
        else int(crop_start_samples)
    )
    if crop_start < 0 or crop_start + sample_count > padded_count:
        raise ValueError("crop_start_samples does not fit inside the padded series")
    frequencies = torch.fft.rfftfreq(
        padded_count,
        d=float(intervals[0]),
        device=resolved_device,
        dtype=dtype,
    )
    positive_frequency = frequencies[1:]
    power = torch.as_tensor(
        psd(positive_frequency),
        device=resolved_device,
        dtype=dtype,
    )
    if power.ndim == 1:
        power = power[None]
    if power.ndim != 2 or power.shape[1] != positive_frequency.numel():
        raise ValueError("psd must return [frequency] or [band, frequency]")
    if not bool(torch.all(torch.isfinite(power))) or bool(torch.any(power < 0)):
        raise ValueError("PSD values must be finite and non-negative")
    bands = int(power.shape[0])
    mean = _signal_band_values(
        mean_amplitude,
        bands,
        name="mean_amplitude",
        device=resolved_device,
        dtype=dtype,
    )
    deviation = _signal_band_values(
        standard_deviation,
        bands,
        name="standard_deviation",
        device=resolved_device,
        dtype=dtype,
    )
    if bool(torch.any(mean <= 0)) or bool(torch.any(deviation < 0)):
        raise ValueError("means must be positive and deviations non-negative")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(int(seed))
    spectral_shape = (bands, positive_frequency.numel())
    if fourier_sampling == "random_phase":
        phases = 2.0 * math.pi * torch.rand(
            spectral_shape,
            device=resolved_device,
            dtype=dtype,
            generator=generator,
        )
        positive_spectrum = torch.sqrt(power) * torch.exp(1j * phases)
    else:
        real = torch.randn(
            spectral_shape,
            device=resolved_device,
            dtype=dtype,
            generator=generator,
        )
        imaginary = torch.randn(
            spectral_shape,
            device=resolved_device,
            dtype=dtype,
            generator=generator,
        )
        positive_spectrum = torch.sqrt(0.5 * power) * (real + 1j * imaginary)
    if padded_count % 2 == 0:
        # ``.real`` aliases the complex destination. Newer PyTorch releases
        # reject the overlapping assignment even though the intended Nyquist
        # projection is unambiguous, so materialize the real component first.
        positive_spectrum[:, -1] = positive_spectrum[:, -1].real.clone()
    zero = torch.zeros(
        (bands, 1),
        device=resolved_device,
        dtype=positive_spectrum.dtype,
    )
    spectrum = torch.cat((zero, positive_spectrum), dim=1)
    realization = torch.fft.irfft(spectrum, n=padded_count, dim=1)
    realization = realization[:, crop_start : crop_start + sample_count]
    realization = realization - realization.mean(dim=1, keepdim=True)
    scale = realization.std(dim=1, keepdim=True).clamp_min(1.0e-30)
    realization = realization / scale
    if amplitude_transform == "lognormal":
        log_std = torch.sqrt(torch.log1p((deviation / mean).square()))
        log_mean = torch.log(mean) - 0.5 * log_std.square()
        values = torch.exp(log_mean[:, None] + log_std[:, None] * realization)
    else:
        values = mean[:, None] + deviation[:, None] * realization
        if bool(torch.any(values < 0)):
            raise ValueError(
                "linear PSD realization became negative. Lower the deviation "
                "or use amplitude_transform='lognormal'"
            )
    psd_metadata = psd.metadata() if hasattr(psd, "metadata") else {
        "psd": getattr(psd, "__name__", type(psd).__name__)
    }
    return TabulatedDrivingSignal(
        times,
        values.transpose(0, 1).contiguous(),
        extrapolation=extrapolation,
        name=name,
        user_metadata={
            **dict(psd_metadata),
            "synthesis": "inverse_real_fft",
            "fourier_sampling": fourier_sampling,
            "amplitude_transform": amplitude_transform,
            "padding_factor": padding_factor,
            "crop_start_samples": crop_start,
            "seed": seed,
        },
    )


def broken_power_law_driving_signal(
    times_days: torch.Tensor | Sequence[float],
    *,
    break_timescale_days: float = 200.0,
    alpha_L: float = 1.0,
    alpha_R: float = 3.0,
    mean_amplitude: float | Sequence[float] | torch.Tensor = 1.0,
    standard_deviation: float | Sequence[float] | torch.Tensor = 0.3,
    seed: int | None = None,
    extrapolation: str = "error",
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> TabulatedDrivingSignal:
    """Generate a padded lognormal broken-power-law driving signal.

    ``alpha_L`` and ``alpha_R`` are the positive low- and high-frequency PSD
    slopes on the two sides of the break.
    """

    psd = BrokenPowerLawPSD(
        break_timescale_days=break_timescale_days,
        low_frequency_slope=alpha_L,
        high_frequency_slope=alpha_R,
    )
    sample_count = int(torch.as_tensor(times_days).numel())
    return driving_signal_from_psd(
        times_days,
        psd,
        mean_amplitude=mean_amplitude,
        standard_deviation=standard_deviation,
        padding_factor=5,
        crop_start_samples=sample_count,
        fourier_sampling="random_phase",
        amplitude_transform="lognormal",
        seed=seed,
        extrapolation=extrapolation,
        name="broken_power_law_driving_signal",
        dtype=dtype,
        device=device,
    )


def lognormal_damped_random_walk(
    times_days: torch.Tensor | Sequence[float],
    *,
    damping_timescale_days: float = 200.0,
    asymptotic_log_std: float = 0.1,
    mean_amplitude: float = 1.0,
    seed: int | None = None,
    extrapolation: str = "error",
    name: str = "lognormal_damped_random_walk",
) -> TabulatedDrivingSignal:
    """Generate a positive, irregular-cadence damped-random-walk driver.

    The latent Gaussian process uses the exact Ornstein--Uhlenbeck transition
    for each time separation. Exponentiation with the ``-sigma**2/2`` shift
    preserves ``mean_amplitude`` in expectation and makes the result directly
    usable as a multiplicative :class:`DrivingSignal`.
    """

    times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1).cpu()
    if times.numel() < 1 or not bool(torch.all(torch.isfinite(times))):
        raise ValueError("times_days must be a non-empty finite time axis")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times_days must be strictly increasing")
    if damping_timescale_days <= 0 or not math.isfinite(damping_timescale_days):
        raise ValueError("damping_timescale_days must be finite and positive")
    if asymptotic_log_std < 0 or not math.isfinite(asymptotic_log_std):
        raise ValueError("asymptotic_log_std must be finite and non-negative")
    if mean_amplitude <= 0 or not math.isfinite(mean_amplitude):
        raise ValueError("mean_amplitude must be finite and positive")
    if seed is None:
        normal = torch.randn(times.numel(), dtype=torch.float64)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        normal = torch.randn(times.numel(), dtype=torch.float64, generator=generator)
    latent = torch.empty_like(times)
    sigma = float(asymptotic_log_std)
    latent[0] = sigma * normal[0]
    for index in range(1, times.numel()):
        transition = math.exp(
            -float(times[index] - times[index - 1]) / damping_timescale_days
        )
        innovation = sigma * math.sqrt(max(0.0, 1.0 - transition**2))
        latent[index] = transition * latent[index - 1] + innovation * normal[index]
    values = float(mean_amplitude) * torch.exp(latent - 0.5 * sigma**2)
    return TabulatedDrivingSignal(
        times,
        values,
        extrapolation=extrapolation,
        name=name,
    )


@dataclass(frozen=True)
class ModulatedSource:
    """Multiply any pixelated source by an independent driving signal."""

    source: PixelatedSource
    signal: DrivingSignal
    name: str = "modulated"
    is_time_static: bool = False

    @property
    def geometry(self) -> SourceGeometry:
        """Spatial and spectral geometry inherited from the wrapped source."""

        return self.source.geometry

    def support_radius_m(self, distances) -> float | None:
        """Forward an optional physical support radius from the base source."""

        method = getattr(self.source, "support_radius_m", None)
        return None if method is None else float(method(distances))

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Evaluate the base source and apply band-dependent amplitudes."""

        times = _as_times(times_days)
        device = times.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        brightness = self.source.brightness(times, device=device, dtype=dtype)
        amplitudes = self.signal.amplitudes(
            times,
            bands=len(self.geometry.band_names),
            device=device,
            dtype=dtype,
        )
        return brightness * amplitudes[:, None, None, :]

    def metadata(self) -> Mapping[str, object]:
        """Return nested source and driving-signal provenance."""

        return {
            "type": "modulated",
            "name": self.name,
            "source": dict(self.source.metadata()),
            "signal": dict(self.signal.metadata()),
            "is_time_static": False,
        }


@dataclass(frozen=True)
class DelayedModulatedSource:
    """Apply a driving signal at a different retarded time in every pixel.

    This is a generic delayed-modulation adapter, not a hard-coded thermal
    reprocessing prescription. It is suitable for user response models and
    for approximations where a static source is multiplied by
    ``signal(t - delay[y, x])``. A physical temperature-response disk can
    implement :class:`PixelatedSource` directly while reusing the same delay
    map and driving-signal protocol.
    """

    source: PixelatedSource
    signal: DrivingSignal
    delay_days: torch.Tensor
    valid: torch.Tensor | None = None
    name: str = "delayed_modulated"
    is_time_static: bool = False

    def __post_init__(self) -> None:
        delay = torch.as_tensor(self.delay_days)
        if delay.shape != self.source.geometry.shape or not delay.is_floating_point():
            raise ValueError("delay_days must be one floating [y, x] source map")
        if self.valid is None:
            valid = torch.isfinite(delay)
        else:
            valid = torch.as_tensor(self.valid, device=delay.device, dtype=torch.bool)
            if valid.shape != delay.shape:
                raise ValueError("valid must match delay_days")
        if bool(torch.any(~torch.isfinite(delay[valid]))):
            raise ValueError("delays must be finite on valid source pixels")
        object.__setattr__(self, "delay_days", delay)
        object.__setattr__(self, "valid", valid)

    @classmethod
    def from_observer_transfer(
        cls,
        source: PixelatedSource,
        signal: DrivingSignal,
        transfer,
        **kwargs,
    ) -> DelayedModulatedSource:
        """Construct delayed modulation from an ``ObserverTransfer``."""

        if transfer.relative_delay_days is None:
            raise ValueError("observer transfer does not contain a delay map")
        return cls(
            source,
            signal,
            transfer.relative_delay_days,
            valid=transfer.hit,
            **kwargs,
        )

    @property
    def geometry(self) -> SourceGeometry:
        """Spatial and spectral geometry inherited from the wrapped source."""

        return self.source.geometry

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Evaluate pixelwise retarded modulation in one batched call."""

        times = _as_times(times_days)
        device = times.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        times = times.to(device=device, dtype=dtype)
        base = self.source.brightness(times, device=device, dtype=dtype)
        delay = self.delay_days.to(device=device, dtype=dtype)
        valid = self.valid.to(device=device)
        safe_delay = torch.where(valid, delay, torch.zeros_like(delay))
        query = times[:, None, None] - safe_delay[None]
        bands = len(self.geometry.band_names)
        amplitudes = self.signal.amplitudes(
            query.reshape(-1),
            bands=bands,
            device=device,
            dtype=dtype,
        ).reshape(times.numel(), *self.geometry.shape, bands)
        amplitudes = torch.where(
            valid[None, :, :, None],
            amplitudes,
            torch.zeros_like(amplitudes),
        )
        return base * amplitudes

    def metadata(self) -> Mapping[str, object]:
        """Return delayed source and signal provenance without pixel arrays."""

        return {
            "type": "delayed_modulated",
            "name": self.name,
            "source": dict(self.source.metadata()),
            "signal": dict(self.signal.metadata()),
            "valid_delay_pixels": int(self.valid.sum().detach().cpu()),
            "is_time_static": False,
        }
