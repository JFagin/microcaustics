"""Composable intrinsic variability for arbitrary pixelated sources."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from .base import PixelatedSource, SourceGeometry, _as_times, _geometry_grid

if TYPE_CHECKING:
    from .physical import PhysicalSourceModel

_DRIVER_SAMPLE_LOCK = Lock()


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
    """Linearly interpolate a sampled scalar or multiband driving signal.

    Treat the sample tensors as immutable. Device/dtype copies are cached for
    repeated source evaluation. Gradient-bearing tables are never cached.
    """

    times_days: torch.Tensor
    values: torch.Tensor
    extrapolation: str = "error"
    name: str = "tabulated"
    user_metadata: Mapping[str, object] | None = None
    _device_tables: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

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
        if not bool(torch.all(torch.isfinite(times))):
            raise ValueError("times_days must be finite")
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
        if not bool(torch.all(torch.isfinite(query))):
            raise ValueError("driving-signal query times must be finite")
        return self._interpolate(query, bands=bands, check_bounds=True)

    def _table_for(self, query):
        """Reuse immutable sample tables, including across independent CUDA streams."""

        if self.times_days.requires_grad or self.values.requires_grad:
            return self.times_days.to(query), self.values.to(query)
        key = (query.device, query.dtype)
        entry = self._device_tables.get(key)
        if entry is None:
            times, values = self.times_days.to(query), self.values.to(query)
            ready = None
            if query.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(query.device))
            entry = (times, values, ready)
            self._device_tables[key] = entry
        times, values, ready = entry
        if ready is not None:
            torch.cuda.current_stream(query.device).wait_event(ready)
        return times, values

    def _interpolate(self, query, *, bands, check_bounds):
        """Interpolate a table after the caller's finite-time and horizon checks."""

        times, values = self._table_for(query)
        if values.shape[1] not in (1, bands):
            raise ValueError("tabulated signal band count does not match the source")
        if (
            check_bounds
            and self.extrapolation == "error"
            and bool(torch.any((query < times[0]) | (query > times[-1])))
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
            result = values[left] + fraction[:, None] * (values[right] - values[left])
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
    # Long fractional-day float32 grids have quantized adjacent differences.
    # Validate the grid against a line, not against its first subtraction.
    spacing = float((times[-1] - times[0]) / (times.numel() - 1))
    regular = (
        times[0]
        + torch.arange(times.numel(), device=resolved_device, dtype=dtype) * spacing
    )
    tolerance = max(1.0e-10, 4 * torch.finfo(dtype).eps * float(times.abs().max()))
    if bool(torch.any(intervals <= 0)) or not torch.allclose(
        times, regular, rtol=0, atol=tolerance
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
        d=spacing,
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
        phases = (
            2.0
            * math.pi
            * torch.rand(
                spectral_shape,
                device=resolved_device,
                dtype=dtype,
                generator=generator,
            )
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
    psd_metadata = (
        psd.metadata()
        if hasattr(psd, "metadata")
        else {"psd": getattr(psd, "__name__", type(psd).__name__)}
    )
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
            "mean_amplitude": mean.detach().cpu().tolist(),
        },
    )


@dataclass(frozen=True)
class _FixedHorizonDrivingSignal:
    """Lazy fixed-grid PSD realization, bound to a system seed before use."""

    cadence_days: float
    max_duration_days: float
    history_days: float
    padding_factor: int
    psd: BrokenPowerLawPSD
    mean_amplitude: object
    standard_deviation: object
    seed: int | None
    dtype: torch.dtype
    device: torch.device | str
    _sampled: TabulatedDrivingSignal | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("cadence_days", "max_duration_days"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.history_days) or self.history_days < 0:
            raise ValueError("history_days must be finite and non-negative")
        if not isinstance(self.padding_factor, int) or self.padding_factor < 1:
            raise ValueError("padding_factor must be a positive integer")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("PSD synthesis requires float32 or float64")
        mean = torch.as_tensor(self.mean_amplitude)
        deviation = torch.as_tensor(self.standard_deviation)
        if not bool(torch.all(torch.isfinite(mean) & (mean > 0))):
            raise ValueError("mean_amplitude must be finite and positive")
        if not bool(torch.all(torch.isfinite(deviation) & (deviation >= 0))):
            raise ValueError("standard_deviation must be finite and non-negative")

    def with_seed(self, seed: int | None):
        """Bind an inherited seed without sampling or changing an explicit seed."""

        return replace(self, seed=self.seed if self.seed is not None else seed)

    @property
    def _samples(self) -> TabulatedDrivingSignal:
        """Generate once even when concurrent curves share an unseeded driver."""

        if self._sampled is None:
            with _DRIVER_SAMPLE_LOCK:
                if self._sampled is None:
                    object.__setattr__(self, "_sampled", self._generate_samples())
        return self._sampled

    def _generate_samples(self) -> TabulatedDrivingSignal:
        """Synthesize the padded fixed horizon independently of query times."""

        count = (
            math.ceil((self.max_duration_days + self.history_days) / self.cadence_days)
            + 1
        )
        times = (
            torch.arange(count, dtype=self.dtype, device=self.device)
            * self.cadence_days
            - self.history_days
        )
        return driving_signal_from_psd(
            times,
            self.psd,
            mean_amplitude=self.mean_amplitude,
            standard_deviation=self.standard_deviation,
            padding_factor=self.padding_factor,
            crop_start_samples=count if self.padding_factor >= 2 else 0,
            fourier_sampling="random_phase",
            amplitude_transform="lognormal",
            seed=self.seed,
            dtype=self.dtype,
            device=self.device,
        )

    def amplitudes(self, times_days, *, bands, dtype, device) -> torch.Tensor:
        """Interpolate only within the declared horizon, never extend or hold it."""

        times = _as_times(times_days)
        if not bool(
            torch.all(
                torch.isfinite(times)
                & (times >= -self.history_days)
                & (times <= self.max_duration_days)
            )
        ):
            raise ValueError(
                f"Driving-signal queries must lie between {-self.history_days:g} "
                f"and {self.max_duration_days:g} days. Increase history_days for "
                "reverberation or arrival delays, or max_duration_days for longer "
                "curves. Changing this grid generates a different realization "
                "even with the same seed"
            )
        return self._samples._interpolate(
            times.to(dtype=dtype, device=device), bands=bands, check_bounds=False
        )

    def metadata(self) -> Mapping[str, object]:
        """Describe the fixed horizon without generating samples."""

        return {
            "type": "fixed_horizon_broken_power_law",
            **self.psd.metadata(),
            "cadence_days": self.cadence_days,
            "max_duration_days": self.max_duration_days,
            "history_days": self.history_days,
            "padding_factor": self.padding_factor,
            "seed": self.seed,
            "mean_amplitude": torch.as_tensor(self.mean_amplitude).tolist(),
        }


@dataclass(frozen=True)
class _ConstantDrivingSignal:
    """The mean heating or multiplicative amplitude of a disabled driver."""

    mean_amplitude: object = 1.0

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean_amplitude)
        if mean.ndim > 1 or not bool(torch.all(torch.isfinite(mean) & (mean >= 0))):
            raise ValueError(
                "driver mean_amplitude must be finite and non-negative, scalar or one value per band"
            )

    def amplitudes(self, times_days, *, bands, dtype, device) -> torch.Tensor:
        """Broadcast the baseline without synthesizing a stochastic signal."""

        values = torch.as_tensor(
            self.mean_amplitude, dtype=dtype, device=device
        ).reshape(1, -1)
        if values.shape[1] not in (1, bands):
            raise ValueError("driver mean_amplitude band count must match the source")
        return values.expand(_as_times(times_days).numel(), bands)

    def metadata(self) -> Mapping[str, object]:
        """Identify a mean-amplitude calculation with variability disabled."""

        return {
            "type": "constant",
            "mean_amplitude": torch.as_tensor(self.mean_amplitude).tolist(),
        }


def _source_driving_signal(source):
    """Find the explicit driver on built-in physical or wrapped sources."""

    if source is None:
        return None
    signal = getattr(source, "driving_signal", None)
    if signal is None:
        signal = getattr(source, "signal", None)
    return (
        signal
        if signal is not None
        else _source_driving_signal(getattr(source, "source", None))
    )


def _source_with_signal(source, signal):
    """Bind an existing source driver without adding implicit modulation."""

    if _source_driving_signal(source) is signal:
        return source
    if hasattr(source, "with_driving_signal"):
        return source.with_driving_signal(signal)
    if hasattr(source, "signal"):
        return replace(source, signal=signal)
    if _source_driving_signal(getattr(source, "source", None)) is not None:
        return replace(source, source=_source_with_signal(source.source, signal))
    raise TypeError(
        "a custom driven source must implement with_driving_signal(signal) "
        "to bind its driver. Use ModulatedSource for explicit brightness modulation."
    )


def _validate_source_driver(source, apply_driving_signal=None):
    """Validate source-owned driving before realization or stochastic sampling."""

    if apply_driving_signal is not None and not isinstance(apply_driving_signal, bool):
        raise TypeError("apply_driving_signal must be True, False, or None")
    signal = _source_driving_signal(source)
    if signal is not None and not (
        callable(getattr(signal, "amplitudes", None))
        and callable(getattr(signal, "metadata", None))
    ):
        raise TypeError(
            "a source driving signal must provide amplitudes and metadata methods"
        )
    if apply_driving_signal is True and signal is None:
        raise ValueError(
            "apply_driving_signal=True requires a source with a configured driving signal. "
            "Attach a driver to the source or omit apply_driving_signal."
        )


def _source_at_driver_mean(source):
    """Disable driver fluctuations without removing heating or source evolution.

    Custom multiplicative signals use unit baseline unless their metadata
    supplies ``mean_amplitude``. This never samples a stochastic driver.
    """

    from .reprocessing import ThermalReprocessingSource

    signal = _source_driving_signal(source)
    if signal is None:
        return source
    mean = signal.metadata().get("mean_amplitude", 1.0)
    if isinstance(source, ThermalReprocessingSource):
        return replace(source, signal=_ConstantDrivingSignal(mean), is_time_static=True)
    if isinstance(source, (ModulatedSource, DelayedModulatedSource)):
        return replace(
            source,
            signal=_ConstantDrivingSignal(mean),
            is_time_static=source.source.is_time_static,
        )
    if isinstance(source, TimeShiftedSource):
        return replace(source, source=_source_at_driver_mean(source.source))
    if hasattr(source, "with_driving_signal"):
        return source.with_driving_signal(_ConstantDrivingSignal(mean))
    raise TypeError(
        "apply_driving_signal=False requires a built-in driven source, "
        "a custom source with with_driving_signal(signal), or explicit ModulatedSource wrapping"
    )


def broken_power_law_driving_signal(
    times_days: torch.Tensor | Sequence[float] | None = None,
    *,
    cadence_days: float | None = None,
    max_duration_days: float | None = None,
    history_days: float | None = None,
    padding_factor: int = 5,
    break_timescale_days: float = 200.0,
    alpha_L: float = 1.0,
    alpha_R: float = 3.0,
    mean_amplitude: float | Sequence[float] | torch.Tensor = 1.0,
    standard_deviation: float | Sequence[float] | torch.Tensor = 0.3,
    seed: int | None = None,
    extrapolation: str = "error",
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> DrivingSignal:
    """Define a reproducible padded lognormal broken-power-law driver.

    ``alpha_L`` and ``alpha_R`` are the positive low- and high-frequency PSD
    slopes on the two sides of the break. Without explicit times, the signal
    uses a fixed grid from ``-history_days`` to ``max_duration_days`` with
    default 0.1-day cadence and float32 synthesis. A system binds its variability
    seed unless ``seed`` overrides it. Sampling is lazy and independent of the
    duration requested for any light curve. Padding synthesizes a longer random
    series and crops it, not a repetition of the same samples.

    Queries outside the fixed horizon raise an error. Changing the horizon,
    cadence, history, padding, or PSD may change the whole realization.
    Explicit regular times remain available for standalone sampled signals
    and use float64 unless a dtype is selected.
    """

    psd = BrokenPowerLawPSD(
        break_timescale_days=break_timescale_days,
        low_frequency_slope=alpha_L,
        high_frequency_slope=alpha_R,
    )
    if times_days is None:
        if extrapolation != "error":
            raise ValueError(
                "fixed-horizon drivers do not extrapolate or hold endpoints"
            )
        return _FixedHorizonDrivingSignal(
            0.1 if cadence_days is None else cadence_days,
            7300.0 if max_duration_days is None else max_duration_days,
            1000.0 if history_days is None else history_days,
            padding_factor,
            psd,
            mean_amplitude,
            standard_deviation,
            seed,
            torch.float32 if dtype is None else dtype,
            device,
        )
    if any(
        value is not None for value in (cadence_days, max_duration_days, history_days)
    ):
        raise ValueError(
            "supply explicit times_days or cadence_days/max_duration_days/history_days, not both"
        )
    sample_count = int(torch.as_tensor(times_days).numel())
    return driving_signal_from_psd(
        times_days,
        psd,
        mean_amplitude=mean_amplitude,
        standard_deviation=standard_deviation,
        padding_factor=padding_factor,
        crop_start_samples=sample_count if padding_factor >= 2 else 0,
        fourier_sampling="random_phase",
        amplitude_transform="lognormal",
        seed=seed,
        extrapolation=extrapolation,
        name="broken_power_law_driving_signal",
        dtype=torch.float64 if dtype is None else dtype,
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
    """Multiply a physical model or pixelated source by a driving signal.

    Physical models are pixelated automatically by the system. Modulation is
    spatially coherent within each band, not a thermal reverberation model.
    """

    source: PixelatedSource | PhysicalSourceModel
    signal: DrivingSignal
    name: str = "modulated"
    is_time_static: bool = False

    def __post_init__(self) -> None:
        _validate_source_driver(self, True)

    @property
    def geometry(self) -> SourceGeometry:
        """Spatial and spectral geometry inherited from the wrapped source."""

        return self.source.geometry

    def recommended_grid(self, distances, policy=None):
        """Forward automatic source sizing without evaluating its brightness."""
        method = getattr(self.source, "recommended_grid", None)
        return (
            _geometry_grid(self.source.geometry, distances)
            if method is None
            else method(distances, policy)
        )

    def pixelate(
        self,
        distances=None,
        *,
        source_redshift=None,
        H0=None,
        Om0=None,
        grid=None,
        policy=None,
        runtime=None,
    ):
        """Resolve the base source once and retain the same driving signal."""
        from .physical import _pixelate_source, _resolve_source_distances

        distances = _resolve_source_distances(
            distances,
            source_redshift=source_redshift,
            model_redshift=getattr(self.source, "source_redshift", None),
            H0=H0,
            Om0=Om0,
            runtime=runtime,
        )

        method = getattr(self.source, "pixelate", None)
        if method is None:
            if grid is not None and any(
                actual < expected * (1 - 1e-6)
                for actual, expected in zip(
                    grid.field_of_view_uas,
                    self.recommended_grid(distances).field_of_view_uas,
                    strict=True,
                )
            ):
                raise ValueError(
                    "source_grid field of view must enclose the pixelated source geometry"
                )
            return self
        resolved = _pixelate_source(
            self.source, distances, grid=grid, policy=policy, runtime=runtime
        )
        return self if resolved is self.source else replace(self, source=resolved)

    def support_radius_m(self, distances) -> float | None:
        """Forward an optional physical support radius from the base source."""

        method = getattr(self.source, "support_radius_m", None)
        value = None if method is None else method(distances)
        return None if value is None else float(value)

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
            "is_time_static": bool(self.is_time_static),
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
            "is_time_static": bool(self.is_time_static),
        }
