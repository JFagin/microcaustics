"""Shared scheduling and option resolution for high-level workflows."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..config import (
    CausticConfig,
    DynamicConfig,
    IPMConfig,
    IRSConfig,
    _production_static_ipm_config,
    production_dynamic_config,
    production_ipm_config,
)

if TYPE_CHECKING:
    from ..results import MagnificationMap


def _retaining_map_observer(
    times_days: Sequence[float],
    keep_maps_at_days: Sequence[float] | None,
    map_observer,
):
    """Compose optional map retention with an existing streaming observer."""

    retained: dict[float, MagnificationMap] = {}
    if keep_maps_at_days is None:
        return map_observer, retained
    times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
    requested = tuple(float(value) for value in keep_maps_at_days)
    indices: dict[int, float] = {}
    for requested_time in requested:
        if not math.isfinite(requested_time):
            raise ValueError("keep_maps_at_days must contain finite times in days")
        differences = torch.abs(times - requested_time)
        index = int(torch.argmin(differences))
        tolerance = max(1.0e-6, 1.0e-8 * max(1.0, abs(requested_time)))
        if float(differences[index]) > tolerance:
            nearby = sorted(times[torch.argsort(differences)[:2]].tolist())
            warnings.warn(
                f"Requested map at day {requested_time:g} is not an evaluated map "
                f"epoch. Nearby evaluated times are {nearby}. This retention "
                "request will be omitted. Change keep_maps_at_days or the map "
                "cadence to retain that epoch",
                UserWarning,
                stacklevel=3,
            )
            continue
        indices[index] = float(times[index])

    def observer(index, frame):
        if map_observer is not None:
            map_observer(index, frame)
        if index in indices:
            retained[indices[index]] = getattr(frame, "magnification_map", frame)

    def reset():
        retained.clear()
        reset_callback = getattr(map_observer, "reset", None)
        if callable(reset_callback):
            reset_callback()

    observer.reset = reset
    return observer, retained


def _cadence_times(
    times_days,
    *,
    duration_days: float | None,
    cadence_days: float | None,
    start_day: float = 0.0,
) -> torch.Tensor:
    """Resolve an explicit time axis or an inclusive regular cadence."""

    if times_days is not None:
        if duration_days is not None or cadence_days is not None:
            raise ValueError(
                "supply times_days or duration_days/cadence_days, not both"
            )
        times = torch.as_tensor(times_days, dtype=torch.float64)
        if times.ndim != 1:
            raise ValueError("times_days must be a one-dimensional time axis")
    else:
        if duration_days is None or cadence_days is None:
            raise ValueError("supply times_days or both duration_days and cadence_days")
        duration = float(duration_days)
        cadence = float(cadence_days)
        if (
            not all(
                math.isfinite(value) for value in (duration, cadence, float(start_day))
            )
            or duration < 0.0
            or cadence <= 0.0
        ):
            raise ValueError(
                "duration_days must be non-negative and cadence_days positive"
            )
        count = int(math.floor(duration / cadence + 1.0e-10)) + 1
        times = float(start_day) + torch.arange(count, dtype=torch.float64) * cadence
        final = float(start_day) + duration
        if float(times[-1]) < final - 1.0e-10:
            times = torch.cat((times, torch.tensor((final,), dtype=torch.float64)))
    if times.numel() < 1 or not bool(torch.all(torch.isfinite(times))):
        raise ValueError("time axis must contain finite values")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("time axis must be strictly increasing")
    return times


def _with_method_options(kwargs: dict, *, dynamic: bool) -> dict:
    """Resolve common plain solver keywords into an advanced configuration."""

    resolved = dict(kwargs)
    option_names = {
        "rays",
        "scout_ratio",
        "refinement",
        "virtual_refinement",
        "scout_dilation_cells",
        "far_field",
    }
    options = {
        name: resolved.pop(name) for name in tuple(resolved) if name in option_names
    }
    method = resolved.get("method")
    if isinstance(method, str):
        name = method.lower().replace("-", "_")
        rays = int(options.pop("rays", 10_000_000))
        if name == "ipm":
            method = (
                production_ipm_config(rays=rays)
                if dynamic
                else _production_static_ipm_config(rays=rays)
            )
        elif name == "irs":
            if options:
                raise ValueError("scout/refinement options apply only to method='ipm'")
            method = IRSConfig(rays=rays)
        else:
            raise ValueError("method must be 'ipm', 'irs', or a config object")
        resolved["method"] = method
    elif isinstance(method, IRSConfig):
        rays = options.pop("rays", None)
        if options:
            raise ValueError("scout/refinement options apply only to IPM")
        if rays is not None:
            resolved["method"] = replace(method, rays=int(rays))
        return resolved
    if not options:
        return resolved
    method = resolved.get("method")
    if method is None:
        method = production_ipm_config() if dynamic else _production_static_ipm_config()
    if not isinstance(method, IPMConfig):
        raise ValueError("IPM numerical options require an IPM method")
    far_field = options.pop("far_field", None)
    if far_field is not None:
        options["far_field_approx"] = replace(
            method.far_field_approx,
            enabled=bool(far_field),
        )
    resolved["method"] = replace(method, **options)
    return resolved


_LIGHT_CURVE_CALL_OPTIONS = frozenset(
    {
        "source",
        "trajectory",
        "strict_coverage",
        "map_observer",
        "keep_maps_at_days",
    }
)
_LABELED_CURVE_CALL_OPTIONS = frozenset({"diagnostic_grid", "include_distance_map"})


def _light_curve_options(
    kwargs: dict, *, include_labels: bool, allowed_options=()
) -> dict:
    """Resolve the common call and warmup controls in one place.

    Plain options override advanced configurations explicitly supplied in the
    same call. Label batches inherit the temporal batch unless overridden.
    """

    options = dict(kwargs)
    updates = {
        name: options.pop(name)
        for name in (
            "temporal_batch_size",
            "scout_refresh_frames",
            "light_curve_batch_size",
        )
        if name in options
    }
    label_batch = options.pop("label_batch_size", None)
    schedule = options.get("schedule")
    if schedule is None:
        schedule = production_dynamic_config(
            temporal_batch_size=30 if include_labels else 49
        )
    options["schedule"] = replace(schedule, **updates) if updates else schedule
    if not include_labels and (
        label_batch is not None or options.get("caustics") is not None
    ):
        raise ValueError("caustic settings require include_labels=True")
    options = _with_method_options(options, dynamic=True)
    if options.get("method") is None:
        options["method"] = production_ipm_config()
    if not include_labels:
        options.pop("caustics", None)
    unknown = set(options) - {"method", "schedule", "caustics"} - set(allowed_options)
    if unknown:
        raise TypeError(f"unsupported light-curve options {sorted(unknown)}")
    if label_batch is not None:
        _, caustics = _production_dynamic_settings(
            options.get("method") or production_ipm_config(),
            options["schedule"],
            options.get("caustics"),
        )
        options["caustics"] = replace(caustics, temporal_batch_size=label_batch)
    return options


def _light_curve_times(
    times_days=None,
    *,
    duration_days=None,
    map_cadence_days=None,
    source_cadence_days=None,
    flux_times_days=None,
    start_day=0.0,
):
    """Resolve regular or irregular map and photometry epochs consistently."""

    map_times = _cadence_times(
        times_days,
        duration_days=duration_days,
        cadence_days=map_cadence_days,
        start_day=start_day,
    )
    if flux_times_days is not None and source_cadence_days is not None:
        raise ValueError("supply flux_times_days or source_cadence_days, not both")
    if flux_times_days is not None:
        flux_times = _cadence_times(
            flux_times_days, duration_days=None, cadence_days=None
        )
    elif source_cadence_days is not None:
        flux_times = _cadence_times(
            None,
            duration_days=float(map_times[-1] - map_times[0]),
            cadence_days=source_cadence_days,
            start_day=float(map_times[0]),
        )
    else:
        return map_times, None
    if float(flux_times[0]) < float(map_times[0]) or float(flux_times[-1]) > float(
        map_times[-1]
    ):
        raise ValueError(
            "flux_times_days must lie within the evaluated map time interval"
        )
    return map_times, None if torch.equal(map_times, flux_times) else flux_times


def _production_dynamic_settings(
    method: IPMConfig | IRSConfig,
    schedule: DynamicConfig | None,
    caustics: CausticConfig | None = None,
    *,
    include_labels: bool = True,
) -> tuple[DynamicConfig, CausticConfig | None]:
    """Resolve coherent high-level dynamic and optional caustic settings."""

    # The fused 8192-square determinant and label workload reaches its best
    # steady-state throughput with thirty frames per shared map/label batch.
    # Light-curve-only calls retain the forty-nine-frame production preset.
    resolved_schedule = (
        production_dynamic_config(temporal_batch_size=30 if include_labels else 49)
        if schedule is None
        else schedule
    )
    if not include_labels:
        return resolved_schedule, None
    if caustics is None:
        inherited_far_field = (
            method.far_field_approx if isinstance(method, IPMConfig) else None
        )
        resolved_caustics = CausticConfig(
            **(
                {"far_field_approx": inherited_far_field}
                if inherited_far_field is not None
                else {}
            ),
            temporal_batch_size=resolved_schedule.temporal_batch_size,
        )
    else:
        resolved_caustics = (
            replace(
                caustics,
                temporal_batch_size=resolved_schedule.temporal_batch_size,
            )
            if caustics.temporal_batch_size is None
            else caustics
        )
    return resolved_schedule, resolved_caustics


def _evaluate_light_curve(
    realization, map_times, flux_times, *, include_labels, **kwargs
):
    """Dispatch one resolved request without changing the numerical schedulers."""

    from ..results import _unified_light_curve

    if flux_times is None:
        calculate = (
            realization.light_curve_with_labels
            if include_labels
            else realization.light_curve
        )
        result = calculate(map_times, **kwargs)
    else:
        calculate = (
            realization.multirate_light_curve_with_labels
            if include_labels
            else realization.multirate_light_curve
        )
        result = calculate(map_times, flux_times, **kwargs)
    return _unified_light_curve(result)
