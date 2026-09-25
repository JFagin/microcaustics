"""Portable serialization for public simulation results.

The helpers in this module use compressed NumPy archives so saved products
can be inspected without a GPU or a PyTorch runtime. Tensor values, physical
grid geometry, method names, times, bands, and JSON-compatible provenance are
preserved. Runtime timing is intentionally excluded because loading a result
is not a numerical simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .geometry import PlaneGrid
from .results import (
    LightCurve,
    LightCurveLabels,
    MagnificationMap,
    TimeDependentSpectrum,
)

_LABEL_ARRAYS = (
    "times_days",
    "crossing_labels",
    "crossing_events",
    "center_distances_uas",
    "center_distance_censored",
)
_LIGHT_CURVE_SCHEMA_VERSION = 1


def _json_text(metadata) -> str:
    """Return deterministic JSON while tolerating descriptive custom values."""

    return json.dumps(dict(metadata), sort_keys=True, default=str)


def _light_curve_payload(light_curve: LightCurve) -> dict[str, np.ndarray]:
    """Convert one light curve to portable arrays used by every writer."""

    # NPZ cannot represent optional arrays without object/pickle storage. Keep
    # the archive portable by pairing an empty numeric sentinel with a flag.
    unlensed = (
        np.asarray([], dtype=np.float32)
        if light_curve.unlensed_flux is None
        else light_curve.unlensed_flux.detach().cpu().numpy()
    )
    microlensing_only = (
        np.asarray([], dtype=np.float32)
        if light_curve.microlensing_only_flux is None
        else light_curve.microlensing_only_flux.detach().cpu().numpy()
    )
    microlensing_only_unlensed = (
        np.asarray([], dtype=np.float32)
        if light_curve.microlensing_only_unlensed_flux is None
        else light_curve.microlensing_only_unlensed_flux.detach().cpu().numpy()
    )
    payload = {
        "schema_version": np.asarray(_LIGHT_CURVE_SCHEMA_VERSION),
        "has_labels": np.asarray(light_curve.labels is not None),
        "times_days": light_curve.times_days.detach().cpu().numpy(),
        "flux": light_curve.flux.detach().cpu().numpy(),
        "unlensed_flux": unlensed,
        "has_unlensed": np.asarray(light_curve.unlensed_flux is not None),
        "microlensing_only_flux": microlensing_only,
        "has_microlensing_only": np.asarray(
            light_curve.microlensing_only_flux is not None
        ),
        "microlensing_only_unlensed_flux": microlensing_only_unlensed,
        "has_microlensing_only_unlensed": np.asarray(
            light_curve.microlensing_only_unlensed_flux is not None
        ),
        "band_names": np.asarray(light_curve.band_names),
        "metadata_json": np.asarray(_json_text(light_curve.metadata)),
        "component_names": np.asarray(tuple(light_curve.component_flux)),
        "has_spectrum": np.asarray(light_curve.spectrum is not None),
    }
    for index, value in enumerate(light_curve.component_flux.values()):
        payload[f"component_{index:03d}_flux"] = value.detach().cpu().numpy()
    if light_curve.spectrum is not None:
        spectrum = light_curve.spectrum
        payload.update(
            {
                "spectrum_times_days": spectrum.times_days.detach().cpu().numpy(),
                "spectrum_wavelengths_angstrom": spectrum.wavelengths_angstrom.detach()
                .cpu()
                .numpy(),
                "spectrum_total_flux": spectrum.total_flux.detach().cpu().numpy(),
                "spectrum_continuum_flux": spectrum.continuum_flux.detach()
                .cpu()
                .numpy(),
                "spectrum_metadata_json": np.asarray(_json_text(spectrum.metadata)),
                "spectrum_component_names": np.asarray(tuple(spectrum.components)),
                "spectrum_has_microlensing_only": np.asarray(
                    spectrum.microlensing_only_continuum_flux is not None
                ),
                "spectrum_has_unlensed": np.asarray(
                    spectrum.unlensed_continuum_flux is not None
                ),
                "spectrum_microlensing_only_continuum_flux": (
                    np.asarray([], dtype=np.float32)
                    if spectrum.microlensing_only_continuum_flux is None
                    else spectrum.microlensing_only_continuum_flux.detach()
                    .cpu()
                    .numpy()
                ),
                "spectrum_unlensed_continuum_flux": (
                    np.asarray([], dtype=np.float32)
                    if spectrum.unlensed_continuum_flux is None
                    else spectrum.unlensed_continuum_flux.detach().cpu().numpy()
                ),
            }
        )
        for index, value in enumerate(spectrum.components.values()):
            payload[f"spectrum_component_{index:03d}_flux"] = (
                value.detach().cpu().numpy()
            )
    if light_curve.labels is not None:
        payload.update(
            {
                f"labels_{name}": getattr(light_curve.labels, name)
                .detach()
                .cpu()
                .numpy()
                for name in _LABEL_ARRAYS
            }
        )
    return payload


def _light_curve_from_payload(
    payload: dict[str, np.ndarray],
    *,
    device: str | torch.device = "cpu",
) -> LightCurve:
    """Reconstruct one light curve from arrays produced by the shared writer."""

    if int(payload.get("schema_version", -1)) != _LIGHT_CURVE_SCHEMA_VERSION:
        raise ValueError("unsupported light-curve archive schema_version")
    labels = None
    if bool(payload.get("has_labels", False)):
        missing = [name for name in _LABEL_ARRAYS if f"labels_{name}" not in payload]
        if missing:
            raise ValueError(f"light-curve archive is missing label arrays {missing}")
        labels = LightCurveLabels(
            **{
                name: torch.from_numpy(payload[f"labels_{name}"].copy()).to(device)
                for name in _LABEL_ARRAYS
            }
        )
    unlensed = None
    if bool(payload["has_unlensed"]):
        unlensed = torch.from_numpy(payload["unlensed_flux"].copy()).to(device)
    microlensing_only = None
    if bool(payload.get("has_microlensing_only", False)):
        microlensing_only = torch.from_numpy(
            payload["microlensing_only_flux"].copy()
        ).to(device)
    microlensing_only_unlensed = None
    if bool(payload.get("has_microlensing_only_unlensed", False)):
        microlensing_only_unlensed = torch.from_numpy(
            payload["microlensing_only_unlensed_flux"].copy()
        ).to(device)
    component_names = tuple(str(value) for value in payload.get("component_names", ()))
    component_flux = {
        name: torch.from_numpy(payload[f"component_{index:03d}_flux"].copy()).to(
            device
        )
        for index, name in enumerate(component_names)
    }
    spectrum = None
    if bool(payload.get("has_spectrum", False)):
        spectrum_names = tuple(
            str(value) for value in payload.get("spectrum_component_names", ())
        )
        spectrum_components = {
            name: torch.from_numpy(
                payload[f"spectrum_component_{index:03d}_flux"].copy()
            ).to(device)
            for index, name in enumerate(spectrum_names)
        }
        spectrum = TimeDependentSpectrum(
            times_days=torch.from_numpy(payload["spectrum_times_days"].copy()).to(
                device
            ),
            wavelengths_angstrom=torch.from_numpy(
                payload["spectrum_wavelengths_angstrom"].copy()
            ).to(device),
            total_flux=torch.from_numpy(payload["spectrum_total_flux"].copy()).to(
                device
            ),
            continuum_flux=torch.from_numpy(
                payload["spectrum_continuum_flux"].copy()
            ).to(device),
            microlensing_only_continuum_flux=(
                torch.from_numpy(
                    payload["spectrum_microlensing_only_continuum_flux"].copy()
                ).to(device)
                if bool(payload.get("spectrum_has_microlensing_only", False))
                else None
            ),
            unlensed_continuum_flux=(
                torch.from_numpy(
                    payload["spectrum_unlensed_continuum_flux"].copy()
                ).to(device)
                if bool(payload.get("spectrum_has_unlensed", False))
                else None
            ),
            components=spectrum_components,
            metadata=json.loads(str(payload["spectrum_metadata_json"])),
        )
    return LightCurve(
        times_days=torch.from_numpy(payload["times_days"].copy()).to(device),
        flux=torch.from_numpy(payload["flux"].copy()).to(device),
        band_names=tuple(str(value) for value in payload["band_names"]),
        unlensed_flux=unlensed,
        metadata=json.loads(str(payload["metadata_json"])),
        labels=labels,
        microlensing_only_flux=microlensing_only,
        microlensing_only_unlensed_flux=microlensing_only_unlensed,
        spectrum=spectrum,
        component_flux=component_flux,
    )


def save_magnification_map(
    magnification_map: MagnificationMap, path: str | Path
) -> Path:
    """Save one magnification map and its physical grid to a compressed NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        values=magnification_map.numpy(),
        shape=np.asarray(magnification_map.grid.shape, dtype=np.int64),
        field_of_view_uas=np.asarray(
            magnification_map.grid.field_of_view_uas, dtype=np.float64
        ),
        center_uas=np.asarray(magnification_map.grid.center_uas, dtype=np.float64),
        time_days=np.asarray(magnification_map.time_days, dtype=np.float64),
        method=np.asarray(magnification_map.method),
        metadata_json=np.asarray(_json_text(magnification_map.metadata)),
    )
    return destination


def load_magnification_map(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> MagnificationMap:
    """Load a map saved by :func:`save_magnification_map`."""

    with np.load(Path(path), allow_pickle=False) as stored:
        grid = PlaneGrid(
            tuple(int(value) for value in stored["shape"]),
            tuple(float(value) for value in stored["field_of_view_uas"]),
            tuple(float(value) for value in stored["center_uas"]),
        )
        values = torch.from_numpy(stored["values"].copy()).to(device=device)
        return MagnificationMap(
            values,
            grid,
            time_days=float(stored["time_days"]),
            method=str(stored["method"]),
            metadata=json.loads(str(stored["metadata_json"])),
        )


def save_light_curve(light_curve: LightCurve, path: str | Path) -> Path:
    """Save a multiband light curve in physical Jy units.

    Retained maps are separate products and can be saved with
    :func:`save_magnification_map`. This keeps light-curve archives compact.
    Source-center label arrays and their independent time axis are preserved.
    Full caustic geometry is not serialized and loads as ``labels.caustics=None``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **_light_curve_payload(light_curve))
    return destination


def load_light_curve(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LightCurve:
    """Load photometry and optional center labels onto the requested device.

    Retained maps and full caustic geometry are separate products, not
    reconstructed from label arrays.
    """

    with np.load(Path(path), allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    return _light_curve_from_payload(payload, device=device)
