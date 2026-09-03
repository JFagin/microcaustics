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
from .results import LightCurve, LightCurveLabels, MagnificationMap

_LABEL_ARRAYS = (
    "times_days",
    "crossing_labels",
    "crossing_events",
    "center_distances_uas",
    "center_distance_censored",
)


def _json_text(metadata) -> str:
    """Return deterministic JSON while tolerating descriptive custom values."""

    return json.dumps(dict(metadata), sort_keys=True, default=str)


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
    unlensed = (
        np.asarray([], dtype=np.float32)
        if light_curve.unlensed_flux is None
        else light_curve.unlensed_flux.detach().cpu().numpy()
    )
    np.savez_compressed(
        destination,
        schema_version=np.asarray(2),
        has_labels=np.asarray(light_curve.labels is not None),
        times_days=light_curve.times_days.detach().cpu().numpy(),
        flux=light_curve.flux.detach().cpu().numpy(),
        unlensed_flux=unlensed,
        has_unlensed=np.asarray(light_curve.unlensed_flux is not None),
        band_names=np.asarray(light_curve.band_names),
        metadata_json=np.asarray(_json_text(light_curve.metadata)),
        **(
            {
                f"labels_{name}": getattr(light_curve.labels, name)
                .detach()
                .cpu()
                .numpy()
                for name in _LABEL_ARRAYS
            }
            if light_curve.labels is not None
            else {}
        ),
    )
    return destination


def load_light_curve(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LightCurve:
    """Load photometry and optional center labels onto the requested device.

    Older photometry-only archives remain readable. Retained maps and full
    caustic geometry are separate products, not reconstructed from label arrays.
    """

    with np.load(Path(path), allow_pickle=False) as stored:
        if int(stored.get("schema_version", 1)) not in (1, 2):
            raise ValueError("unsupported light-curve archive schema_version")
        labels = None
        if bool(stored.get("has_labels", False)):
            missing = [name for name in _LABEL_ARRAYS if f"labels_{name}" not in stored]
            if missing:
                raise ValueError(
                    f"light-curve archive is missing label arrays {missing}"
                )
            labels = LightCurveLabels(
                **{
                    name: torch.from_numpy(stored[f"labels_{name}"].copy()).to(device)
                    for name in _LABEL_ARRAYS
                }
            )
        unlensed = None
        if bool(stored["has_unlensed"]):
            unlensed = torch.from_numpy(stored["unlensed_flux"].copy()).to(device)
        return LightCurve(
            times_days=torch.from_numpy(stored["times_days"].copy()).to(device),
            flux=torch.from_numpy(stored["flux"].copy()).to(device),
            band_names=tuple(str(value) for value in stored["band_names"]),
            unlensed_flux=unlensed,
            metadata=json.loads(str(stored["metadata_json"])),
            labels=labels,
        )
