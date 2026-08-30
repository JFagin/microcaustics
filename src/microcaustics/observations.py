"""Survey cadences and noisy observations of simulated resolved images."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .results import MultiImageLightCurves

RUBIN_GAMMA = {
    "u": 0.038,
    "g": 0.039,
    "r": 0.039,
    "i": 0.039,
    "z": 0.039,
    "y": 0.039,
}


def find_rubin_opsim_database(
    search_roots: Sequence[str | Path] | None = None,
    *,
    environment_variable: str = "MICROCAUSTICS_LSST_OPSIM",
    filename: str = "baseline_v4.3.5_10yrs.db",
) -> Path | None:
    """Find an optional Rubin OpSim database without notebook path plumbing.

    An explicit path in ``environment_variable`` has priority. Otherwise each
    search root and its parents are checked for ``filename``. The function is
    deliberately read-only and returns ``None`` when no database is present.
    """

    import os

    configured = os.environ.get(environment_variable)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    roots = (Path.cwd(),) if search_roots is None else tuple(Path(v) for v in search_roots)
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        candidates = (resolved, *resolved.parents)
        for parent in candidates:
            if parent in seen:
                continue
            seen.add(parent)
            candidate = parent / filename
            if candidate.is_file():
                return candidate
    return None


@dataclass(frozen=True)
class SurveyCadence:
    """One sequence of single-band survey visits.

    ``time_days``, ``band_names``, and ``five_sigma_depth`` contain one value
    per visit. MJD and seeing are optional metadata rather than requirements of
    the photometric noise model.
    """

    time_days: np.ndarray
    band_names: tuple[str, ...]
    five_sigma_depth: np.ndarray
    mjd: np.ndarray | None = None
    seeing_fwhm_arcsec: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = np.asarray(self.time_days, dtype=np.float64).reshape(-1)
        depths = np.asarray(self.five_sigma_depth, dtype=np.float64).reshape(-1)
        bands = tuple(str(value) for value in self.band_names)
        if times.size < 1 or depths.shape != times.shape or len(bands) != times.size:
            raise ValueError("cadence arrays must contain one value per visit")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(depths)):
            raise ValueError("cadence times and depths must be finite")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("cadence times must be non-decreasing")
        if any(not value for value in bands):
            raise ValueError("cadence band names must be non-empty")
        mjd = self._optional_array(self.mjd, times.shape, "mjd")
        seeing = self._optional_array(
            self.seeing_fwhm_arcsec,
            times.shape,
            "seeing_fwhm_arcsec",
        )
        object.__setattr__(self, "time_days", times)
        object.__setattr__(self, "band_names", bands)
        object.__setattr__(self, "five_sigma_depth", depths)
        object.__setattr__(self, "mjd", mjd)
        object.__setattr__(self, "seeing_fwhm_arcsec", seeing)

    @staticmethod
    def _optional_array(value, shape, name: str) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with one value per visit")
        return array

    @property
    def visit_count(self) -> int:
        """Return the number of survey visits."""

        return int(self.time_days.size)

    def select_bands(self, band_names: Sequence[str]) -> SurveyCadence:
        """Return visits whose filters occur in ``band_names``.

        This is useful when an OpSim database contains all Rubin filters but a
        tutorial or source model intentionally synthesizes only a subset.
        Visit ordering and optional MJD/seeing arrays are preserved.
        """

        selected_names = {str(value) for value in band_names}
        if not selected_names:
            raise ValueError("band_names must contain at least one filter")
        mask = np.asarray(
            [name in selected_names for name in self.band_names],
            dtype=bool,
        )
        if not np.any(mask):
            raise ValueError("no cadence visits match the requested filters")
        return SurveyCadence(
            self.time_days[mask],
            tuple(name for name, keep in zip(self.band_names, mask, strict=True) if keep),
            self.five_sigma_depth[mask],
            None if self.mjd is None else self.mjd[mask],
            None if self.seeing_fwhm_arcsec is None else self.seeing_fwhm_arcsec[mask],
            {
                **dict(self.metadata),
                "selected_bands": tuple(sorted(selected_names)),
                "parent_visit_count": self.visit_count,
            },
        )


@dataclass(frozen=True)
class PhotometricObservations:
    """Noisy and noiseless magnitudes for resolved macroimages.

    Arrays have shape ``[visit, image]`` because each survey visit observes one
    band, recorded by ``band_names``.
    """

    time_days: torch.Tensor
    band_names: tuple[str, ...]
    image_names: tuple[str, ...]
    magnitude: torch.Tensor
    magnitude_error: torch.Tensor
    noiseless_magnitude: torch.Tensor
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.time_days)
        magnitude = torch.as_tensor(self.magnitude)
        error = torch.as_tensor(self.magnitude_error)
        noiseless = torch.as_tensor(self.noiseless_magnitude)
        expected = (times.numel(), len(self.image_names))
        if times.ndim != 1 or tuple(magnitude.shape) != expected:
            raise ValueError("observation arrays must have shape [visit, image]")
        if error.shape != magnitude.shape or noiseless.shape != magnitude.shape:
            raise ValueError("all magnitude arrays must have the same shape")
        if len(self.band_names) != times.numel():
            raise ValueError("band_names must contain one value per visit")
        if bool(torch.any(error < 0)) or not bool(torch.all(torch.isfinite(error))):
            raise ValueError("magnitude errors must be finite and non-negative")
        object.__setattr__(self, "time_days", times)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "magnitude_error", error)
        object.__setattr__(self, "noiseless_magnitude", noiseless)


def rubin_magnitude_uncertainty(
    magnitude,
    five_sigma_depth,
    band_names: str | Sequence[str],
    *,
    gamma_by_band: Mapping[str, float] | None = None,
    systematic_floor_mag: float = 0.005,
) -> torch.Tensor:
    """Return the Rubin random-plus-systematic magnitude uncertainty.

    The random term follows ``(0.04-gamma)x + gamma*x**2`` with
    ``x=10**(0.4*(m-m5))``. The default gamma values are the standard
    ``u=0.038`` and ``grizy=0.039`` approximation used by the Rubin mock.
    """

    values = torch.as_tensor(magnitude)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    depth = torch.as_tensor(
        five_sigma_depth,
        device=values.device,
        dtype=values.dtype,
    )
    if isinstance(band_names, str):
        bands = (band_names,) * values.numel()
    else:
        bands = tuple(str(value) for value in band_names)
    if len(bands) != values.numel():
        raise ValueError("band_names must be scalar or match magnitude size")
    gamma_values = dict(RUBIN_GAMMA)
    gamma_values.update(gamma_by_band or {})
    try:
        gamma = torch.tensor(
            [gamma_values[band] for band in bands],
            device=values.device,
            dtype=values.dtype,
        ).reshape(values.shape)
    except KeyError as error:
        raise KeyError(
            f"no Rubin gamma value supplied for band {error.args[0]!r}"
        ) from error
    if systematic_floor_mag < 0 or not math.isfinite(systematic_floor_mag):
        raise ValueError("systematic_floor_mag must be finite and non-negative")
    x = torch.pow(values.new_tensor(10.0), 0.4 * (values - depth))
    random_variance = ((0.04 - gamma) * x + gamma * x.square()).clamp_min(0.0)
    return torch.sqrt(random_variance + float(systematic_floor_mag) ** 2)


def observe_multi_image_light_curves(
    curves: MultiImageLightCurves,
    cadence: SurveyCadence,
    *,
    zero_point_flux: float | Mapping[str, float] = 3631.0,
    seed: int | None = None,
    add_noise: bool = True,
    gamma_by_band: Mapping[str, float] | None = None,
    systematic_floor_mag: float = 0.005,
) -> PhotometricObservations:
    """Sample resolved light curves at survey visits and add Rubin-like noise.

    Source light curves are physical flux densities in Jy, so the default
    converts them directly to AB magnitudes using the 3631 Jy zero point.
    ``zero_point_flux`` is retained for explicitly calibrated non-AB systems.
    """

    if isinstance(zero_point_flux, Mapping):
        zero_points = {str(key): float(value) for key, value in zero_point_flux.items()}
    else:
        zero_points = {band: float(zero_point_flux) for band in set(cadence.band_names)}
    if any(not math.isfinite(value) or value <= 0 for value in zero_points.values()):
        raise ValueError("zero-point fluxes must be finite and positive")
    noiseless = np.empty((cadence.visit_count, len(curves.images)), dtype=np.float64)
    for image_index, image in enumerate(curves.images):
        curve = image.light_curve
        times = curve.times_days.detach().cpu().to(torch.float64).numpy()
        flux = curve.flux.detach().cpu().to(torch.float64).numpy()
        if cadence.time_days[0] < times[0] or cadence.time_days[-1] > times[-1]:
            raise ValueError("survey cadence lies outside a light-curve time axis")
        band_indices = {name: index for index, name in enumerate(curve.band_names)}
        for visit_index, (time, band) in enumerate(
            zip(cadence.time_days, cadence.band_names, strict=True)
        ):
            if band not in band_indices:
                raise KeyError(f"light curve does not contain survey band {band!r}")
            if band not in zero_points:
                raise KeyError(f"zero_point_flux does not contain band {band!r}")
            value = np.interp(time, times, flux[:, band_indices[band]])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("sampled fluxes must be finite and positive")
            noiseless[visit_index, image_index] = -2.5 * np.log10(
                value / zero_points[band]
            )

    noiseless_tensor = torch.from_numpy(noiseless)
    depth = torch.from_numpy(cadence.five_sigma_depth)[:, None].expand_as(
        noiseless_tensor
    )
    repeated_bands = tuple(
        band
        for band in cadence.band_names
        for _ in range(len(curves.images))
    )
    error = rubin_magnitude_uncertainty(
        noiseless_tensor,
        depth,
        repeated_bands,
        gamma_by_band=gamma_by_band,
        systematic_floor_mag=systematic_floor_mag,
    )
    if add_noise:
        if seed is None:
            noise = torch.randn(
                noiseless_tensor.shape,
                dtype=noiseless_tensor.dtype,
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            noise = torch.randn(
                noiseless_tensor.shape,
                dtype=noiseless_tensor.dtype,
                generator=generator,
            )
        magnitude = noiseless_tensor + noise * error
    else:
        magnitude = noiseless_tensor.clone()
    return PhotometricObservations(
        time_days=torch.from_numpy(cadence.time_days.copy()),
        band_names=cadence.band_names,
        image_names=curves.image_names,
        magnitude=magnitude,
        magnitude_error=error,
        noiseless_magnitude=noiseless_tensor,
        metadata={
            "noise_model": "rubin_random_plus_systematic",
            "systematic_floor_mag": float(systematic_floor_mag),
            "noise_added": bool(add_noise),
            "seed": seed,
            "zero_point_flux": zero_points,
        },
    )


def sample_random_rubin_wfd_cadence(
    database_path: str | Path,
    *,
    seed: int,
    radius_deg: float = 1.75,
    min_visits: int = 700,
    max_visits: int = 1250,
    duration_days: float = 3650.0,
    max_tries: int = 12,
) -> SurveyCadence:
    """Select a reproducible random WFD field from a Rubin OpSim database."""

    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if radius_deg <= 0 or min_visits < 1 or max_visits < min_visits:
        raise ValueError("invalid WFD field selection limits")
    rng = np.random.default_rng(seed)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        maximum_row = int(
            connection.execute("SELECT max(rowid) FROM observations").fetchone()[0]
        )
        best = None
        for _ in range(max_tries):
            start = int(rng.integers(1, maximum_row + 1))
            center = connection.execute(
                "SELECT rowid, fieldRA, fieldDec FROM observations "
                "WHERE rowid >= ? AND science_program != 'DD' LIMIT 1",
                (start,),
            ).fetchone()
            if center is None:
                continue
            row_id, center_ra, center_dec = center
            rows = _query_wfd_visits(
                connection,
                float(center_ra),
                float(center_dec),
                radius_deg,
            )
            count = int(rows.shape[0])
            score = abs(count - 0.5 * (min_visits + max_visits))
            candidate = (score, int(row_id), float(center_ra), float(center_dec), rows)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if min_visits <= count <= max_visits:
                break
    finally:
        connection.close()
    if best is None or best[-1].shape[0] == 0:
        raise RuntimeError("could not find a non-empty WFD field")

    _, row_id, center_ra, center_dec, rows = best
    order = np.argsort(rows[:, 3].astype(float), kind="stable")
    rows = rows[order]
    mjd = rows[:, 3].astype(np.float64)
    time_days = mjd - mjd.min()
    keep = time_days <= float(duration_days)
    return SurveyCadence(
        time_days=time_days[keep],
        band_names=tuple(rows[keep, 4].astype(str)),
        five_sigma_depth=rows[keep, 5].astype(np.float64),
        mjd=mjd[keep],
        seeing_fwhm_arcsec=rows[keep, 6].astype(np.float64),
        metadata={
            "survey": "Rubin WFD",
            "database": path.name,
            "selection_radius_deg": float(radius_deg),
            "field_ra_deg": center_ra,
            "field_dec_deg": center_dec,
            "source_rowid": row_id,
            "seed": int(seed),
        },
    )


def _query_wfd_visits(
    connection: sqlite3.Connection,
    center_ra: float,
    center_dec: float,
    radius_deg: float,
) -> np.ndarray:
    """Return non-deep-drilling visits in a circular sky field."""

    cos_dec = max(abs(float(np.cos(np.deg2rad(center_dec)))), 1.0e-3)
    half_ra = radius_deg / cos_dec
    ra_low, ra_high = center_ra - half_ra, center_ra + half_ra
    dec_low, dec_high = center_dec - radius_deg, center_dec + radius_deg
    if ra_low < 0.0:
        ra_clause, ra_args = "(fieldRA >= ? OR fieldRA <= ?)", (ra_low + 360, ra_high)
    elif ra_high >= 360.0:
        ra_clause, ra_args = "(fieldRA >= ? OR fieldRA <= ?)", (ra_low, ra_high - 360)
    else:
        ra_clause, ra_args = "fieldRA BETWEEN ? AND ?", (ra_low, ra_high)
    fields = (
        "rowid, fieldRA, fieldDec, observationStartMJD, band, "
        "fiveSigmaDepth, seeingFwhmEff"
    )
    rows = connection.execute(
        f"SELECT {fields} FROM observations WHERE science_program != 'DD' "
        f"AND fieldDec BETWEEN ? AND ? AND {ra_clause}",
        (dec_low, dec_high, *ra_args),
    ).fetchall()
    if not rows:
        return np.empty((0, 7), dtype=object)
    array = np.asarray(rows, dtype=object)
    ra = array[:, 1].astype(float)
    dec = array[:, 2].astype(float)
    delta_ra = ((ra - center_ra + 180.0) % 360.0 - 180.0) * np.cos(
        np.deg2rad(center_dec)
    )
    separation = np.hypot(delta_ra, dec - center_dec)
    return array[separation < radius_deg]
