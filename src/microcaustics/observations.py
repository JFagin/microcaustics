"""Survey cadences and noisy observations of simulated resolved images."""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .results import LightCurve, MultiImageLightCurves

RUBIN_GAMMA = {
    "u": 0.038,
    "g": 0.039,
    "r": 0.039,
    "i": 0.039,
    "z": 0.039,
    "y": 0.039,
}

_RUBIN_INDEX_CACHE: dict[tuple[Path, int, int, float], RubinOpSimCadenceIndex] = {}


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


class RubinOpSimCadenceIndex:
    """Reusable in-memory spatial index over one Rubin OpSim database.

    Constructing the index reads the OpSim ``observations`` table once. Calls
    to :meth:`at_sky_position` and :meth:`sample` then operate only on compact
    NumPy arrays; they do not reopen or rescan SQLite. Repeated
    :meth:`from_database` calls for an unchanged file reuse the same process
    cache entry.
    """

    _STANDARD_BANDS = ("u", "g", "r", "i", "z", "y")

    def __init__(
        self,
        *,
        database_path: Path,
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        mjd: np.ndarray,
        band_codes: np.ndarray,
        band_names: tuple[str, ...],
        five_sigma_depth: np.ndarray,
        seeing_fwhm_arcsec: np.ndarray,
        is_ddf: np.ndarray,
        ddf_codes: np.ndarray,
        ddf_fields: tuple[str, ...],
        bin_size_deg: float,
        build_seconds: float,
    ) -> None:
        self._database_path = database_path
        self._ra_deg = ra_deg
        self._dec_deg = dec_deg
        self._mjd = mjd
        self._band_codes = band_codes
        self._band_names = band_names
        self._five_sigma_depth = five_sigma_depth
        self._seeing_fwhm_arcsec = seeing_fwhm_arcsec
        self._is_ddf = is_ddf
        self._ddf_codes = ddf_codes
        self._ddf_fields = ddf_fields
        self._bin_size_deg = float(bin_size_deg)
        self._survey_start_mjd = float(np.min(mjd))
        self._build_seconds = float(build_seconds)

        self._n_ra_bins = int(math.ceil(360.0 / self._bin_size_deg))
        self._n_dec_bins = int(math.ceil(180.0 / self._bin_size_deg))
        ra_bins = np.floor((ra_deg % 360.0) / self._bin_size_deg).astype(np.int32)
        dec_bins = np.floor((dec_deg + 90.0) / self._bin_size_deg).astype(np.int32)
        np.clip(dec_bins, 0, self._n_dec_bins - 1, out=dec_bins)
        cell_ids = dec_bins * self._n_ra_bins + ra_bins
        self._cell_order = np.argsort(cell_ids, kind="stable").astype(
            np.int32, copy=False
        )
        counts = np.bincount(
            cell_ids,
            minlength=self._n_ra_bins * self._n_dec_bins,
        )
        self._cell_offsets = np.empty(counts.size + 1, dtype=np.int64)
        self._cell_offsets[0] = 0
        np.cumsum(counts, out=self._cell_offsets[1:])
        self._wfd_indices = np.flatnonzero(~is_ddf).astype(np.int32, copy=False)
        self._ddf_visit_count = int(np.count_nonzero(is_ddf))

        centers: list[tuple[float, float]] = []
        for code in range(len(ddf_fields)):
            selected = ddf_codes == code
            ra_rad = np.deg2rad(ra_deg[selected].astype(np.float64))
            dec_rad = np.deg2rad(dec_deg[selected].astype(np.float64))
            x = np.mean(np.cos(dec_rad) * np.cos(ra_rad))
            y = np.mean(np.cos(dec_rad) * np.sin(ra_rad))
            z = np.mean(np.sin(dec_rad))
            centers.append(
                (
                    float(np.rad2deg(np.arctan2(y, x)) % 360.0),
                    float(np.rad2deg(np.arctan2(z, np.hypot(x, y)))),
                )
            )
        self._ddf_centers = tuple(centers)

    @classmethod
    def from_database(
        cls,
        database_path: str | Path,
        *,
        bin_size_deg: float = 1.75,
    ) -> RubinOpSimCadenceIndex:
        """Load an OpSim database once and return its reusable spatial index."""

        path = Path(database_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not math.isfinite(bin_size_deg) or bin_size_deg <= 0.0:
            raise ValueError("bin_size_deg must be finite and positive")
        stat = path.stat()
        key = (path, int(stat.st_size), int(stat.st_mtime_ns), float(bin_size_deg))
        cached = _RUBIN_INDEX_CACHE.get(key)
        if cached is not None:
            return cached

        started = time.perf_counter()
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(observations)")
            }
            required = {
                "fieldRA",
                "fieldDec",
                "observationStartMJD",
                "fiveSigmaDepth",
            }
            missing = required - columns
            if missing:
                raise ValueError(
                    f"OpSim observations table is missing columns: {sorted(missing)}"
                )
            band_column = "band" if "band" in columns else "filter"
            if band_column not in columns:
                raise ValueError("OpSim observations table requires band or filter")
            seeing_column = (
                "seeingFwhmEff" if "seeingFwhmEff" in columns else "NULL"
            )
            program_column = (
                "science_program" if "science_program" in columns else "''"
            )
            target_column = "target_name" if "target_name" in columns else "''"
            row_count = int(
                connection.execute("SELECT count(*) FROM observations").fetchone()[0]
            )
            if row_count < 1:
                raise ValueError("OpSim observations table is empty")

            ra = np.empty(row_count, dtype=np.float32)
            dec = np.empty(row_count, dtype=np.float32)
            mjd = np.empty(row_count, dtype=np.float64)
            depth = np.empty(row_count, dtype=np.float32)
            seeing = np.empty(row_count, dtype=np.float32)
            band_codes = np.empty(row_count, dtype=np.uint8)
            is_ddf = np.empty(row_count, dtype=bool)
            ddf_codes = np.full(row_count, -1, dtype=np.int16)
            band_lookup = {name: index for index, name in enumerate(cls._STANDARD_BANDS)}
            ddf_lookup: dict[str, int] = {}
            cursor = connection.execute(
                "SELECT fieldRA, fieldDec, observationStartMJD, "
                f"{band_column}, fiveSigmaDepth, {seeing_column}, "
                f"{program_column}, {target_column} FROM observations"
            )
            offset = 0
            while True:
                rows = cursor.fetchmany(100_000)
                if not rows:
                    break
                array = np.asarray(rows, dtype=object)
                count = int(array.shape[0])
                selection = slice(offset, offset + count)
                ra[selection] = array[:, 0].astype(np.float32)
                dec[selection] = array[:, 1].astype(np.float32)
                mjd[selection] = array[:, 2].astype(np.float64)
                depth[selection] = array[:, 4].astype(np.float32)
                seeing_values = np.asarray(
                    [np.nan if value is None else value for value in array[:, 5]],
                    dtype=np.float32,
                )
                seeing[selection] = seeing_values
                bands = array[:, 3].astype(str)
                encoded_bands = np.empty(count, dtype=np.uint8)
                for name in np.unique(bands):
                    if name not in band_lookup:
                        if len(band_lookup) >= np.iinfo(np.uint8).max:
                            raise ValueError("OpSim database contains too many band names")
                        band_lookup[name] = len(band_lookup)
                    encoded_bands[bands == name] = band_lookup[name]
                band_codes[selection] = encoded_bands
                chunk_ddf = array[:, 6].astype(str) == "DD"
                is_ddf[selection] = chunk_ddf
                if np.any(chunk_ddf):
                    target_names = array[:, 7].astype(str)
                    encoded_fields = np.full(count, -1, dtype=np.int16)
                    for target in np.unique(target_names[chunk_ddf]):
                        name = cls._ddf_field_name(target)
                        if name not in ddf_lookup:
                            ddf_lookup[name] = len(ddf_lookup)
                        encoded_fields[chunk_ddf & (target_names == target)] = ddf_lookup[
                            name
                        ]
                    ddf_codes[selection] = encoded_fields
                offset += count
            if offset != row_count:
                raise RuntimeError("OpSim row count changed while building the index")
        finally:
            connection.close()

        names_by_code = tuple(
            name for name, _ in sorted(ddf_lookup.items(), key=lambda item: item[1])
        )
        bands_by_code = tuple(
            name for name, _ in sorted(band_lookup.items(), key=lambda item: item[1])
        )
        index = cls(
            database_path=path,
            ra_deg=ra,
            dec_deg=dec,
            mjd=mjd,
            band_codes=band_codes,
            band_names=bands_by_code,
            five_sigma_depth=depth,
            seeing_fwhm_arcsec=seeing,
            is_ddf=is_ddf,
            ddf_codes=ddf_codes,
            ddf_fields=names_by_code,
            bin_size_deg=bin_size_deg,
            build_seconds=0.0,
        )
        index._build_seconds = time.perf_counter() - started
        for stale_key in tuple(_RUBIN_INDEX_CACHE):
            if stale_key[0] == path:
                _RUBIN_INDEX_CACHE.pop(stale_key)
        _RUBIN_INDEX_CACHE[key] = index
        return index

    @staticmethod
    def _ddf_field_name(target_name: str) -> str:
        name = str(target_name).strip()
        if name.startswith("DD:"):
            name = name[3:]
        return name.split(",", 1)[0].strip() or "unnamed"

    @classmethod
    def clear_process_cache(cls) -> None:
        """Forget process-cached indexes without modifying any database."""

        _RUBIN_INDEX_CACHE.clear()

    @property
    def database_path(self) -> Path:
        """Return the indexed OpSim database path."""

        return self._database_path

    @property
    def survey_start_mjd(self) -> float:
        """Return the common MJD origin used for every selected cadence."""

        return self._survey_start_mjd

    @property
    def ddf_fields(self) -> tuple[str, ...]:
        """Return the available named deep-drilling fields."""

        return self._ddf_fields

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return database and index provenance."""

        return {
            "database": self._database_path.name,
            "visit_count": int(self._mjd.size),
            "wfd_visit_count": int(self._wfd_indices.size),
            "ddf_visit_count": self._ddf_visit_count,
            "ddf_fields": self._ddf_fields,
            "survey_start_mjd": self._survey_start_mjd,
            "bin_size_deg": self._bin_size_deg,
            "build_seconds": self._build_seconds,
        }

    def _candidate_indices(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
    ) -> np.ndarray:
        center_ra_bin = int((ra_deg % 360.0) // self._bin_size_deg)
        center_dec_bin = int((dec_deg + 90.0) // self._bin_size_deg)
        center_dec_bin = min(max(center_dec_bin, 0), self._n_dec_bins - 1)
        dec_radius = int(math.ceil(radius_deg / self._bin_size_deg)) + 1
        cos_dec = abs(math.cos(math.radians(dec_deg)))
        if cos_dec < 1.0e-6:
            ra_bins = range(self._n_ra_bins)
        else:
            ra_radius = int(
                math.ceil(radius_deg / cos_dec / self._bin_size_deg)
            ) + 1
            if 2 * ra_radius + 1 >= self._n_ra_bins:
                ra_bins = range(self._n_ra_bins)
            else:
                ra_bins = sorted(
                    {
                        (center_ra_bin + delta) % self._n_ra_bins
                        for delta in range(-ra_radius, ra_radius + 1)
                    }
                )
        chunks: list[np.ndarray] = []
        for dec_bin in range(
            max(0, center_dec_bin - dec_radius),
            min(self._n_dec_bins, center_dec_bin + dec_radius + 1),
        ):
            for ra_bin in ra_bins:
                cell = dec_bin * self._n_ra_bins + ra_bin
                start = int(self._cell_offsets[cell])
                stop = int(self._cell_offsets[cell + 1])
                if stop > start:
                    chunks.append(self._cell_order[start:stop])
        if not chunks:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(chunks)

    def _indices_at_sky_position(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        survey: str,
        radius_deg: float,
    ) -> np.ndarray:
        candidates = self._candidate_indices(ra_deg, dec_deg, radius_deg)
        if candidates.size == 0:
            return candidates
        center_ra = math.radians(ra_deg)
        center_dec = math.radians(dec_deg)
        candidate_ra = np.deg2rad(self._ra_deg[candidates].astype(np.float64))
        candidate_dec = np.deg2rad(self._dec_deg[candidates].astype(np.float64))
        cosine = np.sin(center_dec) * np.sin(candidate_dec) + np.cos(
            center_dec
        ) * np.cos(candidate_dec) * np.cos(candidate_ra - center_ra)
        inside = cosine >= math.cos(math.radians(radius_deg))
        if survey == "wfd":
            inside &= ~self._is_ddf[candidates]
        elif survey == "ddf":
            inside &= self._is_ddf[candidates]
        return candidates[inside]

    def at_sky_position(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        survey: str = "all",
        radius_deg: float = 1.75,
        duration_days: float | None = 3650.0,
    ) -> SurveyCadence:
        """Return visits whose pointings cover one sky position.

        ``survey`` may be ``"all"``, ``"wfd"``, or ``"ddf"``. Times are
        measured from the common first MJD in the OpSim database, preserving
        seasonal offsets between different sky positions.
        """

        raw_ra = float(ra_deg)
        dec = float(dec_deg)
        radius = float(radius_deg)
        selection = str(survey).lower()
        if (
            not math.isfinite(raw_ra)
            or not math.isfinite(dec)
            or not -90.0 <= dec <= 90.0
        ):
            raise ValueError("ra_deg and dec_deg must be finite sky coordinates")
        ra = raw_ra % 360.0
        if not math.isfinite(radius) or radius <= 0.0 or radius > 180.0:
            raise ValueError("radius_deg must lie between zero and 180")
        if selection not in {"all", "wfd", "ddf"}:
            raise ValueError("survey must be 'all', 'wfd', or 'ddf'")
        indices = self._indices_at_sky_position(
            ra,
            dec,
            survey=selection,
            radius_deg=radius,
        )
        return self._cadence(
            indices,
            duration_days=duration_days,
            metadata={
                "survey": f"Rubin {selection.upper()}",
                "selection": "sky_position",
                "field_ra_deg": ra,
                "field_dec_deg": dec,
                "selection_radius_deg": radius,
            },
        )

    def sample(
        self,
        *,
        seed: int,
        survey: str = "wfd",
        field: str | None = None,
        include_wfd: bool = False,
        radius_deg: float = 1.75,
        min_visits: int = 700,
        max_visits: int = 1250,
        duration_days: float | None = 3650.0,
        max_tries: int = 12,
    ) -> SurveyCadence:
        """Select one reproducible random WFD or DDF cadence.

        WFD is the default. A DDF may be selected randomly or by ``field``;
        ``include_wfd=True`` adds ordinary visits covering the same DDF sky
        position. Visit-count limits apply only to random WFD selection.
        """

        selection = str(survey).lower()
        if selection not in {"wfd", "ddf"}:
            raise ValueError("survey must be 'wfd' or 'ddf'")
        if field is not None and selection != "ddf":
            raise ValueError("field is only valid for survey='ddf'")
        if include_wfd and selection != "ddf":
            raise ValueError("include_wfd is only valid for survey='ddf'")
        if min_visits < 1 or max_visits < min_visits or max_tries < 1:
            raise ValueError("invalid WFD visit limits")
        if not math.isfinite(radius_deg) or radius_deg <= 0.0 or radius_deg > 180.0:
            raise ValueError("radius_deg must lie between zero and 180")
        rng = np.random.default_rng(int(seed))

        if selection == "ddf":
            if not self._ddf_fields:
                raise RuntimeError("OpSim database contains no DDF visits")
            if field is None:
                code = int(rng.integers(0, len(self._ddf_fields)))
            else:
                lookup = {
                    name.casefold(): index
                    for index, name in enumerate(self._ddf_fields)
                }
                try:
                    code = lookup[str(field).casefold()]
                except KeyError as error:
                    raise KeyError(
                        f"unknown DDF field {field!r}; available fields are "
                        f"{self._ddf_fields}"
                    ) from error
            center_ra, center_dec = self._ddf_centers[code]
            cadence = self.at_sky_position(
                center_ra,
                center_dec,
                survey="all" if include_wfd else "ddf",
                radius_deg=radius_deg,
                duration_days=duration_days,
            )
            return SurveyCadence(
                cadence.time_days,
                cadence.band_names,
                cadence.five_sigma_depth,
                cadence.mjd,
                cadence.seeing_fwhm_arcsec,
                {
                    **dict(cadence.metadata),
                    "survey": "Rubin DDF",
                    "selection": "random_ddf" if field is None else "named_ddf",
                    "ddf_field": self._ddf_fields[code],
                    "include_wfd": bool(include_wfd),
                    "seed": int(seed),
                },
            )

        best: tuple[float, int, float, float, np.ndarray] | None = None
        if self._wfd_indices.size == 0:
            raise RuntimeError("OpSim database contains no WFD visits")
        target = 0.5 * (min_visits + max_visits)
        for _ in range(max_tries):
            row_index = int(self._wfd_indices[int(rng.integers(self._wfd_indices.size))])
            center_ra = float(self._ra_deg[row_index])
            center_dec = float(self._dec_deg[row_index])
            indices = self._indices_at_sky_position(
                center_ra,
                center_dec,
                survey="wfd",
                radius_deg=float(radius_deg),
            )
            candidate = (
                abs(int(indices.size) - target),
                row_index,
                center_ra,
                center_dec,
                indices,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
            if min_visits <= indices.size <= max_visits:
                break
        if best is None or best[-1].size == 0:
            raise RuntimeError("could not find a non-empty WFD field")
        _, row_index, center_ra, center_dec, indices = best
        return self._cadence(
            indices,
            duration_days=duration_days,
            metadata={
                "survey": "Rubin WFD",
                "selection": "random_wfd",
                "field_ra_deg": center_ra,
                "field_dec_deg": center_dec,
                "selection_radius_deg": float(radius_deg),
                "source_visit_index": row_index,
                "seed": int(seed),
            },
        )

    def _cadence(
        self,
        indices: np.ndarray,
        *,
        duration_days: float | None,
        metadata: Mapping[str, object],
    ) -> SurveyCadence:
        if indices.size == 0:
            raise RuntimeError("no OpSim visits cover the requested sky position")
        if duration_days is not None:
            duration = float(duration_days)
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError("duration_days must be finite and non-negative")
        else:
            duration = math.inf
        order = np.argsort(self._mjd[indices], kind="stable")
        selected = indices[order]
        time_days = self._mjd[selected] - self._survey_start_mjd
        keep = time_days <= duration
        selected = selected[keep]
        time_days = time_days[keep]
        if selected.size == 0:
            raise RuntimeError("no selected OpSim visits lie within duration_days")
        seeing = self._seeing_fwhm_arcsec[selected].astype(np.float64)
        seeing_output = None if np.any(np.isnan(seeing)) else seeing
        return SurveyCadence(
            time_days=time_days,
            band_names=tuple(self._band_names[code] for code in self._band_codes[selected]),
            five_sigma_depth=self._five_sigma_depth[selected].astype(np.float64),
            mjd=self._mjd[selected].copy(),
            seeing_fwhm_arcsec=seeing_output,
            metadata={
                **dict(metadata),
                "database": self._database_path.name,
                "survey_start_mjd": self._survey_start_mjd,
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


@dataclass(frozen=True)
class PhotometricLightCurve:
    """One single-image light curve sampled at survey visits."""

    time_days: torch.Tensor
    band_names: tuple[str, ...]
    magnitude: torch.Tensor
    magnitude_error: torch.Tensor
    noiseless_magnitude: torch.Tensor
    image_name: str = "source"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.time_days)
        magnitude = torch.as_tensor(self.magnitude)
        error = torch.as_tensor(self.magnitude_error)
        noiseless = torch.as_tensor(self.noiseless_magnitude)
        expected = (times.numel(),)
        if times.ndim != 1 or tuple(magnitude.shape) != expected:
            raise ValueError("single-image observation arrays must have shape [visit]")
        if error.shape != magnitude.shape or noiseless.shape != magnitude.shape:
            raise ValueError("all magnitude arrays must have the same shape")
        if len(self.band_names) != times.numel():
            raise ValueError("band_names must contain one value per visit")
        if not self.image_name:
            raise ValueError("image_name must be non-empty")
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


def _zero_points(
    cadence: SurveyCadence,
    zero_point_flux: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(zero_point_flux, Mapping):
        zero_points = {str(key): float(value) for key, value in zero_point_flux.items()}
    else:
        zero_points = {band: float(zero_point_flux) for band in set(cadence.band_names)}
    if any(not math.isfinite(value) or value <= 0 for value in zero_points.values()):
        raise ValueError("zero-point fluxes must be finite and positive")
    return zero_points


def _sample_noiseless_magnitudes(
    curves: Sequence[LightCurve],
    cadence: SurveyCadence,
    zero_points: Mapping[str, float],
) -> np.ndarray:
    noiseless = np.empty((cadence.visit_count, len(curves)), dtype=np.float64)
    visits_by_band = {
        band: np.flatnonzero(np.asarray(cadence.band_names) == band)
        for band in set(cadence.band_names)
    }
    for image_index, curve in enumerate(curves):
        times = curve.times_days.detach().cpu().to(torch.float64).numpy()
        flux = curve.flux.detach().cpu().to(torch.float64).numpy()
        if cadence.time_days[0] < times[0] or cadence.time_days[-1] > times[-1]:
            raise ValueError("survey cadence lies outside a light-curve time axis")
        band_indices = {name: index for index, name in enumerate(curve.band_names)}
        for band, visit_indices in visits_by_band.items():
            if band not in band_indices:
                raise KeyError(f"light curve does not contain survey band {band!r}")
            if band not in zero_points:
                raise KeyError(f"zero_point_flux does not contain band {band!r}")
            values = np.interp(
                cadence.time_days[visit_indices],
                times,
                flux[:, band_indices[band]],
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("sampled fluxes must be finite and positive")
            noiseless[visit_indices, image_index] = -2.5 * np.log10(
                values / zero_points[band]
            )
    return noiseless


def _noise_and_error(
    noiseless: torch.Tensor,
    cadence: SurveyCadence,
    *,
    seed: int | None,
    add_noise: bool,
    gamma_by_band: Mapping[str, float] | None,
    systematic_floor_mag: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_count = 1 if noiseless.ndim == 1 else int(noiseless.shape[1])
    depth = torch.from_numpy(cadence.five_sigma_depth)
    if noiseless.ndim == 2:
        depth = depth[:, None].expand_as(noiseless)
    repeated_bands = tuple(
        band for band in cadence.band_names for _ in range(image_count)
    )
    error = rubin_magnitude_uncertainty(
        noiseless,
        depth,
        repeated_bands,
        gamma_by_band=gamma_by_band,
        systematic_floor_mag=systematic_floor_mag,
    )
    if not add_noise:
        return noiseless.clone(), error
    if seed is None:
        noise = torch.randn(noiseless.shape, dtype=noiseless.dtype)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(
            noiseless.shape,
            dtype=noiseless.dtype,
            generator=generator,
        )
    return noiseless + noise * error, error


def observe_light_curve(
    curve: LightCurve,
    cadence: SurveyCadence,
    *,
    image_name: str = "source",
    zero_point_flux: float | Mapping[str, float] = 3631.0,
    seed: int | None = None,
    add_noise: bool = True,
    gamma_by_band: Mapping[str, float] | None = None,
    systematic_floor_mag: float = 0.005,
) -> PhotometricLightCurve:
    """Sample one light curve at survey visits and add Rubin-like noise."""

    zero_points = _zero_points(cadence, zero_point_flux)
    noiseless = torch.from_numpy(
        _sample_noiseless_magnitudes((curve,), cadence, zero_points)[:, 0]
    )
    magnitude, error = _noise_and_error(
        noiseless,
        cadence,
        seed=seed,
        add_noise=add_noise,
        gamma_by_band=gamma_by_band,
        systematic_floor_mag=systematic_floor_mag,
    )
    return PhotometricLightCurve(
        time_days=torch.from_numpy(cadence.time_days.copy()),
        band_names=cadence.band_names,
        magnitude=magnitude,
        magnitude_error=error,
        noiseless_magnitude=noiseless,
        image_name=image_name,
        metadata={
            "noise_model": "rubin_random_plus_systematic",
            "systematic_floor_mag": float(systematic_floor_mag),
            "noise_added": bool(add_noise),
            "seed": seed,
            "zero_point_flux": zero_points,
            "cadence": dict(cadence.metadata),
        },
    )

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
    """Sample resolved light curves at shared visits and add Rubin-like noise.

    One cadence is intentionally applied to every macroimage. Source light
    curves are physical flux densities in Jy, so the default converts them
    directly to AB magnitudes using the 3631 Jy zero point.
    """

    zero_points = _zero_points(cadence, zero_point_flux)
    noiseless_tensor = torch.from_numpy(
        _sample_noiseless_magnitudes(
            tuple(image.light_curve for image in curves.images),
            cadence,
            zero_points,
        )
    )
    magnitude, error = _noise_and_error(
        noiseless_tensor,
        cadence,
        seed=seed,
        add_noise=add_noise,
        gamma_by_band=gamma_by_band,
        systematic_floor_mag=systematic_floor_mag,
    )

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
            "cadence": dict(cadence.metadata),
        },
    )
