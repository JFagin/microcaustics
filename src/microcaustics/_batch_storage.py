"""Disk-backed storage for independently batched systems."""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .io import _light_curve_from_payload, _light_curve_payload, load_light_curve
from .results import MacroImageLightCurve, MultiImageLightCurves

_MANIFEST_VERSION = 1


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "image"


def _archive_stem(system_index: int, image_index: int, image_name: str) -> str:
    # Numeric prefixes make lexical order identical to simulation order while
    # retaining a readable, filesystem-safe macroimage label.
    return (
        f"system_{system_index:06d}__image_{image_index:02d}_{_safe_name(image_name)}"
    )


@dataclass(frozen=True)
class StoredSystemLightCurves:
    """Location and grouping metadata for one disk-backed system result."""

    system_index: int
    image_names: tuple[str, ...]
    arrival_time_delays_days: tuple[float, ...]
    storage_mode: str
    output_path: Path
    image_locations: tuple[str, ...]
    is_multi_image: bool
    metadata: dict[str, object]

    @property
    def image_count(self) -> int:
        """Number of macroimage curves belonging to this system."""

        return len(self.image_names)

    def load(self, *, device="cpu"):
        """Load and reconstruct this system without rerunning the simulation."""

        curves = []
        if self.storage_mode == "flat-npz":
            curves = [
                load_light_curve(self.output_path / location, device=device)
                for location in self.image_locations
            ]
        elif self.storage_mode == "combined-npz":
            with np.load(self.output_path, allow_pickle=False) as stored:
                for prefix in self.image_locations:
                    marker = f"{prefix}__"
                    payload = {
                        name[len(marker) :]: stored[name]
                        for name in stored.files
                        if name.startswith(marker)
                    }
                    curves.append(_light_curve_from_payload(payload, device=device))
        else:
            raise ValueError(f"unsupported storage mode {self.storage_mode!r}")
        if not self.is_multi_image:
            return curves[0]
        images = tuple(
            MacroImageLightCurve(name, delay, curve)
            for name, delay, curve in zip(
                self.image_names,
                self.arrival_time_delays_days,
                curves,
                strict=True,
            )
        )
        return MultiImageLightCurves(images, metadata=self.metadata)

    def to_manifest(self) -> dict[str, object]:
        """Return this record as a JSON-compatible manifest entry."""

        return {
            "system_index": self.system_index,
            "image_names": list(self.image_names),
            "arrival_time_delays_days": list(self.arrival_time_delays_days),
            "image_locations": list(self.image_locations),
            "is_multi_image": self.is_multi_image,
            "metadata": self.metadata,
        }


def _system_parts(result):
    if isinstance(result, MultiImageLightCurves):
        return (
            tuple(item.image_name for item in result.images),
            tuple(float(item.arrival_time_delay_days) for item in result.images),
            tuple(item.light_curve for item in result.images),
            True,
            dict(result.metadata),
        )
    return (("image",), (0.0,), (result,), False, {})


def _write_npz(path: Path, payload, *, compression: bool) -> None:
    # Never expose a half-written image as resumable output.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    writer = np.savez_compressed if compression else np.savez
    try:
        with temporary.open("wb") as handle:
            writer(handle, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BatchOutputWriter:
    """Single-threaded flat or combined writer used behind a bounded queue."""

    def __init__(
        self,
        output_path,
        *,
        compression: bool,
        overwrite: bool,
        resume: bool,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.compression = bool(compression)
        self.overwrite = bool(overwrite)
        self.resume = bool(resume)
        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume cannot both be true")
        self.mode = (
            "combined-npz" if self.output_path.suffix.lower() == ".npz" else "flat-npz"
        )
        self.records: list[StoredSystemLightCurves] = []
        self.write_seconds = 0.0
        self._zip = None
        self._temporary = None
        if self.mode == "flat-npz":
            self.output_path.mkdir(parents=True, exist_ok=True)
            manifest = self.output_path / "manifest.json"
            if manifest.exists() and not (overwrite or resume):
                raise FileExistsError(f"batch manifest already exists: {manifest}")
            if overwrite:
                manifest.unlink(missing_ok=True)
        else:
            if resume:
                raise ValueError("resume is supported for flat directory output only")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"batch output already exists: {self.output_path}"
                )
            self._temporary = self.output_path.with_name(
                f".{self.output_path.name}.{os.getpid()}.tmp"
            )
            self._temporary.unlink(missing_ok=True)
            compression_kind = (
                zipfile.ZIP_DEFLATED if self.compression else zipfile.ZIP_STORED
            )
            # NumPy's NPZ format is a ZIP of NPY members. Writing members
            # incrementally avoids retaining every system payload in RAM.
            self._zip = zipfile.ZipFile(
                self._temporary,
                "w",
                compression=compression_kind,
                compresslevel=1 if self.compression else None,
                allowZip64=True,
            )

    def _write_array(self, name: str, value: np.ndarray) -> None:
        # Keep allow_pickle=False compatibility with ordinary np.load callers.
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
        assert self._zip is not None
        self._zip.writestr(f"{name}.npy", buffer.getvalue())

    def write_system(self, system_index: int, result) -> StoredSystemLightCurves:
        """Serialize one complete result and return its location record."""

        started = perf_counter()
        names, delays, curves, is_multi, metadata = _system_parts(result)
        locations = []
        for image_index, (name, curve) in enumerate(zip(names, curves, strict=True)):
            stem = _archive_stem(system_index, image_index, name)
            payload = _light_curve_payload(curve)
            if self.mode == "flat-npz":
                filename = f"{stem}.npz"
                destination = self.output_path / filename
                if destination.exists() and not (self.overwrite or self.resume):
                    raise FileExistsError(
                        f"batch light curve already exists: {destination}"
                    )
                # A complete resumed system never reaches this method. If only
                # some image files existed, rewrite every image so one system
                # can never mix products from different attempts.
                _write_npz(destination, payload, compression=self.compression)
                locations.append(filename)
            else:
                for key, value in payload.items():
                    self._write_array(f"{stem}__{key}", value)
                locations.append(stem)
        record = StoredSystemLightCurves(
            system_index,
            names,
            delays,
            self.mode,
            self.output_path,
            tuple(locations),
            is_multi,
            metadata,
        )
        self.records.append(record)
        self.write_seconds += perf_counter() - started
        return record

    def resume_record(
        self,
        system_index: int,
        image_names: tuple[str, ...],
        arrival_time_delays_days: tuple[float, ...],
        is_multi_image: bool,
        metadata: dict[str, object],
    ) -> StoredSystemLightCurves | None:
        """Return a record when every expected flat file already exists."""

        if not self.resume:
            return None
        locations = tuple(
            f"{_archive_stem(system_index, image_index, name)}.npz"
            for image_index, name in enumerate(image_names)
        )
        if not all((self.output_path / location).is_file() for location in locations):
            return None
        record = StoredSystemLightCurves(
            system_index,
            image_names,
            arrival_time_delays_days,
            self.mode,
            self.output_path,
            locations,
            is_multi_image,
            metadata,
        )
        self.records.append(record)
        return record

    def finalize(self) -> tuple[StoredSystemLightCurves, ...]:
        """Write the manifest and atomically finish the requested output."""

        started = perf_counter()
        records = tuple(sorted(self.records, key=lambda item: item.system_index))
        manifest = {
            "schema_version": _MANIFEST_VERSION,
            "storage_mode": self.mode,
            "systems": [record.to_manifest() for record in records],
        }
        text = json.dumps(manifest, sort_keys=True, default=str, indent=2)
        if self.mode == "flat-npz":
            destination = self.output_path / "manifest.json"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, destination)
        else:
            self._write_array("manifest_json", np.asarray(text))
            assert self._zip is not None and self._temporary is not None
            self._zip.close()
            self._zip = None
            # A combined archive is published only after its central directory
            # and embedded manifest have been written successfully.
            os.replace(self._temporary, self.output_path)
        self.write_seconds += perf_counter() - started
        return records

    def abort(self) -> None:
        """Close and remove an unfinished combined archive."""

        if self._zip is not None:
            self._zip.close()
            self._zip = None
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
