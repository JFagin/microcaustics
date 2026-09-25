"""Observed-frame photometric response curves and synthetic photometry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files

import numpy as np
import torch


@dataclass(frozen=True)
class Bandpass:
    """One dimensionless photon-counting response curve.

    Wavelengths are observed-frame Angstroms. The response need not be
    normalized, but wavelengths must be strictly increasing.
    """

    wavelength_angstrom: tuple[float, ...] | Sequence[float] | np.ndarray
    response: tuple[float, ...] | Sequence[float] | np.ndarray
    name: str

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_angstrom, dtype=np.float64)
        response = np.asarray(self.response, dtype=np.float64)
        if wavelength.ndim != 1 or response.shape != wavelength.shape:
            raise ValueError("bandpass wavelength and response must be equal 1D arrays")
        if wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("bandpass wavelengths must be strictly increasing")
        if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise ValueError("bandpass wavelengths must be finite and positive")
        if not np.all(np.isfinite(response)) or np.any(response < 0.0):
            raise ValueError("bandpass response must be finite and non-negative")
        if not np.any(response > 0.0):
            raise ValueError("bandpass response must be positive somewhere")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("bandpass name must be a non-empty string")
        object.__setattr__(self, "wavelength_angstrom", tuple(wavelength.tolist()))
        object.__setattr__(self, "response", tuple(response.tolist()))

    @property
    def support_angstrom(self) -> tuple[float, float]:
        """Smallest interval containing the nonzero response."""

        wavelength = np.asarray(self.wavelength_angstrom)
        nonzero = np.flatnonzero(np.asarray(self.response) > 0.0)
        lower = max(int(nonzero[0]) - 1, 0)
        upper = min(int(nonzero[-1]) + 1, wavelength.size - 1)
        return float(wavelength[lower]), float(wavelength[upper])

    def quadrature(self, samples: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Return Gauss-Legendre nodes and normalized AB ``f_nu`` weights.

        For a photon-counting response ``R(lambda)``, the returned weights
        approximate ``integral(f_nu R/lambda dlambda) / integral(R/lambda
        dlambda)``. They therefore sum to one and preserve a flat ``f_nu``.
        """

        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
            raise ValueError("samples must be an integer of at least two")
        lower, upper = self.support_angstrom
        roots, base_weights = np.polynomial.legendre.leggauss(samples)
        nodes = 0.5 * (upper - lower) * roots + 0.5 * (upper + lower)
        dlambda_weights = 0.5 * (upper - lower) * base_weights
        response = np.interp(
            nodes,
            np.asarray(self.wavelength_angstrom),
            np.asarray(self.response),
        )
        weights = dlambda_weights * response / nodes
        normalization = weights.sum()
        if not math.isfinite(float(normalization)) or normalization <= 0.0:
            raise RuntimeError(f"bandpass {self.name!r} has zero quadrature response")
        return nodes, weights / normalization


@dataclass(frozen=True)
class BandpassGrid:
    """Shared wavelength nodes and a matrix that integrates them into bands."""

    wavelengths_angstrom: tuple[float, ...]
    band_names: tuple[str, ...]
    weights: torch.Tensor
    _device_weights: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        weights = torch.as_tensor(self.weights)
        expected = (len(self.wavelengths_angstrom), len(self.band_names))
        if weights.shape != expected:
            raise ValueError(f"bandpass weights must have shape {expected}")
        if not weights.is_floating_point() or not torch.isfinite(weights).all():
            raise ValueError("bandpass weights must be finite floating-point values")
        object.__setattr__(self, "weights", weights)

    def integrate(self, flux_nu: torch.Tensor) -> torch.Tensor:
        """Integrate an array whose final axis indexes this wavelength grid."""

        flux = torch.as_tensor(flux_nu)
        if flux.shape[-1] != len(self.wavelengths_angstrom):
            raise ValueError("flux wavelength axis does not match bandpass grid")
        if self.weights.requires_grad:
            return flux @ self.weights.to(device=flux.device, dtype=flux.dtype)
        # Keep immutable integration weights on the target GPU. A version
        # check also honors in-place edits to user-supplied weight tensors.
        try:
            version = self.weights._version
        except RuntimeError:
            # Inference tensors have no version counter; do not assume that a
            # caller's mutable tensor is safe to cache in that case.
            return flux @ self.weights.to(device=flux.device, dtype=flux.dtype)
        key = (flux.device, flux.dtype)
        entry = self._device_weights.get(key)
        if entry is None or entry[0] != version:
            weights = self.weights.to(device=flux.device, dtype=flux.dtype)
            ready = None
            if flux.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(flux.device))
            entry = (version, weights, ready)
            self._device_weights[key] = entry
        _, weights, ready = entry
        if ready is not None:
            torch.cuda.current_stream(flux.device).wait_event(ready)
        return flux @ weights


@dataclass(frozen=True)
class BandpassSet:
    """An ordered collection of photometric response curves."""

    bandpasses: tuple[Bandpass, ...] | Sequence[Bandpass] | Mapping[str, Bandpass]
    version: str | None = None

    def __post_init__(self) -> None:
        values = (
            tuple(self.bandpasses.values())
            if isinstance(self.bandpasses, Mapping)
            else tuple(self.bandpasses)
        )
        if not values:
            raise ValueError("a bandpass set cannot be empty")
        if not all(isinstance(item, Bandpass) for item in values):
            raise TypeError("bandpasses must contain Bandpass objects")
        names = tuple(item.name for item in values)
        if len(set(names)) != len(names):
            raise ValueError("bandpass names must be unique")
        object.__setattr__(self, "bandpasses", values)

    @property
    def names(self) -> tuple[str, ...]:
        """Band names in integration order."""

        return tuple(item.name for item in self.bandpasses)

    @classmethod
    def lsst(cls) -> BandpassSet:
        """Return the bundled Rubin LSST ``ugrizy`` system responses.

        The public name intentionally remains ``lsst``. The precise response
        release is recorded in :attr:`version` and result metadata.
        """

        curves = tuple(_load_lsst_band(name) for name in "ugrizy")
        return cls(curves, version="lsst2023 / throughputs tag 1.9")

    def grid(self, samples_per_band: int = 32) -> BandpassGrid:
        """Build independent quadrature nodes for every response curve."""

        wavelength_parts = []
        weight_parts = []
        band_count = len(self.bandpasses)
        for index, bandpass in enumerate(self.bandpasses):
            nodes, weights = bandpass.quadrature(samples_per_band)
            matrix = np.zeros((samples_per_band, band_count), dtype=np.float64)
            matrix[:, index] = weights
            wavelength_parts.append(nodes)
            weight_parts.append(matrix)
        return BandpassGrid(
            tuple(np.concatenate(wavelength_parts).tolist()),
            self.names,
            torch.from_numpy(np.concatenate(weight_parts, axis=0)),
        )

    def dense_grid(self, minimum_samples_per_band: int = 512) -> BandpassGrid:
        """Return a dense trapezoidal grid for inexpensive 1D components.

        Disk images use :meth:`grid`; empirical lines and host light use this
        denser grid because resolving them adds no spatial source calculation.
        """

        if minimum_samples_per_band < 2:
            raise ValueError("minimum_samples_per_band must be at least two")
        per_band = []
        for bandpass in self.bandpasses:
            wavelength = np.asarray(bandpass.wavelength_angstrom)
            lower, upper = bandpass.support_angstrom
            native = wavelength[(wavelength >= lower) & (wavelength <= upper)]
            if native.size < minimum_samples_per_band:
                native = np.unique(
                    np.concatenate(
                        (
                            native,
                            np.linspace(lower, upper, minimum_samples_per_band),
                        )
                    )
                )
            per_band.append(native)
        shared = np.unique(np.concatenate(per_band))
        matrix = np.zeros((shared.size, len(self.bandpasses)), dtype=np.float64)
        for index, bandpass in enumerate(self.bandpasses):
            response = np.interp(
                shared,
                np.asarray(bandpass.wavelength_angstrom),
                np.asarray(bandpass.response),
                left=0.0,
                right=0.0,
            )
            integrand = response / shared
            delta = np.diff(shared)
            trapezoid_weights = np.zeros_like(shared)
            trapezoid_weights[:-1] += 0.5 * delta * integrand[:-1]
            trapezoid_weights[1:] += 0.5 * delta * integrand[1:]
            matrix[:, index] = trapezoid_weights / trapezoid_weights.sum()
        return BandpassGrid(
            tuple(shared.tolist()),
            self.names,
            torch.from_numpy(matrix),
        )


def resolve_bandpasses(value: str | BandpassSet) -> BandpassSet:
    """Resolve the concise public bandpass argument."""

    if isinstance(value, BandpassSet):
        return value
    if value == "lsst":
        return BandpassSet.lsst()
    raise ValueError("bandpasses must be 'lsst' or a BandpassSet")


@lru_cache(maxsize=6)
def _load_lsst_band(name: str) -> Bandpass:
    """Read a bundled speclite ECSV response without requiring Astropy."""

    resource = files("microcaustics.data.bandpasses").joinpath(f"lsst2023-{name}.ecsv")
    rows = []
    with resource.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("wavelength")
            ):
                continue
            wavelength_nm, response = stripped.split()[:2]
            rows.append((10.0 * float(wavelength_nm), float(response)))
    values = np.asarray(rows, dtype=np.float64)
    return Bandpass(values[:, 0], values[:, 1], name)
