"""Shared post-processing for bandpass-integrated quasar photometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from .bandpasses import BandpassGrid, BandpassSet
from .results import LightCurve, TimeDependentSpectrum
from .spectra import QuasarSpectrum, _log_flux_interpolate


@dataclass(frozen=True)
class SpectralSamplingPlan:
    """Sparse physical continuum and dense one-dimensional photometry grids."""

    wavelengths_angstrom: tuple[float, ...]
    band_grid: BandpassGrid
    static_grid: BandpassGrid
    output_wavelengths_angstrom: tuple[float, ...]

    @property
    def bands_angstrom(self) -> dict[str, float]:
        """Return sparse wavelength nodes in the source band-mapping format."""

        return {
            f"spectral_node_{index:04d}": wavelength
            for index, wavelength in enumerate(self.wavelengths_angstrom)
        }


def prepare_spectral_source(
    source,
    distances,
    bandpasses,
    *,
    spectral_model: QuasarSpectrum | None,
    wavelength_samples: int,
    spectrum_wavelengths,
    return_spectrum: bool,
    host_lensing: str,
):
    """Select log-spaced continuum channels before physical source pixelation."""

    from .bandpasses import resolve_bandpasses

    redshift = distances.source_redshift
    if redshift is None:
        raise ValueError(
            "spectral photometry requires distances with a source_redshift"
        )
    resolved_bandpasses = resolve_bandpasses(bandpasses)
    required_model = (
        replace(spectral_model, include_host=False)
        if spectral_model is not None
        and host_lensing == "omit"
        and spectral_model.include_host
        else spectral_model
    )
    required = (
        ()
        if required_model is None
        else required_model.required_wavelengths(float(redshift))
    )
    plan = spectral_sampling_plan(
        resolved_bandpasses,
        continuum_samples=wavelength_samples,
        spectrum_wavelengths=spectrum_wavelengths,
        return_spectrum=return_spectrum,
        required_wavelengths=required,
    )
    with_bands = getattr(source, "with_bands", None)
    if with_bands is None:
        raise TypeError("bandpasses require a source that supports with_bands")
    return (
        with_bands(plan.bands_angstrom),
        plan,
        resolved_bandpasses,
        float(redshift),
    )


def spectral_sampling_plan(
    bandpasses: BandpassSet,
    *,
    continuum_samples: int,
    spectrum_wavelengths,
    return_spectrum: bool,
    required_wavelengths=(),
) -> SpectralSamplingPlan:
    """Keep physical continuum channels separate from filter quadrature."""

    if (
        not isinstance(continuum_samples, int)
        or isinstance(continuum_samples, bool)
        or continuum_samples < 2
    ):
        raise ValueError("continuum_samples must be an integer of at least two")
    requested = None
    if spectrum_wavelengths is not None:
        requested_array = np.asarray(spectrum_wavelengths, dtype=np.float64).reshape(-1)
        if requested_array.size and (
            np.any(~np.isfinite(requested_array))
            or np.any(requested_array <= 0.0)
            or np.any(np.diff(requested_array) <= 0.0)
        ):
            raise ValueError(
                "spectrum_wavelengths must be finite, positive, and strictly increasing"
            )
        requested = tuple(requested_array.tolist())
    required_array = np.asarray(required_wavelengths, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(required_array)) or np.any(required_array <= 0.0):
        raise ValueError("required wavelengths must be finite and positive")
    return _cached_spectral_sampling_plan(
        bandpasses,
        continuum_samples,
        requested,
        bool(return_spectrum),
        tuple(required_array.tolist()),
    )


@lru_cache(maxsize=128)
def _cached_spectral_sampling_plan(
    bandpasses: BandpassSet,
    continuum_samples: int,
    spectrum_wavelengths: tuple[float, ...] | None,
    return_spectrum: bool,
    required_wavelengths: tuple[float, ...],
) -> SpectralSamplingPlan:
    """Build one immutable plan per wavelength and response contract."""

    # The physical disk is evaluated at a fixed number of log-spaced nodes
    # across the union of filter supports, independent of filter count.
    supports = [band.support_angstrom for band in bandpasses.bandpasses]
    lower = min(support[0] for support in supports)
    upper = max(support[1] for support in supports)
    base = np.geomspace(lower, upper, continuum_samples, dtype=np.float64)
    # Quadrature is cheap after spatial disk evaluation. Enough nodes are used
    # to resolve the response curves without adding spatial disk channels.
    quadrature, static = _spectral_base_grids(bandpasses)
    required = np.asarray(required_wavelengths, dtype=np.float64)
    # Normalization nodes already inside the continuum support are obtained
    # from the sparse physical grid. Only out-of-range nodes need extra disk
    # evaluations; fine-grid template calculations still retain every node.
    outside = required[(required < lower) | (required > upper)]
    combined = np.unique(np.concatenate((base, outside)))
    if return_spectrum and spectrum_wavelengths is None:
        requested = np.linspace(base[0], base[-1], 1000, dtype=np.float64)
    else:
        requested = np.asarray(
            () if spectrum_wavelengths is None else spectrum_wavelengths,
            dtype=np.float64,
        ).reshape(-1)
    if requested.size and (
        requested[0] < combined[0] or requested[-1] > combined[-1]
    ):
        raise ValueError(
            "spectrum_wavelengths must lie within the sampled continuum range "
            f"[{combined[0]:.6g}, {combined[-1]:.6g}] Angstrom; extrapolation "
            "is not supported"
        )
    quadrature_wavelengths = np.asarray(quadrature.wavelengths_angstrom)
    band_wavelengths, band_inverse = np.unique(
        np.concatenate((quadrature_wavelengths, required)), return_inverse=True
    )
    weights = torch.zeros(
        (band_wavelengths.size, len(bandpasses.names)),
        dtype=quadrature.weights.dtype,
    )
    weights.index_add_(
        0, torch.from_numpy(band_inverse[: quadrature_wavelengths.size]), quadrature.weights
    )
    static_wavelengths = np.asarray(static.wavelengths_angstrom)
    static_combined, static_inverse = np.unique(
        np.concatenate((static_wavelengths, required)),
        return_inverse=True,
    )
    static_weights = torch.zeros(
        (static_combined.size, len(bandpasses.names)),
        dtype=static.weights.dtype,
    )
    static_weights.index_add_(
        0,
        torch.from_numpy(static_inverse[: static_wavelengths.size]),
        static.weights,
    )
    return SpectralSamplingPlan(
        tuple(combined.tolist()),
        BandpassGrid(tuple(band_wavelengths.tolist()), bandpasses.names, weights),
        BandpassGrid(
            tuple(static_combined.tolist()),
            bandpasses.names,
            static_weights,
        ),
        tuple(requested.tolist()),
    )


@lru_cache(maxsize=16)
def _spectral_base_grids(bandpasses: BandpassSet) -> tuple[BandpassGrid, BandpassGrid]:
    """Reuse response integration grids across redshifts and spectrum draws."""

    return bandpasses.grid(1024), bandpasses.dense_grid()


def finish_spectral_light_curve(
    continuum_curve: LightCurve,
    mean_curve: LightCurve | None,
    plan: SpectralSamplingPlan,
    *,
    spectral_model: QuasarSpectrum | None,
    source_redshift: float,
    macro_magnification: float,
    include_microlensing_only: bool,
    return_spectrum: bool,
    host_lensing: str,
    bandpass_version: str | None,
    luminosity_distance_m: float | None = None,
) -> LightCurve:
    """Collapse shared continuum channels into photometry and spectrum outputs."""

    wavelengths = continuum_curve.flux.new_tensor(plan.wavelengths_angstrom)
    dynamic_lensed = continuum_curve.flux
    dynamic_unlensed = continuum_curve.unlensed_flux
    if dynamic_unlensed is None:
        raise RuntimeError("spectral photometry requires the unlensed continuum flux")
    reference_curve = continuum_curve if mean_curve is None else mean_curve
    if reference_curve.unlensed_flux is None:
        raise RuntimeError("spectral photometry requires a mean unlensed continuum")
    mean_unlensed = reference_curve.unlensed_flux[0]
    mean_lensed = reference_curve.flux

    if spectral_model is not None:
        if (
            spectral_model.absolute_i_magnitude is None
            and luminosity_distance_m is not None
            and (
                spectral_model.include_host
                or (
                    spectral_model.include_emission_lines
                    and spectral_model.emission_line_type is None
                )
            )
        ):
            spectral_model = replace(
                spectral_model,
                absolute_i_magnitude=spectral_model.infer_absolute_i_magnitude(
                    wavelengths,
                    mean_unlensed,
                    source_redshift=source_redshift,
                    luminosity_distance_m=luminosity_distance_m,
                ),
            )
        evaluated_model = (
            replace(spectral_model, include_host=False)
            if host_lensing == "omit" and spectral_model.include_host
            else spectral_model
        )
    band_wavelengths = wavelengths.new_tensor(plan.band_grid.wavelengths_angstrom)
    band_mean_unlensed = _log_flux_interpolate(
        wavelengths, mean_unlensed, band_wavelengths
    )
    if spectral_model is None:
        band_attenuation = torch.ones_like(band_mean_unlensed)
        band_scale = torch.ones_like(band_wavelengths)
    else:
        band_components = evaluated_model.components(
            band_wavelengths,
            band_mean_unlensed,
            source_redshift=source_redshift,
        )
        band_attenuation = band_components.continuum_fnu / band_mean_unlensed.clamp_min(
            torch.finfo(band_mean_unlensed.dtype).tiny
        )
        pivot = band_wavelengths.new_tensor(6000.0)
        magnitude_offset = spectral_model.global_magnitude_offset + (
            spectral_model.color_tilt_magnitude * torch.log(band_wavelengths / pivot)
        )
        band_scale = torch.pow(
            band_wavelengths.new_tensor(10.0), -0.4 * magnitude_offset
        )
    band_multiplier = band_attenuation * band_scale

    def integrate_continuum(values: torch.Tensor) -> torch.Tensor:
        sampled = _log_flux_interpolate(wavelengths, values, band_wavelengths)
        return plan.band_grid.integrate(sampled * band_multiplier)

    continuum_bands = integrate_continuum(dynamic_lensed)
    mean_continuum_bands = integrate_continuum(mean_lensed)
    unlensed_continuum_bands = integrate_continuum(dynamic_unlensed)
    mean_unlensed_row = plan.band_grid.integrate(band_mean_unlensed * band_multiplier)
    mean_unlensed_continuum_bands = mean_unlensed_row.unsqueeze(0).expand_as(
        unlensed_continuum_bands
    )
    if spectral_model is None:
        line_bands = torch.zeros_like(continuum_bands)
        host_bands = torch.zeros_like(continuum_bands)
        unlensed_line_bands = torch.zeros_like(continuum_bands)
        unlensed_host_bands = torch.zeros_like(continuum_bands)
    else:
        dense_wavelengths = wavelengths.new_tensor(
            plan.static_grid.wavelengths_angstrom
        )
        dense_mean_continuum = _log_flux_interpolate(
            wavelengths,
            mean_unlensed,
            dense_wavelengths,
        )
        dense_components = evaluated_model.components(
            dense_wavelengths,
            dense_mean_continuum,
            source_redshift=source_redshift,
        )
        dense_pivot = dense_wavelengths.new_tensor(6000.0)
        dense_offset = spectral_model.global_magnitude_offset + (
            spectral_model.color_tilt_magnitude
            * torch.log(dense_wavelengths / dense_pivot)
        )
        dense_scale = torch.pow(
            dense_wavelengths.new_tensor(10.0), -0.4 * dense_offset
        )
        dense_unlensed_lines = dense_components.emission_line_fnu * dense_scale
        dense_lines = dense_unlensed_lines * float(macro_magnification)
        dense_unlensed_host = dense_components.host_fnu * dense_scale
        dense_host = _lensed_host(
            dense_unlensed_host,
            macro_magnification=macro_magnification,
            policy=host_lensing,
        )
        line_row = plan.static_grid.integrate(dense_lines)
        host_row = plan.static_grid.integrate(dense_host)
        unlensed_line_row = plan.static_grid.integrate(dense_unlensed_lines)
        unlensed_host_row = plan.static_grid.integrate(
            torch.zeros_like(dense_unlensed_host)
            if host_lensing == "omit"
            else dense_unlensed_host
        )
        line_bands = line_row.unsqueeze(0).expand_as(continuum_bands)
        host_bands = host_row.unsqueeze(0).expand_as(continuum_bands)
        unlensed_line_bands = unlensed_line_row.unsqueeze(0).expand_as(
            continuum_bands
        )
        unlensed_host_bands = unlensed_host_row.unsqueeze(0).expand_as(
            continuum_bands
        )
    total_bands = continuum_bands + line_bands + host_bands
    mean_total_bands = mean_continuum_bands + line_bands + host_bands
    unlensed_total_bands = (
        unlensed_continuum_bands + unlensed_line_bands + unlensed_host_bands
    )
    mean_unlensed_total_bands = (
        mean_unlensed_continuum_bands
        + unlensed_line_bands
        + unlensed_host_bands
    )

    if spectral_model is not None:
        effective_wavelengths = (
            torch.as_tensor(
                plan.band_grid.wavelengths_angstrom,
                dtype=plan.band_grid.weights.dtype,
            )
            @ plan.band_grid.weights
        ).to(device=wavelengths.device, dtype=wavelengths.dtype)
        brightness_offsets = spectral_model.band_offsets(
            plan.band_grid.band_names,
            effective_wavelengths,
            include_smooth=False,
        )
        brightness_scale = torch.pow(
            total_bands.new_tensor(10.0), -0.4 * brightness_offsets
        )
        total_bands = total_bands * brightness_scale
        mean_total_bands = mean_total_bands * brightness_scale
        continuum_bands = continuum_bands * brightness_scale
        mean_continuum_bands = mean_continuum_bands * brightness_scale
        line_bands = line_bands * brightness_scale
        host_bands = host_bands * brightness_scale
        unlensed_continuum_bands = unlensed_continuum_bands * brightness_scale
        mean_unlensed_continuum_bands = (
            mean_unlensed_continuum_bands * brightness_scale
        )
        unlensed_line_bands = unlensed_line_bands * brightness_scale
        unlensed_host_bands = unlensed_host_bands * brightness_scale
        unlensed_total_bands = unlensed_total_bands * brightness_scale
        mean_unlensed_total_bands = mean_unlensed_total_bands * brightness_scale

        amplitude_offsets = spectral_model.band_offsets(
            plan.band_grid.band_names,
            effective_wavelengths,
            amplitude=True,
        )
        amplitude_scale = torch.pow(
            total_bands.new_tensor(10.0), -0.4 * amplitude_offsets
        )
        continuum_bands = mean_continuum_bands + amplitude_scale * (
            continuum_bands - mean_continuum_bands
        )
        unlensed_continuum_bands = mean_unlensed_continuum_bands + (
            amplitude_scale
            * (unlensed_continuum_bands - mean_unlensed_continuum_bands)
        )
        total_bands = continuum_bands + line_bands + host_bands
        unlensed_total_bands = (
            unlensed_continuum_bands
            + unlensed_line_bands
            + unlensed_host_bands
        )

    retained = None
    if return_spectrum:
        output_wavelengths = wavelengths.new_tensor(
            plan.output_wavelengths_angstrom
        )
        output_dynamic_lensed = _log_flux_interpolate(
            wavelengths, dynamic_lensed, output_wavelengths
        )
        output_dynamic_unlensed = _log_flux_interpolate(
            wavelengths, dynamic_unlensed, output_wavelengths
        )
        output_mean_lensed = _log_flux_interpolate(
            wavelengths, mean_lensed, output_wavelengths
        )
        output_mean_unlensed = _log_flux_interpolate(
            wavelengths, mean_unlensed, output_wavelengths
        )
        if spectral_model is None:
            output_attenuation = torch.ones_like(output_mean_unlensed)
            output_lines = torch.zeros_like(output_mean_unlensed)
            output_host = torch.zeros_like(output_mean_unlensed)
            output_scale = torch.ones_like(output_wavelengths)
        else:
            required_wavelengths = wavelengths.new_tensor(
                evaluated_model.required_wavelengths(source_redshift)
            )
            component_wavelengths, output_inverse = torch.unique(
                torch.cat((output_wavelengths, required_wavelengths)),
                sorted=True,
                return_inverse=True,
            )
            component_mean = _log_flux_interpolate(
                wavelengths, mean_unlensed, component_wavelengths
            )
            output_components = evaluated_model.components(
                component_wavelengths,
                component_mean,
                source_redshift=source_redshift,
            )
            output_indices = output_inverse[: output_wavelengths.numel()]
            component_continuum = component_mean.index_select(0, output_indices)
            output_attenuation = (
                output_components.continuum_fnu.index_select(0, output_indices)
                / component_continuum.clamp_min(
                    torch.finfo(component_continuum.dtype).tiny
                )
            )
            output_lines = output_components.emission_line_fnu.index_select(
                0, output_indices
            )
            output_host = output_components.host_fnu.index_select(0, output_indices)
            output_pivot = output_wavelengths.new_tensor(6000.0)
            output_offset = spectral_model.global_magnitude_offset + (
                spectral_model.color_tilt_magnitude
                * torch.log(output_wavelengths / output_pivot)
            )
            output_scale = torch.pow(
                output_wavelengths.new_tensor(10.0), -0.4 * output_offset
            )
        output_continuum_lensed = (
            output_dynamic_lensed * output_attenuation * output_scale
        )
        output_continuum_unlensed = (
            output_dynamic_unlensed * output_attenuation * output_scale
        )
        output_mean_continuum_lensed = (
            output_mean_lensed * output_attenuation * output_scale
        )
        output_line_lensed = (
            output_lines * float(macro_magnification) * output_scale
        ).unsqueeze(0).expand_as(output_continuum_lensed)
        output_host_lensed = _lensed_host(
            output_host * output_scale,
            macro_magnification=macro_magnification,
            policy=host_lensing,
        ).unsqueeze(0).expand_as(output_continuum_lensed)
        retained = TimeDependentSpectrum(
            times_days=continuum_curve.times_days,
            wavelengths_angstrom=output_wavelengths,
            total_flux=(
                output_continuum_lensed + output_line_lensed + output_host_lensed
            ),
            continuum_flux=output_continuum_lensed,
            microlensing_only_continuum_flux=(
                output_mean_continuum_lensed.expand_as(output_continuum_lensed)
                if include_microlensing_only
                else None
            ),
            unlensed_continuum_flux=output_continuum_unlensed,
            components={
                "emission_lines": output_line_lensed,
                "host": output_host_lensed,
            },
            metadata={
                "frame": "observed",
                "flux_units": "Jy",
                "source_redshift": float(source_redshift),
            },
        )

    return replace(
        continuum_curve,
        flux=total_bands,
        unlensed_flux=unlensed_total_bands,
        band_names=plan.band_grid.band_names,
        microlensing_only_flux=(
            mean_total_bands if include_microlensing_only else None
        ),
        microlensing_only_unlensed_flux=(
            mean_unlensed_total_bands if include_microlensing_only else None
        ),
        spectrum=retained,
        component_flux={
            "continuum": continuum_bands,
            "emission_lines": line_bands,
            "host": host_bands,
        },
        metadata={
            **continuum_curve.metadata,
            "bandpasses": {
                "names": list(plan.band_grid.band_names),
                "version": bandpass_version,
            },
            "wavelength_samples": len(plan.wavelengths_angstrom),
            "continuum_sampling": "log_wavelength_log_flux",
            "spectrum_retained": bool(return_spectrum),
            "host_lensing": host_lensing,
            "spectral_model": (
                None
                if spectral_model is None
                else {
                    "type": "empirical_quasar_spectrum",
                    "population_seed": spectral_model.population_seed,
                    "absolute_i_magnitude": spectral_model._absolute_magnitude(
                        source_redshift
                    ),
                }
            ),
        },
    )


def _lensed_host(
    host_intrinsic: torch.Tensor,
    *,
    macro_magnification: float,
    policy: str,
) -> torch.Tensor:
    if policy == "omit":
        return torch.zeros_like(host_intrinsic)
    if policy == "macro":
        return host_intrinsic * float(macro_magnification)
    if policy == "unlensed":
        return host_intrinsic
    raise ValueError("host_lensing must be 'omit', 'macro', or 'unlensed'")
