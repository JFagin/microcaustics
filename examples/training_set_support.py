"""Shared, script-local helpers for the training-set examples.

This module deliberately lives under ``examples`` rather than the installed
package. It assembles one scientifically explicit demonstration system from
the public API. None of these Q2237-specific choices are hidden production
defaults.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import microcaustics as mc

WAVELENGTHS_ANGSTROM = (3671.0, 4827.0, 6223.0, 7546.0, 8691.0, 9712.0)
BAND_NAMES = ("u", "g", "r", "i", "z", "y")


def plot_training_light_curves(
    records,
    output_path: str | Path,
    *,
    annotations: list[str] | None = None,
):
    """Plot training curves and their three source-center diagnostics.

    This presentation helper keeps the generation notebooks focused on the
    physical systems and priors. Stored fluxes remain in Jy; conversion to AB
    magnitudes uses the public package conversion.
    """

    import matplotlib.pyplot as plt

    from microcaustics import plotting as mcp

    records = list(records)
    annotations = [""] * len(records) if annotations is None else annotations
    if len(annotations) != len(records):
        raise ValueError("annotations must match the number of records")
    figure = plt.figure(figsize=(14.2, 2.28 * len(records)))
    layout = figure.add_gridspec(
        len(records), 4,
        width_ratios=(5.4, 1.45, 1.15, 1.55), hspace=0.10, wspace=0.32,
    )
    for index, (record, annotation) in enumerate(zip(records, annotations, strict=True)):
        time = record["times_days"]
        map_time = record["map_times_days"] if "map_times_days" in record.files else time
        names = record["band_names"].astype(str).tolist()
        band = names.index("i")
        total_mag = mc.flux_to_magnitude(torch.as_tensor(record["flux"][:, band])).numpy()
        micro_mag = mc.flux_to_magnitude(
            torch.as_tensor(record["flux_microlensing_only"][:, band])
        ).numpy()
        diagnostics = (
            np.log10(np.clip(record["center_magnification"], 1.0e-12, None)),
            record["crossing_events"].astype(float),
            record["center_distance_uas"],
        )

        axis = figure.add_subplot(layout[index, 0])
        axis.plot(time, micro_mag, color="black", lw=1.35, label="Microlensing only")
        axis.plot(
            time, total_mag, color="darkorange", lw=1.15,
            label="Microlensing + intrinsic variability",
        )
        axis.invert_yaxis()
        axis.set_ylabel(f"LC {index + 1}\nbrightness [mag]")
        if annotation:
            axis.text(0.012, 0.08, annotation, transform=axis.transAxes, fontsize=8.5)
        if index == 0:
            axis.legend(loc="upper right", frameon=True, fontsize=9)
        if index < len(records) - 1:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("time [days]")
        mcp.finish_axis(axis)

        labels = (r"$\log_{10}\mu$", "crossing", r"$d_{\rm caustic}$ [$\mu$as]")
        for column, (values, label) in enumerate(
            zip(diagnostics, labels, strict=True), start=1
        ):
            axis = figure.add_subplot(layout[index, column])
            if column == 2:
                axis.step(map_time, values, where="pre", color="black", lw=1.05)
                axis.set_ylim(-0.05, 1.05)
                axis.set_yticks((0, 1))
            else:
                axis.plot(map_time, values, color="black", lw=1.05)
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.set_ylabel(label, fontsize=9, labelpad=2)
            if index == 0 and column == 2:
                axis.set_title("Labels at source center", fontsize=10)
            if index < len(records) - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("time [days]", fontsize=9)
            mcp.finish_axis(axis)
    figure.align_ylabels()
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.065, top=0.98)
    figure.savefig(Path(output_path), bbox_inches="tight")
    return figure


def plot_training_map_gallery(
    frames,
    output_path: str | Path,
    *,
    source_widths_uas=None,
):
    """Plot one consistently normalized labeled map per training realization."""

    import matplotlib.pyplot as plt

    from microcaustics import plotting as mcp

    frames = list(frames)
    widths = [None] * len(frames) if source_widths_uas is None else source_widths_uas
    logs = [
        np.log10(np.clip(frame.magnification_map.numpy(), 1.0e-12, None))
        for frame in frames
    ]
    finite = np.concatenate([values[np.isfinite(values)] for values in logs])
    vmin, vmax = np.percentile(finite, (0.25, 99.75))
    figure = plt.figure(figsize=(2.96 * len(frames), 3.15))
    layout = figure.add_gridspec(
        1, len(frames) + 1,
        width_ratios=(*([1] * len(frames)), 0.045), wspace=0.06,
    )
    axes = [figure.add_subplot(layout[0, index]) for index in range(len(frames))]
    colorbar_axis = figure.add_subplot(layout[0, -1])
    for index, (axis, frame, width) in enumerate(zip(axes, frames, widths, strict=True)):
        scale_bar = 2.0 if width is None or float(width) >= 4.0 else 0.5
        mcp.plot_magnification_map(
            frame.magnification_map, ax=axis, caustics=frame.caustics.caustics,
            log10=True, vmin=float(vmin), vmax=float(vmax), colorbar=False,
            scale_bar_uas=scale_bar, show_axes=False,
            title=rf"LC {index + 1}, $t=0$ days",
        )
    colorbar = figure.colorbar(axes[-1].images[-1], cax=colorbar_axis)
    colorbar.set_label(r"$\log_{10}\mu$")
    colorbar.ax.tick_params(direction="in")
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.88)
    figure.savefig(Path(output_path), bbox_inches="tight")
    return figure


@dataclass(frozen=True)
class TrainingSystem:
    """One high-level physical system and its training-set provenance."""

    system: mc.MicrolensingSystem
    metadata: Mapping[str, object]

    @property
    def realization(self) -> mc.MicrolensingRealization:
        """Return the cached stellar and numerical realization."""

        return self.system.realize()

    @property
    def simulation(self) -> mc.MicrolensingSimulation:
        """Expose the low-level simulation for advanced diagnostics."""

        return self.realization.simulation

    @property
    def lens_region(self) -> mc.PlaneRegion:
        """Return the automatically derived lens-plane integration region."""

        return self.realization.lens_region

    @property
    def source_grid(self) -> mc.PlaneGrid:
        """Return the source-plane pixel grid."""

        return self.realization.source_grid

    @property
    def lens_grid(self) -> mc.PlaneGrid:
        """Return the high-resolution caustic grid."""

        return self.realization.lens_grid

    @property
    def source(self) -> mc.PixelatedSource:
        """Return the resolved time-dependent source."""

        source = self.realization.source
        if source is None:  # pragma: no cover - constructor always supplies one
            raise RuntimeError("training systems require a source")
        return source

    @property
    def distances(self) -> mc.LensingDistances:
        """Return the system's angular-diameter distances."""

        return self.system.distances


def runtime_for_device(device: str) -> mc.RuntimeConfig:
    """Select strict Triton on CUDA and the portable eager path elsewhere."""

    cuda = str(device).startswith("cuda")
    return mc.RuntimeConfig(
        device=device,
        backend=mc.Backend.TRITON if cuda else mc.Backend.TORCH_EAGER,
        dtype="float32",
        strict_backend=cuda,
        memory_fraction=0.95,
    )


def _thin_disk(
    distances: mc.LensingDistances,
    *,
    source_redshift: float,
    black_hole_mass_solar: float,
    eddington_ratio: float,
    spin: float,
    inclination_deg: float,
    position_angle_deg: float,
    times_days: torch.Tensor,
    driver_seed: int,
    source_resolution: int,
) -> mc.PixelatedSource:
    """Construct a multiband disk with a reproducible broken-PSD driver."""

    model = mc.ThinDiskModel(
        black_hole_mass_solar=black_hole_mass_solar,
        eddington_ratio=eddington_ratio,
        bands_angstrom=dict(zip(BAND_NAMES, WAVELENGTHS_ANGSTROM, strict=True)),
        source_redshift=source_redshift,
        spin=spin,
        inclination_deg=inclination_deg,
        position_angle_deg=position_angle_deg,
        relativity="approximate",
        support_lamp_fraction=0.1,
        support_corona_height_above_isco_rg=20.0,
        source_grid_shape=source_resolution,
        enclosed_flux_fraction=0.999,
        source_margin=1.05,
    )
    disk = model.pixelate(distances)
    driver = mc.broken_power_law_driving_signal(
        times_days,
        break_timescale_days=200.0,
        alpha_L=1.0,
        alpha_R=3.0,
        standard_deviation=0.10,
        seed=int(driver_seed),
        extrapolation="hold",
    )
    return mc.ModulatedSource(disk, driver, name="variable_thin_disk")


def q2237_b_system(
    *,
    seed: int,
    driver_seed: int,
    device: str,
    times_days: torch.Tensor,
    source_resolution: int = 1024,
    label_resolution: int = 8192,
) -> TrainingSystem:
    """Build one Q2237 image-B-like realization from public package objects."""

    lens_redshift, source_redshift = 0.0395, 1.695
    distances = mc.LensingDistances.from_redshifts(lens_redshift, source_redshift)
    macro = mc.MacroLens(
        convergence=0.391,
        shear=0.391,
        shear_angle_deg=141.73,
        smooth_matter_fraction=0.0,
    )
    black_hole_mass_solar = 10.0**9.08
    spin = 0.74
    source = _thin_disk(
        distances,
        source_redshift=source_redshift,
        black_hole_mass_solar=black_hole_mass_solar,
        eddington_ratio=0.34,
        spin=spin,
        inclination_deg=10.0,
        position_angle_deg=175.0,
        times_days=times_days,
        driver_seed=driver_seed,
        source_resolution=source_resolution,
    )
    population = mc.StellarPopulation.salpeter(
        mean_mass_solar=0.3,
        mass_ratio=100.0,
        kinematics=mc.IsotropicKinematics(dispersion_km_s=180.0),
    )
    system = mc.MicrolensingSystem(
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        macro=macro,
        source=source,
        stellar_population=population,
        duration_days=float(times_days.max()) if times_days.numel() else 0.0,
        light_loss=0.01,
        safety_scale=1.5,
        seed=int(seed),
        runtime=runtime_for_device(device),
        caustic_grid_shape=int(label_resolution),
    )
    realization = system.realize()
    return TrainingSystem(
        system,
        {
            "system": "Q2237+0305",
            "image": "B",
            "seed": int(seed),
            "driver_seed": int(driver_seed),
            "star_count": len(realization.stars),
            "source_fov_uas": realization.source_grid.field_of_view_uas[0],
            "lens_fov_uas": realization.lens_region.field_of_view_uas[0],
            "kappa": macro.convergence,
            "gamma": macro.shear,
        },
    )


def random_system(
    *,
    seed: int,
    device: str,
    times_days: torch.Tensor,
    source_resolution: int = 1024,
    label_resolution: int = 8192,
) -> TrainingSystem:
    """Sample a simple, documented lens-and-disk prior for one realization."""

    rng = np.random.default_rng(int(seed))
    lens_redshift = float(rng.uniform(0.15, 0.75))
    source_redshift = float(rng.uniform(max(1.0, lens_redshift + 0.4), 3.0))
    # Avoid the nearly singular macro eigenvalue regime in this compact demo.
    # Production studies may deliberately supply a broader scientific prior.
    for _ in range(10_000):
        kappa = float(rng.uniform(0.20, 0.65))
        gamma = float(rng.uniform(0.10, 0.60))
        if min(abs(1.0 - kappa - gamma), abs(1.0 - kappa + gamma)) > 0.12:
            break
    else:  # pragma: no cover - effectively impossible with the prior above
        raise RuntimeError("failed to sample a stable macroimage")
    smooth_fraction = float(rng.uniform(0.0, 0.5))
    shear_angle_deg = float(rng.uniform(0.0, 180.0))
    log_mass = float(rng.uniform(7.5, 9.5))
    eddington_ratio = float(10.0 ** rng.uniform(-1.3, -0.05))
    spin = float(rng.uniform(-0.8, 0.95))
    # Isotropic orientations truncated at 60 degrees.  More edge-on type-1
    # quasars are both less representative and increasingly sensitive to
    # obscuration physics absent from this compact thin-disk demonstration.
    inclination = float(np.degrees(np.arccos(rng.uniform(0.5, 1.0))))
    disk_angle = float(rng.uniform(0.0, 180.0))
    distances = mc.LensingDistances.from_redshifts(lens_redshift, source_redshift)
    macro = mc.MacroLens(
        convergence=kappa,
        shear=gamma,
        shear_angle_deg=shear_angle_deg,
        smooth_matter_fraction=smooth_fraction,
    )
    black_hole_mass_solar = 10.0**log_mass
    source = _thin_disk(
        distances,
        source_redshift=source_redshift,
        black_hole_mass_solar=black_hole_mass_solar,
        eddington_ratio=eddington_ratio,
        spin=spin,
        inclination_deg=inclination,
        position_angle_deg=disk_angle,
        times_days=times_days,
        driver_seed=seed + 200_000,
        source_resolution=source_resolution,
    )
    population = mc.StellarPopulation.salpeter(
        mean_mass_solar=0.3,
        mass_ratio=100.0,
        kinematics=mc.IsotropicKinematics(dispersion_km_s=180.0),
    )
    system = mc.MicrolensingSystem(
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        macro=macro,
        source=source,
        stellar_population=population,
        duration_days=float(times_days.max()) if times_days.numel() else 0.0,
        light_loss=0.01,
        safety_scale=1.5,
        seed=int(seed + 100_000),
        runtime=runtime_for_device(device),
        caustic_grid_shape=int(label_resolution),
    )
    realization = system.realize()
    return TrainingSystem(
        system,
        {
            "seed": int(seed),
            "lens_redshift": lens_redshift,
            "source_redshift": source_redshift,
            "kappa": kappa,
            "gamma": gamma,
            "smooth_matter_fraction": smooth_fraction,
            "shear_angle_deg": shear_angle_deg,
            "log10_black_hole_mass_solar": log_mass,
            "eddington_ratio": eddington_ratio,
            "spin": spin,
            "inclination_deg": inclination,
            "disk_position_angle_deg": disk_angle,
            "source_fov_uas": realization.source_grid.field_of_view_uas[0],
            "lens_fov_uas": realization.lens_region.field_of_view_uas[0],
            "star_count": len(realization.stars),
        },
    )


def generate_labeled_example(
    system: TrainingSystem,
    map_times_days: torch.Tensor,
    flux_times_days: torch.Tensor,
    *,
    rays: int,
) -> tuple[mc.MultirateLabeledLightCurve, np.ndarray, float]:
    """Generate sparse dynamic maps/labels and a fine-cadence light curve."""

    method = mc.production_ipm_config(rays=int(rays))
    caustics = mc.CausticConfig(
        far_field_approx=method.far_field_approx,
        jacobian_chunk_size=1_048_576,
        anchor_count=9,
        gauge_count=9,
    )
    center_magnifications: list[torch.Tensor] = []

    def retain_center_magnification(_index, frame) -> None:
        values = frame.magnification_map.values
        # Retain one device scalar per frame and transfer the complete series
        # once after the streamed calculation.  Calling ``.cpu()`` here would
        # serialize all 147 GPU frames and distort the warmed runtime.
        center_magnifications.append(
            values[values.shape[0] // 2, values.shape[1] // 2].detach().clone()
        )

    system.simulation.runtime.synchronize()
    start = perf_counter()
    result = system.system.multirate_light_curve_with_labels(
        map_times_days,
        flux_times_days,
        method=method,
        caustics=caustics,
        map_observer=retain_center_magnification,
    )
    system.simulation.runtime.synchronize()
    elapsed = perf_counter() - start
    center_series = (
        torch.stack(center_magnifications).cpu().numpy().astype(np.float32, copy=False)
    )
    return result, center_series, elapsed


class _CenterMagnificationCollector:
    """Collect one device scalar per streamed map without framewise transfers."""

    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []

    def __call__(self, _index, frame) -> None:
        values = frame.magnification_map.values
        self.values.append(
            values[values.shape[0] // 2, values.shape[1] // 2].detach().clone()
        )

    def reset(self) -> None:
        """Discard a partial attempt before CUDA OOM backoff retries it."""

        self.values.clear()

    def numpy(self) -> np.ndarray:
        """Transfer the complete collected series once."""

        return torch.stack(self.values).cpu().numpy().astype(np.float32, copy=False)


def generate_labeled_examples(
    systems: Sequence[TrainingSystem],
    map_times_days: torch.Tensor,
    flux_times_days: torch.Tensor,
    *,
    rays: int,
    curves_per_batch: int,
) -> tuple[
    tuple[tuple[mc.MultirateLabeledLightCurve, np.ndarray, float], ...],
    mc.IndependentLightCurveBatch,
]:
    """Generate independent labeled systems concurrently on one device."""

    systems = tuple(systems)
    if not systems:
        raise ValueError("at least one training system is required")
    method = mc.production_ipm_config(rays=int(rays))
    caustics = mc.CausticConfig(
        far_field_approx=method.far_field_approx,
        jacobian_chunk_size=1_048_576,
        anchor_count=9,
        gauge_count=9,
    )
    collectors = tuple(_CenterMagnificationCollector() for _ in systems)
    batch = mc.batched_system_light_curves(
        tuple(system.system for system in systems),
        map_times_days,
        flux_times_days,
        curves_per_batch=int(curves_per_batch),
        include_labels=True,
        method=method,
        caustics=caustics,
        map_observers=collectors,
        oom_backoff=True,
    )
    amortized = batch.seconds_per_curve
    generated = tuple(
        (result, collector.numpy(), amortized)
        for result, collector in zip(batch.light_curves, collectors, strict=True)
    )
    return generated, batch


def save_example(
    output_dir: Path,
    index: int,
    result: mc.LabeledLightCurve | mc.MultirateLabeledLightCurve,
    center_magnifications: np.ndarray,
    source: mc.PixelatedSource,
    metadata: Mapping[str, object],
    *,
    runtime_seconds: float,
    worker: int,
) -> Path:
    """Write one compact, self-describing training example and metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"light_curve_{int(index):05d}"
    curve = result.light_curve
    if isinstance(source, mc.ModulatedSource):
        amplitudes = source.signal.amplitudes(
            curve.times_days,
            bands=len(curve.band_names),
            device=curve.flux.device,
            dtype=curve.flux.dtype,
        )
    else:
        amplitudes = torch.ones_like(curve.flux)
    microlensing_only_flux = curve.flux / amplitudes
    static_unlensed_flux = curve.unlensed_flux / amplitudes
    center_distances = np.asarray(
        [frame.labels.center_distance_uas for frame in result.caustics],
        dtype=np.float32,
    )
    # The public label product caps no-in-field-caustic cases at the inscribed
    # source radius and records explicitly that they are right-censored.
    center_distance_cap_uas = 0.5 * float(metadata["source_fov_uas"])
    center_distance_censored = np.asarray(
        [frame.labels.center_distance_censored for frame in result.caustics],
        dtype=bool,
    )
    center_distances_capped = np.where(
        np.isfinite(center_distances),
        np.minimum(center_distances, center_distance_cap_uas),
        center_distance_cap_uas,
    ).astype(np.float32, copy=False)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        times_days=curve.times_days.detach().cpu().numpy(),
        map_times_days=(
            result.map_times_days.detach().cpu().numpy()
            if isinstance(result, mc.MultirateLabeledLightCurve)
            else curve.times_days.detach().cpu().numpy()
        ),
        flux=curve.flux.detach().cpu().numpy(),
        unlensed_flux=curve.unlensed_flux.detach().cpu().numpy(),
        flux_microlensing_only=microlensing_only_flux.detach().cpu().numpy(),
        unlensed_flux_static=static_unlensed_flux.detach().cpu().numpy(),
        driver_amplitudes=amplitudes.detach().cpu().numpy(),
        center_magnification=np.asarray(center_magnifications, dtype=np.float32),
        center_distance_uas=center_distances,
        center_distance_capped_uas=center_distances_capped,
        center_distance_censored=center_distance_censored,
        center_distance_cap_uas=np.asarray(center_distance_cap_uas, dtype=np.float32),
        crossing_labels=result.crossing_labels.numpy(),
        crossing_events=result.crossing_events.numpy(),
        band_names=np.asarray(curve.band_names),
    )
    payload = {
        **dict(metadata),
        "index": int(index),
        "worker": int(worker),
        "runtime_seconds": float(runtime_seconds),
        "map_epochs": int(len(result.caustics)),
        "flux_epochs": int(curve.times_days.numel()),
        "map_cadence_days": (
            float(result.map_times_days[1] - result.map_times_days[0])
            if isinstance(result, mc.MultirateLabeledLightCurve)
            and result.map_times_days.numel() > 1
            else None
        ),
        "source_cadence_days": (
            float(curve.times_days[1] - curve.times_days[0])
            if curve.times_days.numel() > 1
            else None
        ),
        "timing_components_seconds": dict(curve.timing.component_seconds),
        "steady_seconds": float(curve.timing.steady_seconds),
        "method": asdict(mc.production_ipm_config()),
        "dynamic_schedule": asdict(
            mc.production_dynamic_config(temporal_batch_size=30)
        ),
    }
    # Convert enum-like values and nested tensors only through JSON's explicit
    # string fallback. Numerical arrays stay in the NPZ rather than metadata.
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return stem.with_suffix(".npz")
