"""Typed outputs returned by public simulation functions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from .geometry import ImagePlaneGrid, PlaneGrid


@dataclass(frozen=True)
class TimingBreakdown:
    """Compile, warmup, and steady-state timings for one calculation.

    ``collected`` is false when the runtime disabled profiling. In that mode,
    numerical work is unchanged but timing-only accelerator barriers are not
    inserted and the numeric timing fields must not be interpreted.
    """

    compile_seconds: float = 0.0
    warmup_seconds: float = 0.0
    steady_seconds: float = 0.0
    component_seconds: Mapping[str, float] = field(default_factory=dict)
    peak_device_memory_bytes: int | None = None
    collected: bool = False

    @property
    def delivered_seconds(self) -> float:
        """Runtime normally relevant to repeated production calculations."""

        return float(self.steady_seconds)


@dataclass(frozen=True)
class MagnificationMap:
    """A magnification field and complete numerical provenance."""

    values: torch.Tensor
    grid: PlaneGrid
    time_days: float = 0.0
    method: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values)
        if tuple(values.shape) != self.grid.shape:
            raise ValueError(
                f"map shape {tuple(values.shape)} does not match grid {self.grid.shape}"
            )
        if not values.is_floating_point():
            raise TypeError("magnification values must use a floating dtype")
        object.__setattr__(self, "values", values)

    def numpy(self) -> np.ndarray:
        """Return a detached CPU NumPy view of the magnification values."""

        return self.values.detach().cpu().numpy()


@dataclass(frozen=True)
class LightCurve:
    """Lensed and optional unlensed multiband flux densities in Jy."""

    times_days: torch.Tensor
    flux: torch.Tensor
    band_names: tuple[str, ...]
    unlensed_flux: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    maps: Mapping[float, MagnificationMap] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.times_days)
        flux = torch.as_tensor(self.flux)
        if times.ndim != 1 or flux.ndim != 2:
            raise ValueError("times and flux must have shapes [time] and [time, band]")
        if flux.shape != (times.numel(), len(self.band_names)):
            raise ValueError("flux dimensions must match times and band_names")
        if self.unlensed_flux is not None:
            unlensed = torch.as_tensor(self.unlensed_flux)
            if unlensed.shape != flux.shape:
                raise ValueError("unlensed_flux must match flux shape")
            object.__setattr__(self, "unlensed_flux", unlensed)
        object.__setattr__(self, "times_days", times)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "maps", dict(self.maps))


@dataclass(frozen=True)
class RenderedMacroImage:
    """A noiseless or observed multiband macro-lensed image.

    Arrays use shape ``[y, x, band]``. ``values`` is the delivered image;
    without a noise model it equals ``noiseless_values``. Component arrays are
    retained only when requested by the renderer.
    """

    values: torch.Tensor
    noiseless_values: torch.Tensor
    grid: ImagePlaneGrid
    band_names: tuple[str, ...]
    wavelengths_angstrom: tuple[float, ...]
    time_days: float = 0.0
    variance: torch.Tensor | None = None
    component_values: Mapping[str, torch.Tensor] = field(default_factory=dict)
    units: str = "arbitrary pixel flux"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (*self.grid.shape, len(self.band_names))
        values = torch.as_tensor(self.values)
        noiseless = torch.as_tensor(self.noiseless_values)
        if values.shape != expected or noiseless.shape != expected:
            raise ValueError(f"rendered arrays must have shape {expected}")
        if not values.is_floating_point() or not noiseless.is_floating_point():
            raise TypeError("rendered images must use floating dtypes")
        if len(self.wavelengths_angstrom) != len(self.band_names):
            raise ValueError("wavelengths and band names must have equal length")
        if self.variance is not None:
            variance = torch.as_tensor(self.variance)
            if (
                variance.shape != expected
                or not torch.isfinite(variance).all()
                or bool(torch.any(variance < 0))
            ):
                raise ValueError("variance must be non-negative and match the image")
            object.__setattr__(self, "variance", variance)
        for name, component in self.component_values.items():
            if not name or torch.as_tensor(component).shape != expected:
                raise ValueError("retained components must be named image arrays")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "noiseless_values", noiseless)

    def numpy(self) -> np.ndarray:
        """Return a detached CPU NumPy copy of the delivered image."""

        return self.values.detach().cpu().numpy()


@dataclass(frozen=True)
class TransferFunction:
    """One steady-state or microlensing-weighted response function.

    ``values`` has shape ``[delay_bin, band]``. Mean delays are calculated
    from the centers of the supplied bins and are therefore reported in days.
    """

    delay_edges_days: torch.Tensor
    values: torch.Tensor
    mean_delays_days: torch.Tensor
    band_names: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        edges = torch.as_tensor(self.delay_edges_days)
        values = torch.as_tensor(self.values)
        means = torch.as_tensor(self.mean_delays_days)
        if edges.ndim != 1 or edges.numel() < 2 or not bool(
            torch.all(edges[1:] > edges[:-1])
        ):
            raise ValueError("delay_edges_days must be strictly increasing")
        expected = (edges.numel() - 1, len(self.band_names))
        if values.shape != expected:
            raise ValueError("transfer values must have shape [delay_bin, band]")
        if means.shape != (len(self.band_names),):
            raise ValueError("mean delays must contain one value per band")
        object.__setattr__(self, "delay_edges_days", edges)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mean_delays_days", means)


@dataclass(frozen=True)
class TransferFunctionSeries:
    """Time-dependent response functions from a dynamic microlens field."""

    times_days: torch.Tensor
    delay_edges_days: torch.Tensor
    values: torch.Tensor
    mean_delays_days: torch.Tensor
    band_names: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.times_days)
        edges = torch.as_tensor(self.delay_edges_days)
        values = torch.as_tensor(self.values)
        means = torch.as_tensor(self.mean_delays_days)
        expected = (times.numel(), edges.numel() - 1, len(self.band_names))
        if times.ndim != 1 or edges.ndim != 1 or values.shape != expected:
            raise ValueError("transfer-function series dimensions are inconsistent")
        if means.shape != (times.numel(), len(self.band_names)):
            raise ValueError("mean delays must have shape [time, band]")
        if edges.numel() < 2 or not bool(torch.all(edges[1:] > edges[:-1])):
            raise ValueError("delay_edges_days must be strictly increasing")
        object.__setattr__(self, "times_days", times)
        object.__setattr__(self, "delay_edges_days", edges)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mean_delays_days", means)

    def at(self, index: int) -> TransferFunction:
        """Return one epoch as an independent transfer-function product."""

        return TransferFunction(
            delay_edges_days=self.delay_edges_days,
            values=self.values[index],
            mean_delays_days=self.mean_delays_days[index],
            band_names=self.band_names,
            metadata={
                **self.metadata,
                "time_days": float(self.times_days[index]),
                "series_index": int(index),
            },
            timing=self.timing,
        )


@dataclass(frozen=True)
class CausticField:
    """Lens-plane critical segments and mapped source-plane caustics.

    Segment arrays have shape ``[segment, endpoint, coordinate]`` with two
    endpoints and Cartesian ``(x, y)`` coordinates in microarcseconds.
    Individual segments are retained because winding and distance queries do
    not require potentially fragile polyline stitching.
    """

    critical_segments_uas: torch.Tensor
    caustic_segments_uas: torch.Tensor
    lens_grid: PlaneGrid
    time_days: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    invalid_segment_mask: torch.Tensor | None = None

    @classmethod
    def from_unordered_closed_segments(
        cls,
        critical_segments_uas,
        caustic_segments_uas,
        lens_grid: PlaneGrid,
        *,
        closure_tolerance_uas: float = 1.0e-5,
        time_days: float = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> CausticField:
        """Construct a field from unordered external closed-curve segments.

        Segment connectivity is recovered in the lens plane, paired mapped
        caustic endpoints are preserved, and every closed component receives
        a positive global orientation suitable for signed winding queries.
        """

        from .caustics import orient_mapped_closed_segments

        critical, caustic = orient_mapped_closed_segments(
            critical_segments_uas,
            caustic_segments_uas,
            closure_tolerance_uas=closure_tolerance_uas,
        )
        return cls(
            critical,
            caustic,
            lens_grid,
            time_days=float(time_days),
            metadata={
                **dict(metadata or {}),
                "segment_registration": "closed_lens_plane_components",
                "closure_tolerance_uas": float(closure_tolerance_uas),
                "orientation": "positive_source_plane_signed_area",
            },
        )

    def __post_init__(self) -> None:
        critical = torch.as_tensor(self.critical_segments_uas)
        caustic = torch.as_tensor(self.caustic_segments_uas)
        if critical.ndim != 3 or tuple(critical.shape[1:]) != (2, 2):
            raise ValueError("critical segments must have shape [segment, 2, 2]")
        if caustic.shape != critical.shape:
            raise ValueError("caustic and critical segment shapes must match")
        if not critical.is_floating_point() or not caustic.is_floating_point():
            raise TypeError("caustic coordinates must use a floating dtype")
        invalid = self.invalid_segment_mask
        if invalid is not None:
            invalid = torch.as_tensor(invalid, device=caustic.device, dtype=torch.bool)
            if invalid.shape != (critical.shape[0],):
                raise ValueError("invalid_segment_mask must have shape [segment]")
        object.__setattr__(self, "critical_segments_uas", critical)
        object.__setattr__(self, "caustic_segments_uas", caustic)
        object.__setattr__(self, "invalid_segment_mask", invalid)

    @property
    def segment_count(self) -> int:
        """Number of independently represented critical/caustic segments."""

        return int(self.caustic_segments_uas.shape[0])

    @property
    def source_length_uas(self) -> float:
        """Total length of the piecewise-linear source-plane caustics."""

        if self.segment_count == 0:
            return 0.0
        lengths = torch.linalg.vector_norm(
            self.caustic_segments_uas[:, 1] - self.caustic_segments_uas[:, 0],
            dim=1,
        )
        return float(lengths.sum().detach().cpu())

    def crossing_parity(
        self,
        points_uas,
        *,
        point_chunk_size: int = 4096,
        segment_chunk_size: int = 16384,
    ) -> torch.Tensor:
        """Return modulo-two winding labels for arbitrary source-plane points.

        The half-open vertex rule counts a shared segment endpoint exactly
        once. Labels assume the supplied caustic field is complete and closed;
        source fields deliberately clipped through caustics require the later
        anchor/gauge boundary-offset interface.
        """

        from .caustics import crossing_parity

        return crossing_parity(
            self.caustic_segments_uas,
            points_uas,
            point_chunk_size=point_chunk_size,
            segment_chunk_size=segment_chunk_size,
        )

    def center_parity(self) -> int:
        """Return the binary caustic-crossing label at the source origin."""

        point = self.caustic_segments_uas.new_zeros((1, 2))
        return int(self.crossing_parity(point)[0].detach().cpu())

    def winding_number(
        self,
        points_uas,
        *,
        point_chunk_size: int = 4096,
        segment_chunk_size: int = 16384,
    ) -> torch.Tensor:
        """Return signed winding numbers of the oriented caustic segments.

        Marching-squares segments preserve a consistent determinant-side
        orientation through their lens-equation mapping. Imported segment
        fields must provide their own consistent orientation for this signed
        quantity. Parity labels remain valid without that requirement.
        """

        from .caustics import winding_number

        return winding_number(
            self.caustic_segments_uas,
            points_uas,
            point_chunk_size=point_chunk_size,
            segment_chunk_size=segment_chunk_size,
        )

    def center_winding_number(self) -> int:
        """Return the signed caustic winding number at the source origin."""

        point = self.caustic_segments_uas.new_zeros((1, 2))
        return int(self.winding_number(point)[0].detach().cpu())

    def winding_map(
        self,
        grid: PlaneGrid,
        *,
        orientation: str = "signed",
        point_chunk_size: int = 4096,
        segment_chunk_size: int = 16384,
    ) -> LabelMap:
        """Evaluate winding number at every cell center of ``grid``.

        ``orientation='signed'`` preserves the supplied curve orientation.
        ``orientation='positive'`` applies one global sign convention so the
        dominant nonzero interior is positive. It does not take an absolute
        value independently at each pixel and therefore preserves topology.
        """

        if orientation not in {"signed", "positive"}:
            raise ValueError("orientation must be 'signed' or 'positive'")

        x, y = grid.mesh(
            device=self.caustic_segments_uas.device,
            dtype=self.caustic_segments_uas.dtype,
        )
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        values = self.winding_number(
            points,
            point_chunk_size=point_chunk_size,
            segment_chunk_size=segment_chunk_size,
        ).reshape(grid.shape)
        sign_flipped = False
        if orientation == "positive":
            nonzero = values[values != 0]
            if nonzero.numel() and int(torch.median(nonzero).detach().cpu()) < 0:
                values = -values
                sign_flipped = True
        return LabelMap(
            values,
            grid,
            metadata={
                "method": "signed_half_open_winding_number",
                "caustic_segments": self.segment_count,
                "oriented_closed_caustic_field_required": True,
                "orientation": orientation,
                "global_sign_flipped": sign_flipped,
            },
        )

    def label_map(
        self,
        grid: PlaneGrid,
        *,
        point_chunk_size: int = 4096,
        segment_chunk_size: int = 16384,
    ) -> LabelMap:
        """Evaluate binary crossing parity at every cell center of ``grid``."""

        x, y = grid.mesh(
            device=self.caustic_segments_uas.device,
            dtype=self.caustic_segments_uas.dtype,
        )
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        labels = self.crossing_parity(
            points,
            point_chunk_size=point_chunk_size,
            segment_chunk_size=segment_chunk_size,
        ).reshape(grid.shape)
        return LabelMap(
            labels,
            grid,
            metadata={
                "method": "half_open_crossing_parity",
                "caustic_segments": self.segment_count,
                "complete_closed_caustic_field_required": True,
            },
        )

    def distance(
        self,
        points_uas,
        *,
        point_chunk_size: int = 4096,
        segment_chunk_size: int = 16384,
    ) -> torch.Tensor:
        """Return the minimum Euclidean distance to a caustic segment."""

        from .caustics import distance_to_segments

        return distance_to_segments(
            self.caustic_segments_uas,
            points_uas,
            point_chunk_size=point_chunk_size,
            segment_chunk_size=segment_chunk_size,
        )

    def distance_map(
        self,
        grid: PlaneGrid,
        *,
        point_chunk_size: int = 4096,
        segment_chunk_size: int = 16384,
    ) -> DistanceMap:
        """Evaluate source-plane distance to the nearest caustic on ``grid``."""

        x, y = grid.mesh(
            device=self.caustic_segments_uas.device,
            dtype=self.caustic_segments_uas.dtype,
        )
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        values = self.distance(
            points,
            point_chunk_size=point_chunk_size,
            segment_chunk_size=segment_chunk_size,
        ).reshape(grid.shape)
        return DistanceMap(
            values,
            grid,
            metadata={
                "method": "minimum_segment_distance",
                "caustic_segments": self.segment_count,
                "units": "microarcseconds",
            },
        )


@dataclass(frozen=True)
class LabelMap:
    """Integer-valued source-plane labels evaluated on a regular grid."""

    values: torch.Tensor
    grid: PlaneGrid
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values)
        if tuple(values.shape) != self.grid.shape:
            raise ValueError("label values must match the label grid shape")
        if values.dtype == torch.bool:
            values = values.to(torch.int8)
        if values.is_floating_point() or values.is_complex():
            raise TypeError("label values must use an integer or boolean dtype")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class DistanceMap:
    """Distance to the nearest source-plane caustic in microarcseconds."""

    values_uas: torch.Tensor
    grid: PlaneGrid
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values_uas)
        if tuple(values.shape) != self.grid.shape:
            raise ValueError("distance values must match the distance grid shape")
        if not values.is_floating_point() or bool(torch.any(values < 0)):
            raise ValueError("distance values must be non-negative floating values")
        object.__setattr__(self, "values_uas", values)


@dataclass(frozen=True)
class AnchorGaugeLabels:
    """Robust source-region label diagnostics for one caustic frame.

    ``raw_center_label`` is the majority class in the frame's local anchor
    gauge. ``center_label`` additionally applies the temporal XOR inferred
    from safe gauge probes. The absolute zero/one naming is conventional;
    transitions of ``center_label`` are the physical crossing observable.
    ``center_distance_uas`` is capped at the inscribed source-field radius;
    ``center_distance_censored`` identifies values that represent the lower
    bound ``d_caustic >= R_src`` rather than an in-field measurement.
    """

    raw_center_label: int
    center_label: int
    center_crossing: bool
    center_distance_uas: float
    center_vote_count: int
    center_valid_count: int
    gauge_labels: torch.Tensor
    gauge_distances_uas: torch.Tensor
    gauge_vote_counts: torch.Tensor
    gauge_valid_counts: torch.Tensor
    anchor_offsets: torch.Tensor
    anchor_points_uas: torch.Tensor
    gauge_points_uas: torch.Tensor
    center_distance_censored: bool = False
    frame_xor: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        gauge_labels = torch.as_tensor(self.gauge_labels, dtype=torch.int8)
        gauge_distances = torch.as_tensor(self.gauge_distances_uas)
        gauge_votes = torch.as_tensor(self.gauge_vote_counts, dtype=torch.int32)
        gauge_valid = torch.as_tensor(self.gauge_valid_counts, dtype=torch.int32)
        offsets = torch.as_tensor(self.anchor_offsets, dtype=torch.int8)
        anchors = torch.as_tensor(self.anchor_points_uas)
        gauges = torch.as_tensor(self.gauge_points_uas)
        gauge_count = int(gauge_labels.numel())
        if any(int(value.numel()) != gauge_count for value in (
            gauge_distances,
            gauge_votes,
            gauge_valid,
        )):
            raise ValueError("all gauge diagnostics must have one value per gauge")
        if anchors.ndim != 2 or anchors.shape[-1] != 2:
            raise ValueError("anchor_points_uas must have shape [anchor, 2]")
        if gauges.shape != (gauge_count, 2):
            raise ValueError("gauge_points_uas must have shape [gauge, 2]")
        if offsets.shape != (anchors.shape[0],):
            raise ValueError("anchor_offsets must have one value per anchor")
        if int(self.center_vote_count) > int(self.center_valid_count):
            raise ValueError("center votes cannot exceed valid anchors")
        center_distance = float(self.center_distance_uas)
        if not math.isfinite(center_distance) or center_distance < 0.0:
            raise ValueError("center_distance_uas must be finite and non-negative")
        object.__setattr__(self, "center_distance_uas", center_distance)
        object.__setattr__(
            self,
            "center_distance_censored",
            bool(self.center_distance_censored),
        )
        object.__setattr__(self, "gauge_labels", gauge_labels)
        object.__setattr__(self, "gauge_distances_uas", gauge_distances)
        object.__setattr__(self, "gauge_vote_counts", gauge_votes)
        object.__setattr__(self, "gauge_valid_counts", gauge_valid)
        object.__setattr__(self, "anchor_offsets", offsets)
        object.__setattr__(self, "anchor_points_uas", anchors)
        object.__setattr__(self, "gauge_points_uas", gauges)


@dataclass(frozen=True)
class LabeledCausticFrame:
    """Critical curves, caustics, and production labels for one epoch."""

    caustics: CausticField
    labels: AnchorGaugeLabels
    label_map: LabelMap | None = None
    distance_map: DistanceMap | None = None

    @property
    def time_days(self) -> float:
        """Epoch inherited from the caustic field."""

        return float(self.caustics.time_days)


@dataclass(frozen=True)
class LabeledMapFrame:
    """One magnification map paired with same-epoch caustics and labels."""

    magnification_map: MagnificationMap
    caustics: LabeledCausticFrame

    def __post_init__(self) -> None:
        if abs(self.magnification_map.time_days - self.caustics.time_days) > 1.0e-9:
            raise ValueError("map and labeled caustic epochs must match")

    @property
    def time_days(self) -> float:
        """Shared map and caustic epoch in days."""

        return float(self.magnification_map.time_days)


def _cap_center_distances(
    raw_distances_uas: torch.Tensor,
    maximum_distance_uas: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an explicit finite-field right-censoring threshold."""

    maximum = float(maximum_distance_uas)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_distance_uas must be positive and finite")
    raw = torch.as_tensor(raw_distances_uas, dtype=torch.float64)
    censored = ~torch.isfinite(raw) | (raw > maximum)
    capped = torch.where(
        torch.isfinite(raw),
        torch.clamp(raw, max=maximum),
        torch.full_like(raw, maximum),
    )
    return capped, censored


@dataclass(frozen=True)
class LabeledLightCurve:
    """A streamed finite-source light curve and its per-epoch labels."""

    light_curve: LightCurve
    caustics: tuple[LabeledCausticFrame, ...]

    @property
    def maps(self) -> Mapping[float, MagnificationMap]:
        """Selected magnification maps retained with the light curve."""

        return self.light_curve.maps

    def __post_init__(self) -> None:
        if len(self.caustics) != int(self.light_curve.times_days.numel()):
            raise ValueError("caustic frames must match the light-curve time axis")
        for time_days, frame in zip(
            self.light_curve.times_days.detach().cpu().tolist(),
            self.caustics,
            strict=True,
        ):
            if abs(float(time_days) - frame.time_days) > 1.0e-4:
                raise ValueError("caustic and light-curve epochs must match")

    @property
    def crossing_labels(self) -> torch.Tensor:
        """Temporally aligned binary label at the source center."""

        return torch.tensor(
            [frame.labels.center_label for frame in self.caustics],
            dtype=torch.int8,
        )

    @property
    def crossing_events(self) -> torch.Tensor:
        """Boolean caustic-crossing transition flags."""

        return torch.tensor(
            [frame.labels.center_crossing for frame in self.caustics],
            dtype=torch.bool,
        )

    @property
    def center_distances_uas(self) -> torch.Tensor:
        """Finite, source-radius-capped center-to-caustic distances."""

        return torch.tensor(
            [frame.labels.center_distance_uas for frame in self.caustics],
            dtype=torch.float64,
        )

    @property
    def center_distance_censored(self) -> torch.Tensor:
        """True where the nearest caustic lies outside the source-radius cap."""

        return torch.tensor(
            [frame.labels.center_distance_censored for frame in self.caustics],
            dtype=torch.bool,
        )

    def capped_center_distances_uas(
        self,
        maximum_distance_uas: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite center distances and their right-censoring mask."""

        distances, censored = _cap_center_distances(
            self.center_distances_uas,
            maximum_distance_uas,
        )
        return distances, censored | self.center_distance_censored


@dataclass(frozen=True)
class MultirateLabeledLightCurve:
    """Fine-cadence light curve with labels at its sparse map epochs.

    ``light_curve`` may be sampled more finely than ``caustics``. Each
    caustic frame is aligned to one dynamic magnification-map epoch, while the
    light curve is obtained by interpolating the two bracketing map--source
    contractions. This preserves the production multirate calculation without
    pretending that caustics were recomputed at every fine source epoch.
    """

    light_curve: LightCurve
    caustics: tuple[LabeledCausticFrame, ...]

    @property
    def maps(self) -> Mapping[float, MagnificationMap]:
        """Selected magnification maps retained with the light curve."""

        return self.light_curve.maps

    def __post_init__(self) -> None:
        if not self.caustics:
            raise ValueError("at least one labeled map epoch is required")
        times = torch.as_tensor(self.light_curve.times_days).detach().cpu()
        caustic_times = torch.tensor(
            [frame.time_days for frame in self.caustics], dtype=torch.float64
        )
        if caustic_times.numel() > 1 and not bool(
            torch.all(caustic_times[1:] > caustic_times[:-1])
        ):
            raise ValueError("caustic map epochs must be strictly increasing")
        tolerance = 1.0e-4
        if (
            float(caustic_times[0]) < float(times[0]) - tolerance
            or float(caustic_times[-1]) > float(times[-1]) + tolerance
        ):
            raise ValueError("caustic map epochs must lie within the light curve")

    @property
    def map_times_days(self) -> torch.Tensor:
        """Times of the dynamic maps and corresponding caustic labels."""

        return torch.tensor(
            [frame.time_days for frame in self.caustics], dtype=torch.float64
        )

    @property
    def crossing_labels(self) -> torch.Tensor:
        """Temporally aligned binary labels at the source center."""

        return torch.tensor(
            [frame.labels.center_label for frame in self.caustics],
            dtype=torch.int8,
        )

    @property
    def crossing_events(self) -> torch.Tensor:
        """Boolean center-label transitions at the sparse map epochs."""

        return torch.tensor(
            [frame.labels.center_crossing for frame in self.caustics],
            dtype=torch.bool,
        )

    @property
    def center_distances_uas(self) -> torch.Tensor:
        """Finite, source-radius-capped center-to-caustic distances."""

        return torch.tensor(
            [frame.labels.center_distance_uas for frame in self.caustics],
            dtype=torch.float64,
        )

    @property
    def center_distance_censored(self) -> torch.Tensor:
        """True where the nearest caustic lies outside the source-radius cap."""

        return torch.tensor(
            [frame.labels.center_distance_censored for frame in self.caustics],
            dtype=torch.bool,
        )

    def capped_center_distances_uas(
        self,
        maximum_distance_uas: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite center distances and their right-censoring mask.

        Parameters
        ----------
        maximum_distance_uas:
            Largest resolved distance. For a finite rectangular source field,
            half its smaller field-of-view dimension is a conservative choice:
            it is the radius of the largest centered circle contained in the
            field.

        Returns
        -------
        distances, censored:
            ``distances`` is ``min(raw_distance, maximum_distance_uas)`` and
            ``censored`` is true wherever the raw value is non-finite or lies
            beyond the cap. The cap therefore denotes a lower bound rather than
            a claimed caustic location.
        """

        distances, censored = _cap_center_distances(
            self.center_distances_uas,
            maximum_distance_uas,
        )
        return distances, censored | self.center_distance_censored


@dataclass(frozen=True)
class MultiImageMapFrame:
    """One source-independent map identified with its macroimage."""

    image_name: str
    magnification_map: MagnificationMap

    def __post_init__(self) -> None:
        if not isinstance(self.image_name, str) or not self.image_name:
            raise ValueError("image_name must be a non-empty string")

    @property
    def time_days(self) -> float:
        """Return the observer-frame map epoch."""

        return float(self.magnification_map.time_days)


@dataclass(frozen=True)
class MacroImageLightCurve:
    """Light curve and optional caustic labels for one macroimage."""

    image_name: str
    arrival_time_delay_days: float
    light_curve: LightCurve
    caustics: tuple[LabeledCausticFrame, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_name, str) or not self.image_name:
            raise ValueError("image_name must be a non-empty string")
        if not np.isfinite(float(self.arrival_time_delay_days)):
            raise ValueError("arrival_time_delay_days must be finite")
        if self.caustics is not None:
            caustics = tuple(self.caustics)
            if len(caustics) == int(self.light_curve.times_days.numel()):
                LabeledLightCurve(self.light_curve, caustics)
            else:
                MultirateLabeledLightCurve(self.light_curve, caustics)
            object.__setattr__(self, "caustics", caustics)

    @property
    def label_times_days(self) -> torch.Tensor | None:
        """Return sparse map epochs associated with optional labels."""

        if self.caustics is None:
            return None
        return torch.tensor(
            [frame.time_days for frame in self.caustics], dtype=torch.float64
        )

    @property
    def crossing_labels(self) -> torch.Tensor | None:
        """Return aligned center labels when caustics were requested."""

        if self.caustics is None:
            return None
        return torch.tensor(
            [frame.labels.center_label for frame in self.caustics],
            dtype=torch.int8,
        )

    @property
    def crossing_events(self) -> torch.Tensor | None:
        """Return center-label transitions when caustics were requested."""

        if self.caustics is None:
            return None
        return torch.tensor(
            [frame.labels.center_crossing for frame in self.caustics],
            dtype=torch.bool,
        )


@dataclass(frozen=True)
class MultiImageLightCurves:
    """Ordered resolved light curves for one multiply imaged source."""

    images: tuple[MacroImageLightCurve, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images:
            raise ValueError("at least one macroimage light curve is required")
        names = tuple(item.image_name for item in images)
        if len(set(names)) != len(names):
            raise ValueError("macroimage light-curve names must be unique")
        object.__setattr__(self, "images", images)

    @property
    def image_names(self) -> tuple[str, ...]:
        """Return macroimage names in calculation order."""

        return tuple(item.image_name for item in self.images)

    @property
    def arrival_time_delays_days(self) -> torch.Tensor:
        """Return arrival delays in the same order as :attr:`image_names`."""

        return torch.tensor(
            [item.arrival_time_delay_days for item in self.images],
            dtype=torch.float64,
        )

    def __getitem__(self, image_name: str) -> MacroImageLightCurve:
        """Look up one resolved macroimage by name."""

        for item in self.images:
            if item.image_name == image_name:
                return item
        raise KeyError(image_name)

    def flux_tensor(
        self,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Stack flux as ``[image, time, band]`` for a shared cadence.

        Per-image cadences are allowed by the simulation interface. This
        convenience operation therefore validates that time axes and band
        names agree before stacking.
        """

        reference = self.images[0].light_curve
        for item in self.images[1:]:
            curve = item.light_curve
            if curve.band_names != reference.band_names:
                raise ValueError("macroimages do not share the same bands")
            same_shape = curve.times_days.shape == reference.times_days.shape
            same_values = same_shape and torch.allclose(
                curve.times_days.detach().cpu().to(torch.float64),
                reference.times_days.detach().cpu().to(torch.float64),
                rtol=0.0,
                atol=1.0e-6,
            )
            if not same_values:
                raise ValueError("macroimages do not share the same time axis")
        return torch.stack(
            [item.light_curve.flux.to(device=device) for item in self.images]
        )


@dataclass(frozen=True)
class MacroImageTransferFunctions:
    """Microlensing-weighted response functions for one macroimage.

    Values have shape ``[map_epoch, delay_bin, band]``. Delay bins describe
    reverberation relative to emission seen in that macroimage. The separate
    macro arrival delay can be added when placing the response on a shared
    observer timeline.
    """

    image_name: str
    arrival_time_delay_days: float
    map_times_days: torch.Tensor
    delay_edges_days: torch.Tensor
    values: torch.Tensor
    mean_delays_days: torch.Tensor
    band_names: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.map_times_days)
        edges = torch.as_tensor(self.delay_edges_days)
        values = torch.as_tensor(self.values)
        means = torch.as_tensor(self.mean_delays_days)
        if not self.image_name:
            raise ValueError("image_name must be non-empty")
        if not np.isfinite(float(self.arrival_time_delay_days)):
            raise ValueError("arrival_time_delay_days must be finite")
        expected = (times.numel(), edges.numel() - 1, len(self.band_names))
        if times.ndim != 1 or edges.ndim != 1 or tuple(values.shape) != expected:
            raise ValueError("transfer-function dimensions are inconsistent")
        if means.shape != (times.numel(), len(self.band_names)):
            raise ValueError("mean_delays_days must have shape [epoch, band]")
        if edges.numel() < 2 or not bool(torch.all(edges[1:] > edges[:-1])):
            raise ValueError("delay_edges_days must be strictly increasing")
        object.__setattr__(self, "map_times_days", times)
        object.__setattr__(self, "delay_edges_days", edges)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mean_delays_days", means)

    @property
    def observer_mean_delays_days(self) -> torch.Tensor:
        """Return macro arrival delay plus the internal response mean."""

        return self.mean_delays_days + float(self.arrival_time_delay_days)


@dataclass(frozen=True)
class MultiImageTransferFunctions:
    """Ordered microlensing-weighted response functions for macroimages."""

    images: tuple[MacroImageTransferFunctions, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images:
            raise ValueError("at least one macroimage response is required")
        names = tuple(item.image_name for item in images)
        if len(set(names)) != len(names):
            raise ValueError("macroimage response names must be unique")
        object.__setattr__(self, "images", images)

    @property
    def image_names(self) -> tuple[str, ...]:
        """Return macroimage names in calculation order."""

        return tuple(item.image_name for item in self.images)

    def __getitem__(self, image_name: str) -> MacroImageTransferFunctions:
        """Look up the response sequence for one macroimage."""

        for item in self.images:
            if item.image_name == image_name:
                return item
        raise KeyError(image_name)
