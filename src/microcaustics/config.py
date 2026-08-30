"""Typed numerical and runtime settings for microlensing calculations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import torch


class Backend(str, Enum):
    """Requested numerical execution backend.

    ``AUTO`` selects the fastest compatible implementation for each operation.
    A Triton runtime may still use small PyTorch operations around a fused
    Triton kernel. It denotes the primary numerical implementation rather than
    a promise that every tensor operation is written in Triton.
    """

    AUTO = "auto"
    TORCH_EAGER = "torch-eager"
    TORCH_COMPILE = "torch-compile"
    TRITON = "triton"


class ProfilingLevel(str, Enum):
    """Synchronization policy used to collect runtime measurements."""

    OFF = "off"
    TOTAL = "total"
    DETAILED = "detailed"


_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def normalize_dtype(value: str | torch.dtype) -> torch.dtype:
    """Return a supported floating-point dtype.

    Float16 and bfloat16 are deliberately excluded from the initial public
    contract because lens singularities and caustic predicates require more
    dynamic range and precision.
    """

    if isinstance(value, torch.dtype):
        if value in _DTYPES.values():
            return value
    else:
        key = str(value).lower().replace("torch.", "")
        if key in _DTYPES:
            return _DTYPES[key]
    raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True)
class AutoTuningConfig:
    """Optional steady-state temporal, spatial, and memory autotuning.

    Tuning is disabled by default. When enabled, a short representative prefix
    is warmed and benchmarked with a coordinate sweep. Spatial chunks come first,
    then temporal batches. Candidates that OOM, exceed the runtime memory
    budget, or fail numerical verification are rejected. Successful choices
    are cached for the lifetime of the process.
    """

    enabled: bool = False
    tune_temporal_batch: bool = True
    tune_spatial_chunk: bool = True
    temporal_candidates: tuple[int, ...] = (
        1,
        4,
        8,
        12,
        16,
        20,
        24,
        32,
        40,
        49,
        64,
    )
    spatial_candidates: tuple[int, ...] = (
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
    )
    warmup_runs: int = 1
    benchmark_runs: int = 2
    maximum_trial_frames: int = 64
    memory_fraction: float | None = None
    memory_headroom_fraction: float = 0.95
    verify_numerics: bool = True
    verification_rtol: float = 5.0e-5
    verification_atol: float = 5.0e-6
    cache: bool = True

    def __post_init__(self) -> None:
        for name in ("warmup_runs", "benchmark_runs", "maximum_trial_frames"):
            minimum = 0 if name == "warmup_runs" else 1
            if int(getattr(self, name)) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        for name in ("temporal_candidates", "spatial_candidates"):
            values = tuple(dict.fromkeys(int(value) for value in getattr(self, name)))
            if not values or any(value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            object.__setattr__(self, name, values)
        if self.memory_fraction is not None and not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must lie in (0, 1]")
        if not 0.0 < self.memory_headroom_fraction <= 1.0:
            raise ValueError("memory_headroom_fraction must lie in (0, 1]")
        if self.verification_rtol < 0.0 or self.verification_atol < 0.0:
            raise ValueError("verification tolerances must be non-negative")


@dataclass(frozen=True)
class RuntimeConfig:
    """User preferences for device placement and kernel execution.

    Parameters
    ----------
    device:
        ``"auto"``, a PyTorch device string, or a :class:`torch.device`.
        Automatic selection prefers CUDA, then Apple MPS, then CPU.
    backend:
        Execution backend. ``"auto"`` prefers Triton on compatible CUDA
        float32 operations, then compiled PyTorch, then eager PyTorch.
    dtype:
        Compute dtype. Float32 is the production default. Float64 is available
        for validation and precision-sensitive calculations.
    strict_backend:
        If true, an unavailable requested backend raises an exception. If
        false, the resolver selects the best safe fallback and records why.
    memory_fraction:
        Maximum fraction of visible device memory available to package-managed
        batches and temporary workspaces.
    torch_compile_mode:
        Optional mode forwarded to :func:`torch.compile`. ``None`` uses the
        PyTorch default. It has no effect on eager or Triton runtimes.
    profiling:
        ``"off"`` avoids timing-only accelerator barriers. ``"total"``
        synchronizes only around complete public calculations.
        ``"detailed"`` also synchronizes component boundaries.
    """

    device: str | torch.device = "auto"
    backend: Backend | str = Backend.AUTO
    dtype: str | torch.dtype = torch.float32
    strict_backend: bool = False
    memory_fraction: float = 0.85
    torch_compile_mode: str | None = None
    profiling: ProfilingLevel | str = ProfilingLevel.OFF

    def __post_init__(self) -> None:
        backend = Backend(self.backend)
        profiling = ProfilingLevel(self.profiling)
        dtype = normalize_dtype(self.dtype)
        if not 0.0 < float(self.memory_fraction) <= 1.0:
            raise ValueError("memory_fraction must lie in (0, 1]")
        if self.torch_compile_mode is not None and not str(
            self.torch_compile_mode
        ).strip():
            raise ValueError("torch_compile_mode must be non-empty or None")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "profiling", profiling)


@dataclass(frozen=True)
class FarFieldApproxConfig:
    """Local-exact and far-field settings for point-mass deflections.

    ``cells_per_axis`` partitions the lens field for exact-near/far-star
    membership. ``nodes_per_cell_axis`` is the Taylor evaluation lattice
    within each partition cell. The production values are 16 and 8. The two
    controls therefore have different meanings and are not interchangeable.
    """

    enabled: bool = True
    cells_per_axis: int = 16
    nodes_per_cell_axis: int = 8
    exact_radius_cells: float = 1.0
    taylor_order: int = 4
    center_translation_order: int = 10

    def __post_init__(self) -> None:
        if self.cells_per_axis < 2:
            raise ValueError("cells_per_axis must be at least 2")
        if self.nodes_per_cell_axis < 1:
            raise ValueError("nodes_per_cell_axis must be positive")
        if self.exact_radius_cells < 0:
            raise ValueError("exact_radius_cells must be non-negative")
        if not 0 <= self.taylor_order <= 8:
            raise ValueError("taylor_order must lie in [0, 8]")
        if self.center_translation_order < self.taylor_order:
            raise ValueError(
                "center_translation_order must be at least taylor_order"
            )


@dataclass(frozen=True)
class IPMConfig:
    """Inverse polygon mapping configuration.

    ``rays`` is the requested fine base-cell budget. ``scout_ratio`` (``k``)
    coarsens only source-region discovery in tiled mode. ``refinement``
    (``r``) controls true lens-equation samples inside retained cells, while
    ``virtual_refinement`` (``v``) controls their interpolated polygon
    representation without additional lens-equation evaluations. The paper
    production setting is ``N=10M, k=2, r=2, v=4``, but these values are not
    hard-coded.
    """

    rays: int = 10_000_000
    scout_ratio: int = 2
    refinement: int = 2
    virtual_refinement: int = 4
    tiled: bool = True
    # The validated production scout does not enlarge the source aperture in
    # source-pixel units. Its safety margin is instead the one-cell lens-plane
    # support ring below.
    scout_halo_pixels: float = 0.0
    scout_dilation_cells: int = 1
    scout_trace_centers: bool = True
    dual_scout_scalar_correction: bool = False
    cell_chunk_size: int = 65_536
    far_field_approx: FarFieldApproxConfig = field(default_factory=FarFieldApproxConfig)

    def __post_init__(self) -> None:
        for name in (
            "rays",
            "scout_ratio",
            "refinement",
            "virtual_refinement",
            "cell_chunk_size",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.virtual_refinement < self.refinement:
            raise ValueError(
                "virtual_refinement must be at least refinement. Reducing the "
                "mapped node lattice would discard traced information"
            )
        if self.scout_halo_pixels < 0:
            raise ValueError("scout_halo_pixels must be non-negative")
        if self.scout_dilation_cells < 0:
            raise ValueError("scout_dilation_cells must be non-negative")
        if self.dual_scout_scalar_correction and (
            not self.tiled or self.scout_ratio != 2
        ):
            raise ValueError(
                "dual_scout_scalar_correction requires tiled=True and "
                "scout_ratio=2"
            )


@dataclass(frozen=True)
class IRSConfig:
    """Uniform-grid inverse ray-shooting configuration."""

    rays: int = 10_000_000
    ray_chunk_size: int = 262_144
    star_chunk_size: int = 4096
    far_field_approx: FarFieldApproxConfig = field(
        default_factory=lambda: FarFieldApproxConfig(enabled=False)
    )

    def __post_init__(self) -> None:
        if self.rays < 1:
            raise ValueError("rays must be positive")
        if self.ray_chunk_size < 1 or self.star_chunk_size < 1:
            raise ValueError("ray and star chunk sizes must be positive")


@dataclass(frozen=True)
class DynamicConfig:
    """Scheduling and conservative reuse settings for dynamic maps.

    ``temporal_batch_size`` controls the number of map frames presented to a
    fused solver. When omitted, CUDA autotuning begins from the validated
    production size of forty. ``fused_temporal_ipm=False`` retains the readable
    independent-frame path for validation. ``pad_temporal_batches`` permits a
    short CUDA tail to reuse the compiled batch shape. Padded maps are never
    returned.

    ``scout_refresh_frames`` controls an approximate optimization for moving
    point-mass fields. With ``endpoint_union=True``, the selected fine cells
    from both ends of each refresh interval are retained throughout that
    interval. Set the refresh interval to one to recompute every frame.
    Static lens fields are detected and reused exactly.
    """

    temporal_batch_size: int | None = None
    light_curve_batch_size: int | None = None
    fused_temporal_ipm: bool = True
    pad_temporal_batches: bool = True
    scout_refresh_frames: int = 10
    endpoint_union: bool = True
    tuning: AutoTuningConfig = field(default_factory=AutoTuningConfig)
    reuse_static_maps: bool = True
    minimum_cell_chunk_size: int = 1_024

    def __post_init__(self) -> None:
        optional_positive = ("temporal_batch_size", "light_curve_batch_size")
        for name in optional_positive:
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when supplied")
        if self.scout_refresh_frames < 1:
            raise ValueError("scout_refresh_frames must be positive")
        if self.minimum_cell_chunk_size < 1:
            raise ValueError("minimum_cell_chunk_size must be positive")


def production_ipm_config(
    *,
    rays: int = 10_000_000,
    **overrides,
) -> IPMConfig:
    """Return the validated dynamic tiled-IPM configuration.

    High-level static-map calls automatically use a complete ``k=1`` scout.
    Dynamic sequences use the faster ``k=2`` scout plus the one-time
    frame-zero ``k=1`` to ``k=2`` correction. Users therefore do not select a
    static/dynamic mode. ``overrides`` remains available for documented
    numerical experiments.

    Parameters
    ----------
    rays:
        Requested fine base-cell budget.
    **overrides:
        Fields passed to :func:`dataclasses.replace` after constructing the
        validated baseline.
    """

    if int(overrides.get("scout_ratio", 2)) == 1:
        overrides.setdefault("dual_scout_scalar_correction", False)
    config = IPMConfig(
        rays=int(rays),
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        tiled=True,
        scout_halo_pixels=0.0,
        scout_dilation_cells=1,
        dual_scout_scalar_correction=True,
        cell_chunk_size=524_288,
        far_field_approx=FarFieldApproxConfig(
            cells_per_axis=16,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
            taylor_order=4,
            center_translation_order=10,
        ),
    )
    return replace(config, **overrides) if overrides else config


def _production_static_ipm_config(
    *,
    rays: int = 10_000_000,
    **overrides,
) -> IPMConfig:
    """Return the internal complete-scout configuration for one static map."""

    config = production_ipm_config(
        rays=rays,
        scout_ratio=1,
        dual_scout_scalar_correction=False,
    )
    return replace(config, **overrides) if overrides else config


def production_dynamic_config(**overrides) -> DynamicConfig:
    """Return the validated dynamic scheduling configuration.

    The preset uses a forty-frame fused temporal batch and a ten-frame
    endpoint-union scout refresh. It deliberately keeps the far-field time
    stride at one. Every epoch receives independently constructed Taylor
    coefficients.
    """

    config = DynamicConfig(
        temporal_batch_size=40,
        fused_temporal_ipm=True,
        pad_temporal_batches=True,
        scout_refresh_frames=10,
        endpoint_union=True,
    )
    return replace(config, **overrides) if overrides else config


@dataclass(frozen=True)
class CausticConfig:
    """Production critical-curve and anchor/gauge label configuration.

    The determinant grid itself is supplied as a :class:`~microcaustics.PlaneGrid`.
    This object controls how it is evaluated and how finite-field caustics are
    converted into robust source-region labels. Anchors and gauges use distinct,
    reproducibly jittered paths just inside the source boundary. Nine of each is
    the validated production layout.
    """

    far_field_approx: FarFieldApproxConfig = field(default_factory=FarFieldApproxConfig)
    tuning: AutoTuningConfig = field(default_factory=AutoTuningConfig)
    temporal_batch_size: int | None = None
    jacobian_chunk_size: int = 1_048_576
    determinant_cleanup: str = "local"
    minimum_sign_component_pixels: int = 4
    anchor_count: int = 9
    gauge_count: int = 9
    anchor_inset_fraction: float = 0.05
    gauge_inset_fraction: float = 0.08
    anchor_phase: float = 0.5
    gauge_phase: float = 1.17
    anchor_radial_jitter_fraction: float = 0.01
    gauge_radial_jitter_fraction: float = 0.01
    safe_gauge_distance_uas: float = 0.0
    minimum_safe_gauges: int = 3
    weighted_temporal_alignment: bool = True
    crossing_distance_uas: float | None = None
    point_chunk_size: int = 4096
    segment_chunk_size: int = 16384
    triton_segment_block: int = 256
    float64_label_fallback: bool = False

    def __post_init__(self) -> None:
        for name in (
            "jacobian_chunk_size",
            "minimum_sign_component_pixels",
            "anchor_count",
            "gauge_count",
            "minimum_safe_gauges",
            "point_chunk_size",
            "segment_chunk_size",
            "triton_segment_block",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.temporal_batch_size is not None and self.temporal_batch_size < 1:
            raise ValueError("temporal_batch_size must be positive when supplied")
        for name in (
            "anchor_inset_fraction",
            "gauge_inset_fraction",
            "anchor_radial_jitter_fraction",
            "gauge_radial_jitter_fraction",
            "safe_gauge_distance_uas",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.anchor_inset_fraction >= 1.0 or self.gauge_inset_fraction >= 1.0:
            raise ValueError("anchor and gauge inset fractions must be below one")
        if self.crossing_distance_uas is not None and self.crossing_distance_uas < 0:
            raise ValueError("crossing_distance_uas must be non-negative")
        if self.triton_segment_block not in {64, 128, 256, 512, 1024}:
            raise ValueError("triton_segment_block must be a power of two from 64 to 1024")
        cleanup = str(self.determinant_cleanup).lower()
        if cleanup not in {"none", "local"}:
            raise ValueError("determinant_cleanup must be 'none' or 'local'")
        object.__setattr__(self, "determinant_cleanup", cleanup)
        if self.minimum_safe_gauges > self.gauge_count:
            raise ValueError("minimum_safe_gauges cannot exceed gauge_count")
