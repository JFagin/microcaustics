# Dynamic-map scheduling

Point-lens velocities are expressed in microarcseconds per day. Passing a time
axis to `MicrolensingSimulation.dynamic_maps` streams source-independent maps
in the same order:

```python
schedule = mc.production_dynamic_config()
ipm = mc.production_ipm_config()

maps = simulation.dynamic_maps(
    lens_region,
    source_grid,
    times_days,
    method=ipm,
    schedule=schedule,
)
```

On CUDA float32 with the Triton backend, this is genuine fused temporal IPM:

- far-field Taylor coefficients are accumulated together at anchor frames.
- local-star membership and positions remain exact at every frame.
- one Taylor far-field query launch traces the shared node coordinates for every
  frame in a temporal batch.
- retained cells are interpolated and rasterized directly into a stack of
  maps without materializing global triangle arrays. And
- a short final batch may be padded to the compiled temporal shape, with the
  synthetic maps discarded.

CPU, Apple, float64, eager PyTorch, and compiled PyTorch follow the same public
contract through the exact portable calculation. The iterator does not retain
the complete map cube after values have been consumed by the caller.

## Different map and source cadences

Intrinsic source evolution need not use the dynamic-map cadence. For example,
generate the slowly changing microlensing field every 25 days while evaluating
a variable source every day:

```python
map_times = torch.arange(0.0, 3650.0 + 12.5, 25.0)
source_times = torch.arange(0.0, 3650.0 + 0.5, 1.0)
curve = simulation.multirate_light_curve(
    lens_region,
    source_grid,
    map_times,
    source_times,
    source,
    distances,
    method=ipm,
    schedule=schedule,
)
```

At each fine epoch the implementation evaluates the source and interpolates
its contractions with the two bracketing maps. It retains at most two maps and
never constructs a daily map cube. This is a source-independent interface:
`source` may be a quasar disk, expanding supernova, analytic profile, or any
user-defined `CallableSource`. Use `multirate_light_curve_with_labels` to add
source-center labels at the sparse map epochs.

## Streaming finite-source light curves

`MicrolensingSimulation.light_curve` evaluates a source in temporal batches and
consumes each map immediately. Only the final flux array is retained. To derive
several source or trajectory realizations from the same map sequence, use
`light_curves`:

```python
requests = [
    mc.LightCurveRequest(source_a, distances, trajectory=track_a, name="a"),
    mc.LightCurveRequest(source_b, distances, trajectory=track_b, name="b"),
]
curves = simulation.light_curves(
    lens_region,
    source_grid,
    times_days,
    requests,
    method=ipm,
    schedule=mc.DynamicConfig(
        temporal_batch_size=49,
        light_curve_batch_size=8,
    ),
)
```

Compatible source-array shapes are passed through one batched map-sampling
operation. Different shapes, band counts, physical scales, and coverage rules
remain valid and are grouped automatically. The expensive maps are still made
only once. This is independent-source photometry behind a shared macroimage.
the package does not describe it as fused map generation across unrelated lens
systems.

## Exact and approximate reuse

If the point-mass field has no velocities, all requested frames share the same
lens state. The scheduler calculates one map and reuses its tensor exactly.

For a moving tiled-IPM field, `scout_refresh_frames > 1` is an explicit
approximation. Scouts are evaluated at the beginning and end of each interval.
All fine cells selected by either endpoint are retained throughout the
interval. This coverage safeguard is always active. A refresh value of one
recomputes the scout at every frame and disables temporal scout reuse.
Full-field IPM and IRS do not use scout reuse.

The far-field approximation constructs the local-star packs and complete coefficient
tables independently at every map epoch. Multiple epochs can share a batched
CUDA/Triton coefficient-accumulation call, but no coefficient table is
interpolated in time. This preserves pixel-level map fidelity while retaining
the throughput benefit of temporal batching.

Every result records its anchor frames, selected-cell count, whether endpoint
union was used, and whether the selection was approximate. This provenance is
kept even when map values are consumed immediately by a light-curve routine.

The scheduler halves `cell_chunk_size` after a CUDA out-of-memory allocation
until the configured lossless minimum is reached. This changes throughput and
memory use only. It never changes `N`, `k`, `r`, `v`, or map normalization. If
the minimum spatial chunk still does not fit, the temporal batch is split and
retried. This second backoff is also lossless.

`temporal_batch_size` defines the fused map shape and source-evaluation batch.
When it is omitted, CUDA uses the validated paper batch of 49 while portable
devices stream one frame at a time. Enable `AutoTuningConfig` explicitly to
benchmark alternative temporal and spatial work sizes. See
[`tuning.md`](tuning.md). A user-supplied value is honored when tuning is
disabled, unless lossless OOM backoff must split it.
`light_curve_batch_size` limits the number of compatible source requests in a
single map-sampling call. Compile, warmup, and steady-state timing remain
separate. The current package only reports a fused operation when the
underlying solver actually executes one.

This shared-map source batching is distinct from independent-system
concurrency. Use `batched_system_light_curves` when every curve has its own
stellar realization, macro lens, source, and map sequence. Its explicit
`curves_per_batch` value controls how many complete systems run concurrently
on one CUDA device. Lossless OOM backoff halves that concurrency and retries.
CPU and Apple MPS retain the same interface and execute systems sequentially.

## Paper production and conservative reference modes

The validated production settings have named constructors so examples do not
silently drift apart:

```python
ipm = mc.production_ipm_config()
schedule = mc.production_dynamic_config()
```

This resolves to $N=10^7$, $k=2$, $r=2$, $v=4$, a 16-by-16 local-membership
partition with 8-by-8 Taylor nodes per cell, temporal batch 49, and refresh-10
endpoint-union scouting. It uses no source-pixel halo
(`scout_halo_pixels=0`) and retains one lens-plane scout-cell support ring
(`scout_dilation_cells=1`). The dual-scout option performs the frame-zero `k=1`
to `k=2` scalar normalization correction and reuses that constant for later
frames. For one unrelated static map, use
the default static-map operation, which selects `k=1` and does not run
the dynamic correction. For a
more conservative validation run, also set `scout_refresh_frames=1` and
optionally disable the scalar correction. The returned metadata identifies
every activated approximation, anchor frame, padding count, OOM reduction,
and actual raster backend.
