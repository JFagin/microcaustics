# Dynamic-map scheduling

For a physical `MicrolensingSystem`, the usual light-curve call is

```python
curve = system.light_curve(
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    rays=10_000_000,
    temporal_batch_size=30,
    scout_refresh_frames=10,
    include_labels=True,
    include_microlensing_only=True,
    keep_maps_at_days=(0.0,),
)

magnitudes = curve.magnitude       # [photometry epoch, band], apparent AB mag
micro_only = curve.microlensing_only_magnitude  # same maps, mean driver
labels = curve.labels             # source-center labels at map epochs
first_map = curve.maps[0]          # first retained map
```

An optional driver belongs to the source. Omit `apply_driving_signal` to use
that driver when present, or set it to `False` to retain its baseline without
fluctuations. Setting it to `True` requires a source with a configured driver.
Supernova expansion and other independent source evolution are unaffected by
this switch.

For a variable source, `include_microlensing_only=True` adds the corresponding
constant-mean-driver flux and magnitude to the same result. Map generation,
caustic labels, and source geometry are shared. Because the comparison source
is static, its brightness is calculated once and reused at every epoch.

The lower-level examples below use `MicrolensingSimulation`, named
`simulation`, when direct control of the grids and scheduler is useful.

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

- far-field Taylor coefficients are accumulated in batches for every map epoch.
- local-star membership and positions remain exact at every frame.
- one Taylor far-field query launch traces the shared node coordinates for every
  frame in a temporal batch.
- retained cells are interpolated and rasterized directly into a stack of
  maps without materializing global triangle arrays.
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

`MicrolensingSystem.light_curve` evaluates a source in temporal batches and
consumes each map immediately. Only the final flux array is retained. To derive
several source or trajectory realizations from the same map sequence, use
`light_curves`:

```python
requests = [
    mc.LightCurveRequest(source_a, trajectory=track_a, name="a"),
    mc.LightCurveRequest(source_b, trajectory=track_b, name="b"),
]
curves = system.light_curves(
    requests=requests,
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    rays=10_000_000,
    temporal_batch_size=49,
    light_curve_batch_size=8,
)
```

Compatible source-array shapes are passed through one batched map-sampling
operation. Different shapes, band counts, physical scales, and coverage rules
remain valid and are grouped automatically. The expensive maps are still made
only once. This is independent-source photometry behind a shared macroimage.
The package does not describe it as fused map generation across unrelated lens
systems.

Shared-map requests inherit the system distances unless explicitly overridden.
Source models are pixelated once before sampling. Fine-cadence evolution uses
the same two-map interpolation as individual light curves. Each dynamic map
is generated once for the whole request list.

Requests may also select wavelengths and cadences without constructing new
sources manually. This makes an evolving spectrum an ordinary multiband light
curve. The system source should include the reddest requested wavelength when
its spatial support is selected automatically:

```python
spectral_bands = {
    f"lambda_{w:05d}": float(w) for w in range(3000, 11001, 20)
}
daily, spectra, mean_spectra = system.light_curves(
    duration_days=3650,
    map_cadence_days=25,
    requests=(
        mc.LightCurveRequest(
            bands_angstrom=lsst_bands,
            flux_cadence_days=1,
        ),
        mc.LightCurveRequest(bands_angstrom=spectral_bands),
        mc.LightCurveRequest(
            bands_angstrom=spectral_bands,
            apply_driving_signal=False,
        ),
    ),
    band_batch_size=32,
)
```

`band_batch_size` controls only memory and execution shape. The final partial
wavelength group is padded to the same size so it does not trigger another
compiled kernel, and those filler channels are removed from the result.

Use `batched_system_light_curves` for unrelated stellar realizations. It and
`tune_system_light_curve_batch` accept the same duration, cadence, rays, and
temporal-batch keywords. Static batches use
`mc.batched_system_maps(systems, rays=10_000_000, batch_size=2)`.

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

Every map result records its scout refresh endpoints, selected-cell count,
whether endpoint union was used, and whether the selection was approximate. This provenance is
kept even when map values are consumed immediately by a light-curve routine.

The scheduler halves `cell_chunk_size` after a CUDA out-of-memory allocation
until the configured lossless minimum is reached. This changes throughput and
memory use only. It never changes `N`, `k`, `r`, `v`, or map normalization. If
the minimum spatial chunk still does not fit, the temporal batch is split and
retried. This second backoff is also lossless.

`temporal_batch_size` defines the fused map shape and source-evaluation batch.
High-level light curves default to 49 without labels and 30 with labels.
The low-level scheduler can choose its device default when its batch is
unspecified. Enable `AutoTuningConfig` explicitly to
benchmark alternative temporal and spatial work sizes. See
[`tuning.md`](tuning.md). A user-supplied value is honored when tuning is
disabled, unless lossless OOM backoff must split it.
`light_curve_batch_size` limits the number of compatible source requests in a
single map-sampling call. Compile, warmup, and steady-state timing remain
separate. The current package only reports a fused operation when the
underlying solver actually executes one.

This shared-map source batching is distinct from independent-system
concurrency. `batched_system_light_curves` accepts arbitrary mixtures of
single- and multi-image systems. It realizes inputs lazily, batches individual
macroimage curves with `curves_per_batch`, and reconstructs each system in its
original image order. Compatible tiled-IPM Triton curves fuse their far-field,
ragged scout-cell rasterization, and optional Jacobian/marching-squares queues
without mixing stellar fields or tracing a union of different systems' cells.
For mixed doubles or quads, matching macroimage contracts are grouped across
systems while incompatible images use the established private-stream path.
Lossless OOM backoff halves CUDA concurrency and retries; CPU and Apple MPS use
the same interface and execute curves sequentially.
The same `include_microlensing_only=True` option applies to every single or
multi-image result and is preserved by both disk-backed output modes.

For collections of relativistic variable sources, set
`source_setup_batch_size` independently of `curves_per_batch`. Compatible Kerr
disks then pool their directly traced rays through the same fixed-size compiled
observer-delay kernel before the stellar realizations are built. A double or
quad contributes one shared source, not one source per macroimage. Repeated
batches reuse the compiled specialization even when the disk parameters or
batch occupancy change; only source resolution, dtype, compile mode, or the
explicit observer-coordinate chunk size can require another specialization.

Set `output_path` to a directory for flat per-image NPZ files and a manifest,
or to a `.npz` file for one combined NumPy archive. A bounded single-writer
queue overlaps serialization with calculation. The returned
`IndependentLightCurveBatch` stores compact records instead of every curve;
`batch.load_system(index)` reconstructs a single `LightCurve` or complete
`MultiImageLightCurves`. Use `compression=False` when write speed matters more
than space. Flat-directory output supports `resume=True`; a complete system is
reused only when every expected image file exists. Combined archives are
atomically finalized and do not support partial resume. Retained maps remain
separate products and should be handled with map observers in disk-backed runs.

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
and actual raster backend. The high-level labeled light-curve call uses batch
30 unless overridden. Both the map batch and the label batch can be set
explicitly with `temporal_batch_size` and `label_batch_size`.
