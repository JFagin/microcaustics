# Optional automatic work-size tuning

Automatic tuning is opt-in. It adjusts scheduling parameters:
the number of frames presented to a fused temporal calculation and the size
of spatial work chunks. It never changes the physical lens, map field,
resolution, ray budget, or IPM values `k`, `r`, and `v`.
With temporal scout reuse, changing the temporal batch can change the extra
cells retained in its conservative union. The tuner therefore checks numerical
agreement rather than assuming bitwise identity across temporal batch sizes.

For publication benchmarks, set `RuntimeConfig(strict_backend=True)`. Normal
interactive runs retain portable fallbacks and emit a one-time warning if an
accelerated kernel fails. Strict mode instead stops immediately so a fallback
cannot be mistaken for an accelerated timing.

```python
tuning = mc.AutoTuningConfig(
    enabled=True,
    temporal_candidates=(8, 16, 24, 32, 40, 49),
    spatial_candidates=(131_072, 262_144, 524_288, 1_048_576),
    memory_fraction=0.95,
)

schedule = mc.DynamicConfig(tuning=tuning)
maps = simulation.dynamic_maps(
    lens_region,
    source_grid,
    times_days,
    method=ipm,
    schedule=schedule,
)
```

The tuner first compares spatial chunks at a fixed temporal batch, then
compares temporal batches using the winning spatial chunk. Every candidate is
warmed before it is timed. The median steady-state seconds per frame selects
the winner. Warmup and tuning time are excluded from returned production
timings and recorded separately in result metadata.

Candidates are rejected when they:

- raise a CUDA out-of-memory error.
- exceed the configured memory ceiling. Or
- fail a deterministic numerical signature check against the first accepted
  candidate.

`RuntimeConfig.memory_fraction` is the default device-memory ceiling.
`AutoTuningConfig.memory_fraction` can impose a more conservative ceiling for
one tuning request, while `memory_headroom_fraction` reserves space beneath
that limit. Ordinary production execution retains its independent lossless
OOM backoff even after tuning.

## Solver-specific spatial chunks

The generic `spatial_candidates` values tune the relevant lossless work unit:

- IPM maps use `IPMConfig.cell_chunk_size`.
- IRS maps use `IRSConfig.ray_chunk_size`.
- Caustics and labels use `CausticConfig.jacobian_chunk_size`.

IPM can tune a genuinely fused temporal batch. The current IRS implementation
streams frames independently, so IRS tuning preserves the requested temporal
batch and tunes only its ray chunk. Caustic tuning jointly validates detA,
mapped segments, and anchor/gauge labels.

```python
caustic_config = mc.CausticConfig(
    anchor_count=13,
    gauge_count=17,
    tuning=mc.AutoTuningConfig(enabled=True),
)
labels = simulation.dynamic_labeled_caustics(
    lens_grid,
    source_region,
    times_days,
    config=caustic_config,
)
```

Anchor and gauge counts are unrelated to the tuner and remain fully
user-selectable.

## Reuse and provenance

Successful decisions are cached in memory for the current Python process,
device, lens field, geometry, numerical settings, time prefix, and tuning
configuration. Repeating the same request does not rerun the sweep. Call
`mc.clear_tuning_cache()` to force a new decision, for example after changing
GPU load conditions.

Every tuned map or caustic field records:

- the selected temporal and spatial sizes.
- total and accepted candidate counts.
- the memory budget.
- tuning time excluded from production timing.
- whether the decision came from the process cache.

For explicit experiments, `mc.autotune_dynamic_maps`,
`mc.autotune_dynamic_ipm`, `mc.autotune_dynamic_irs`, and
`mc.autotune_caustics` return a `TuningResult` containing every `TuningTrial`.
Leaving `AutoTuningConfig.enabled=False` uses the supplied work sizes exactly
and incurs no tuning overhead.

## Independent-system concurrency

`AutoTuningConfig` optimizes work inside one light curve. Independent systems
have a separate opt-in tuner because their best concurrency depends strongly
on stellar count, source geometry, labels, and available GPU memory.

```python
tuned = mc.tune_system_light_curve_batch(
    representative_systems,
    map_times_days,
    flux_times_days,
    candidates=(1, 2, 3, 4),
    include_labels=True,
)
batch = mc.batched_system_light_curves(
    systems,
    map_times_days,
    flux_times_days,
    curves_per_batch=tuned.curves_per_batch,
    include_labels=True,
)
```

The tuner verifies fluxes and labels against sequential execution and rejects
CUDA OOM candidates. Use systems representative of the intended dataset. The
selected value is not treated as universal and is never applied implicitly.

At a fixed temporal batch size, fused independent systems use the same
per-system temporal-batch scout union as single-curve map generation. No scout
cells are shared between independent stellar fields. Caustic labels retain
their per-refresh-interval support in both paths. Changing `curves_per_batch`
should therefore preserve maps, fluxes, and labels up to floating-point
accumulation differences. Keep the temporal batch and scout refresh fixed
when checking this equivalence.

## Thermal spectra and intrinsic variability

Use `wavelength_batch_size` with bandpass photometry (or `band_batch_size` with
explicit wavelength channels) to bound the spectral working set. This does not
change source cadence or replace the nonlinear thermal response with a scalar
modulation. The same prepared heating state is reused across wavelength chunks,
and compiled backends use pixel-contiguous brightness/flux reductions.
On NVIDIA CUDA with float32, a Triton kernel evaluates the wavelength chunk
together for each pixel tile. Differentiable calculations, float64, AMD GPUs,
and other devices use the compiled PyTorch implementation. Scalar wavelength,
redshift, and color inputs are reused across source batches on the same device.
The internal spatial tiles do not change the disk resolution or the
user-selected temporal/wavelength batch sizes. Exactly aligned,
stationary sources do not resample the magnification map.

Advanced users can set `RuntimeConfig(thermal_flux_block_pixels=512)` to
select 128, 256, 512, or 1024 pixels per Triton thermal-flux block. The
default `None` keeps the measured choice: 512 pixels for a 16-wavelength
chunk and 256 for narrower chunks. On a sparse observer mask, the exact
active-tile layout follows the chosen block size; otherwise the kernel uses
the complete linear pixel grid. Neither mode drops hit pixels. This setting
does not affect the map solver or the
PyTorch fallback. Larger blocks can increase register pressure and become
slower or use more memory. Warm each candidate before timing it on the target
GPU; changing the block size may compile another kernel specialization.

For dynamic spectra, source evaluation uses no more than the largest number
of requested flux epochs within one map interval. This avoids padding a
sparse source request to the map/caustic-label temporal batch size; the map
and label batches themselves are unchanged. On one 64-wavelength Q2237 B
CUDA workload, 16-wavelength chunks were modestly faster than chunks of 8.
Treat that as a starting point to measure on the intended GPU and lens cases,
not a universal optimum. A new source-batch shape may need one compilation;
repeated requests with the same shape reuse it.

For large response-approximation datasets with a repeated disk, the default
`RuntimeConfig(static_response_projection=True)` compiles the map-response
projection for its exact tensor shapes. The gain on a Q2237 B 1024-pixel
source was about 0.03 s per 147-map, 10-year light curve. A new valid-pixel
count, delay-bin count, or wavelength batch can trigger another compilation,
so set `static_response_projection=False` for heterogeneous source
populations. A fixed observing cadence alone does not fix these shapes.
This option affects only
`linear_response` and `quadratic_response`, not the exact disk path.

Static disks use a per-call brightness cache when it fits within one thirty-second
of the configured device memory budget, capped at 512 MiB across all requests.
Above that limit the thermal contraction remains wavelength-streamed. Templates
are cached separately on the requested device. Initial kernel compilation is
excluded from steady-state timings; warm each requested shape before comparing
throughput. Reducing source cadence to the microlensing cadence removes daily
variability information and is not required to use these optimizations.
