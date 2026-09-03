# Optional automatic work-size tuning

Automatic tuning is opt-in. It changes only lossless scheduling parameters:
the number of frames presented to a fused temporal calculation and the size
of spatial work chunks. It never changes the physical lens, map field,
resolution, ray budget, or IPM values `k`, `r`, and `v`.

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
