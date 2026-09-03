# Timing conventions

Every accelerated result distinguishes three costs:

- **Compile time.** One-time graph or kernel compilation for a new shape.
- **Warmup time.** Untimed calls used to populate caches and stabilize memory.
- **Steady-state time.** Delivered runtime after compilation and warmup.

Timing-only accelerator synchronization is disabled by default. Set
`RuntimeConfig(profiling="total")` for synchronized end-to-end time, or
`profiling="detailed"` for component timing and peak-memory diagnostics.
`TimingBreakdown.collected` records whether its numeric fields are valid.
This makes accessing `result.timing` free of hidden profiling work during an
ordinary production calculation.

`system.warmup_light_curve(include_labels=True, temporal_batch_size=30)`
uses the same configuration resolver as `system.light_curve`. No explicit
Torch time array is needed. Warmup is optional because the first ordinary
call also populates compatible compiled kernels.

Independent-curve batching collects no wall time by default. Set
`profile=True` on `batched_system_light_curves` to populate `wall_seconds` and
`seconds_per_curve`. Otherwise both are `None`. Necessary CUDA stream
dependencies remain in place regardless of profiling. Explicit concurrency
tuning measures runtime and is therefore always profiled.

Component timings and peak device memory are stored in `TimingBreakdown`.
Examples and benchmarks must never combine compile time with steady-state time
without labeling the combined quantity explicitly.

For an end-to-end callable, use `benchmark_callable` and pass the resolved
runtime's synchronization method:

```python
measurement = mc.benchmark_callable(
    lambda: simulation.magnification_map(lens_region, source_grid, method=ipm),
    warmup=1,
    repeats=5,
    synchronize=simulation.runtime.synchronize,
)
print(measurement.first_call_seconds)
print(measurement.median_steady_seconds)
```

The first-call value is deliberately named **first-call time**, not compile
time. A portable measurement cannot disentangle kernel/graph compilation from
allocator initialization, cache population, and the first numerical execution.
`estimated_first_call_overhead_seconds` subtracts the median steady time but is
still labeled as an estimate rather than pure compilation cost.

Map metadata distinguishes `requested_backend` from `effective_backend`.
Composite calculations also provide `backend_components`. For example, a
portable IPM request can use compiled Taylor far-field queries together with the exact
Python polygon reference. Only an effective backend of `torch-compile` means
all reported primary tensor components used compiled Torch. A
`torch-compile-partial` label is intentionally explicit.
