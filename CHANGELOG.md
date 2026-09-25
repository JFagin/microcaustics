# Changelog

## Unreleased

- Add shared-map microlensing-only comparison fluxes to single-system,
  multi-image, labeled, multirate, and independently batched light curves.
- Add a reusable in-memory Rubin OpSim cadence index with fast random WFD and
  DDF sampling, named DDF selection, and coordinate-based visit queries.
- Add a single-light-curve observation API while preserving one shared cadence
  across every macroimage in resolved lensed-system observations.
- Expose configurable Rubin band-noise parameters and document the cadence and
  observation workflow.
- Add built-in Novikov–Thorne and Shakura–Sunyaev viscous profiles, custom
  profile and radiative-efficiency callables, and a dedicated source notebook.
- Add compilation warnings, enabled by default and configurable through the
  runtime, so first-call Torch and Triton specialization costs are explicit.
- Add `batched_system_light_curves` for mixed single-, double-, and quad-image
  datasets, with bounded-memory direct NPZ or combined-archive output.
- Add independent cross-disk Kerr source setup batching, with one source per
  lensed system, fixed-shape observer-delay launches, and lossless OOM backoff.
- Fuse compatible independent systems through shared scout preparation,
  ragged IPM and caustic queues, indexed far-field evaluation, and cached
  compact-node plans.
- Batch transfer-function and mean-response-delay generation across times,
  bands, and macroimages.
- Cache immutable reverberating-disk tensors and compile the relativistic
  multiband brightness kernel for faster, lower-memory temporal source batches.
- Reorganize and re-execute the complete tutorial collection, including new
  spectral-microlensing and disk-profile demonstrations.

## 1.0.0 - 2026-08-27

First stable release of `microcaustics`.

- Static and dynamic inverse polygon mapping with Cartesian or reproducible
  random inverse ray shooting.
- Fused Triton acceleration with portable eager and compiled Torch fallbacks.
- Taylor far-field acceleration, source-tile scouting, and automatic memory tuning.
- Caustic, critical-curve, source-center label, label-map, distance-map, and winding-map products.
- Compact temporal marching and shared endpoint queries for dense full-field and rectangular caustic sequences, as well as scouted sequences.
- Quasar, relativistic thin-disk, reverberation, supernova, and custom-source interfaces.
- Multi-image simulations, intrinsic variability, observational cadence, and macro-image rendering.
- Eighteen executable scientific tutorials and validation against independent analytic and external implementations.
