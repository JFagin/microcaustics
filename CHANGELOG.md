# Changelog

## 1.0.1

updated user guides, notebooks, added astrometry, and added cubic transfer function

## 1.0.0

Initial release of `microcaustics`:

- Apple GPU (MPS) IPM rasterization and batched far-field calculations, with chunked queries and Metal memory-budget reporting. Compilation follows the requested runtime settings.
- MPS compatibility fixes for float64 host copies and faint-pixel flux accumulation, with CPU comparison tests.
- Fewer Triton IPM recompilations when map bounds, pixel scales, or triangle weights change.
- Static and dynamic inverse polygon mapping and inverse ray shooting, with fused Triton acceleration and portable PyTorch fallbacks.
- Source-directed scouting, far-field acceleration, automatic memory tuning, and temporal and cross-system batching.
- Critical curves, caustics, source-center labels, distance maps, and winding-map products.
- Relativistic quasar disks, lamppost reprocessing, intrinsic variability, supernovae, custom sources, and optional response-based source evolution.
- Single- and multi-image light curves, microlensing-only comparisons, Rubin cadence sampling, and configurable observation models.
- Optional quasar spectral components and bandpass-integrated photometry, including Rubin/LSST filters.
- Executable tutorials, documentation, and numerical validation against independent implementations.
