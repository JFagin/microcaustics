# Changelog

## Unreleased

- Apple GPU (MPS) support. IPM maps are rasterized on the device by a batched PyTorch port of the Triton direct-cell kernel (its elementwise math fused with `torch.compile`), far-field coefficients are built for all cells in one batch, and far-field queries are chunked to a fixed memory budget, so `RuntimeConfig(device="auto")` runs the whole map pipeline on the Apple GPU without exhausting unified memory. The runtime now reports Metal's recommended working set as the MPS memory budget. CUDA/Triton and CPU paths are unchanged.
- Fixed MPS light curves: PyTorch MPS returns zeros for a combined MPS-to-CPU float64 copy, which emptied every IPM map, and flushes the subnormal brightness of faint disk pixels. MPS now matches CPU within float32 tolerance.

## 1.0.0

Initial release of `microcaustics`:

- Static and dynamic inverse polygon mapping and inverse ray shooting, with fused Triton acceleration and portable PyTorch fallbacks.
- Source-directed scouting, far-field acceleration, automatic memory tuning, and temporal and cross-system batching.
- Critical curves, caustics, source-center labels, distance maps, and winding-map products.
- Relativistic quasar disks, lamppost reprocessing, intrinsic variability, supernovae, custom sources, and optional response-based source evolution.
- Single- and multi-image light curves, microlensing-only comparisons, Rubin cadence sampling, and configurable observation models.
- Optional quasar spectral components and bandpass-integrated photometry, including Rubin/LSST filters.
- Executable tutorials, documentation, and numerical validation against independent implementations.
