# Choosing a numerical method

Start from the scientific product, then choose numerical settings. Batch and
chunk tuning comes last because it changes throughput, not the calculation.

## Map-only calculations

- Use `IRSConfig` for a familiar uniform inverse-ray-shooting reference or
  when Poisson sampling is part of the intended calculation.
- Use full-field `IPMConfig(tiled=False)` for polygon mapping over every
  lens-plane cell.
- Use tiled `IPMConfig(tiled=True)` when the requested source field occupies a
  small fraction of the mapped lens plane. Validate representative frames
  against full-field IPM before adopting aggressive scout settings.
- For one independent static tiled map, start with `k=1`. For a dynamic
  sequence, `k=2` plus the one-time dual-scout scalar repair can amortize the
  conservative `k=1` normalization across all epochs.

IRS and IPM return absolute, source-independent magnification maps on the same
`PlaneGrid`. A source is required only for finite-source photometry.

## IPM controls

`N`, `k`, `r`, and `v` are independent:

- `rays=N` sets the base fine-cell budget.
- `scout_ratio=k` coarsens tiled source-region discovery only.
- `refinement=r` adds true lens-equation samples within retained cells. And
- `virtual_refinement=v` interpolates the mapped polygon representation and
  must satisfy `v >= r`.

Increase `N` until the map or light-curve metric relevant to the study is
stable. Increase `r` when finite-cell mapping artifacts remain. Test larger
`v` when a smoother polygon representation improves those artifacts without
additional lens-equation samples. No one configuration is optimal for every
source size, macroimage, or output resolution.

## Local-exact Taylor far-field approximation

Disable `FarFieldApproxConfig` for a direct point-mass reference. With it enabled,
nearby stars are exact and the far field uses a complex Taylor expansion.
Validate its grid, exact radius, and Taylor order against the direct
calculation. Every epoch receives its own complete far-field coefficient table.
temporal batching changes throughput but not the numerical approximation.

For moving tiled-IPM fields, `scout_refresh_frames=1` recomputes the scout at
every frame. Larger values reuse an endpoint-union cell selection and are an
explicit approximation. Returned metadata records the selected scout anchors
and whether endpoint-union reuse was active.

## A defensible validation sequence

1. Fix the physical lens, source field, output grid, and metric.
2. Generate a higher-quality full-field reference on representative epochs.
3. Converge `N`, then audit `r` and `v`.
4. Compare tiled and full-field maps without removing a constant offset.
5. Validate the far-field and temporal settings.
6. Check finite-source light curves and caustic labels when they are products.
7. Only then tune temporal batches and spatial chunks.

See `examples/notebooks/16_accuracy_and_performance.ipynb` for an executable
CUDA-aware version of this workflow.
