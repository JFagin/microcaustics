# Configuring inverse polygon mapping

The inverse polygon mapping (IPM) interface separates four controls that are
often conflated. They can be changed independently, subject only to
`virtual_refinement >= refinement`.

```python
method = mc.IPMConfig(
    rays=10_000_000,       # N: fine lens-plane base cells
    scout_ratio=2,         # k: scout coarsening along each axis
    refinement=2,          # r: true subcells per selected base-cell axis
    virtual_refinement=4,  # v: polygon subcells per base-cell axis
    tiled=True,
)
```

## What each parameter changes

`rays` (`N`) is the requested number of fine lens-plane base cells. The actual
rectangular grid is chosen to match the lens-field aspect ratio and may differ
slightly from `N`. In tiled mode its dimensions are rounded upward to multiples
of `k` so every scout tile contains exactly `k × k` fine cells. Increasing `N`
reduces finite-cell mapping error and normally increases tracing and
rasterization cost.

`scout_ratio` (`k`) applies only when `tiled=True`. A `k=2` scout has half as
many cells along each axis as the fine grid. Every selected scout tile is then
expanded back into its exact `k × k` fine cells before IPM. Thus `k` changes
the cost and spatial resolution of source-region discovery. It does not change
the retained cells' final lens-plane size. Smaller `k` is more conservative.
The scout safety controls act in different coordinate spaces:

- `scout_halo_pixels` expands the requested source aperture before testing
  mapped scout cells, in units of source pixels. The validated production
  default is `0.0`.
- `scout_dilation_cells` expands the selected mask in the lens plane after the
  scout test. The production default is one cell, retaining a support ring
  around the selected region.
- `scout_trace_centers` supplements mapped scout-corner tests with cell-center
  queries. It is enabled by default.

These controls are independent of `scout_ratio` (`k`) and remain configurable.
Increasing either halo is more conservative but retains more cells. Setting
`k=1` is the complete tiled scout and is the recommended baseline for one
independent static map.

For a dynamic sequence's optional normalization repair, set
`dual_scout_scalar_correction=True` with tiled `k=2`. A one-time dense `k=1`
corner trace supplies both scouts. Only cells in their signed symmetric
difference are mapped into one source-plane bin, yielding a reference-free
constant correction. The correction is added to every magnification-map pixel
and therefore propagates exactly through finite-source map-to-flux
convolution. It repairs the small global `k=2` selection offset. It is not a
spatial residual-map correction. For one independent static map, use `k=1`
directly. There is no temporal sequence over which to amortize the repair.

`refinement` (`r`) sets the number of true mapped subcells along each selected
base-cell axis. The lens equation is evaluated on an `(r+1) × (r+1)` lattice.
Increasing `r` therefore spends additional ray traces to model curvature.

`virtual_refinement` (`v`) sets the number of raster polygons per base-cell
axis. The traced `r` lattice is interpolated to `(v+1) × (v+1)` nodes before
rasterization, without further lens-equation evaluations. `r=1` is affine,
`r=2` is the paper's biquadratic mapping, and higher `r` uses the corresponding
tensor-product Lagrange interpolant. Increasing `v` can make curved mapped
boundaries smoother, but cannot recover information absent from the true
`r` samples.

## Suggested starting points

- **Independent static map.** The high-level static operation selects
  `N=10_000_000, k=1, r=2, v=4` by default.
- **Dynamic production sequence.** The high-level dynamic operation selects
  `N=10_000_000, k=2, r=2, v=4` and computes the one-time `k=1` to `k=2`
  normalization repair.
- **Fast exploratory map.** Reduce `N`, keep `r=2, v=4`, and validate the chosen
  scout ratio against `k=1` for the intended lens population.
- **Conservative tiled validation.** Use `k=1` and/or a larger scout dilation.
- **Full-field validation.** Set `tiled=False`. `k` is then ignored.
- **Mapping convergence.** Increase `N` and `r` independently. Raising only `v`
  tests raster representation rather than additional lens-equation accuracy.

CUDA float32 uses a direct-cell Triton scanline kernel for `v <= 16`. Other
devices, float64, and larger `v` fall back to exact Sutherland–Hodgman overlap
clipping. Backend choice changes implementation and performance, not the IPM
configuration or output normalization.

Always test convergence for a new source scale, lens population, or extreme
macro-magnification. The defaults are strong general-purpose settings, not a
substitute for a problem-specific accuracy audit.

## Batching unrelated static maps

Independent star fields cannot share a physical map, but compatible fields can
share one GPU launch schedule. Use `StaticMapRequest` and
`batched_magnification_maps`. Each request retains its own point masses,
far-field coefficients, scout mask, normalization, timing, and metadata. The
helper requires static `k=1` tiled IPM and losslessly reduces the batch after an
out-of-memory allocation. It never substitutes the dynamic scalar repair.
