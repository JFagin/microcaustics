# Configuring inverse polygon mapping

Inverse polygon mapping (IPM) maps lens-plane cells into the source plane and
distributes their area among overlapping source pixels. Four controls set its
sampling and reconstruction. They can be changed independently, subject to
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
The search includes cells intersecting the circular stellar aperture and one
neighboring layer outside it. The retained neighbors described below are a
separate guard around cells selected by the scout.

The scout safety controls act in different coordinate spaces:

- `scout_halo_pixels` expands the requested source aperture before testing
  mapped scout cells, in units of source pixels. The validated production
  default is `0.0`.
- `scout_dilation_cells` retains neighboring scout cells in the lens plane
  after selection. The production default is one layer. At `k=2`, that layer
  is two fine cells wide.
- `scout_trace_centers` supplements mapped scout-corner tests with cell-center
  queries. It is disabled by default in `IPMConfig` and both static and dynamic
  presets. Set it to `True` for an additional interior sample when checking
  scout coverage. This check can recover cells missed by their mapped corners,
  but it does not guarantee complete coverage.

These controls are independent of `scout_ratio` (`k`) and remain configurable.
Increasing either halo is more conservative but retains more cells. Setting
`k=1` scouts at the fine-grid spacing and is the recommended baseline for one
independent static map.

For a dynamic sequence, `dual_scout_scalar_correction=True` enables the initial
normalization correction with tiled `k=2`. At the first epoch, the denser
`k=1` scout estimates a uniform magnification offset between the two
selections. This estimate requires no external reference map. The correction
is added uniformly to subsequent magnification maps before integrating the
source brightness. It accounts for an approximately uniform missing
contribution but does not recover unresolved spatial structure. For one
independent static map, use `k=1` directly.

`refinement` (`r`) sets the number of true mapped subcells along each selected
base-cell axis. The lens equation is evaluated on an `(r+1) × (r+1)` lattice.
Increasing `r` therefore spends additional ray traces to model curvature.

`virtual_refinement` (`v`) sets the number of raster polygons per base-cell
axis. The traced `r` lattice is interpolated to `(v+1) × (v+1)` nodes before
rasterization, without further lens-equation evaluations. `r=1` interpolates
the four corner nodes,
`r=2` is the paper's biquadratic mapping, and higher `r` uses the corresponding
tensor-product Lagrange interpolant. Increasing `v` can make curved mapped
boundaries smoother, but cannot recover information absent from the true
`r` samples.

## Suggested starting points

- **Independent static map.** The high-level static operation selects
  `N=10_000_000, k=1, r=2, v=4` with corner-only scouting by default.
- **Dynamic production sequence.** The high-level dynamic operation selects
  `N=10_000_000, k=2, r=2, v=4`, uses mapped cell corners for selection,
  retains one layer of neighboring scout cells, and computes the initial
  normalization correction.
- **Center-assisted validation.** Start from `production_ipm_config` and pass
  `scout_trace_centers=True`.
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
out-of-memory allocation. It does not use the dynamic normalization correction.
