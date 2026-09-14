# Critical curves, caustics, and production labels

`simulation.caustics(lens_grid)` is the compact static reference interface. It
evaluates the analytic Jacobian determinant, extracts critical segments with
marching squares, and maps their endpoints to the source plane. Passing a
`FarFieldApproxConfig` selects the local-exact Taylor accelerator. Omitting it uses
the direct point-mass reference.

A returned `CausticField` provides point, center, and full-grid parity,
winding-number, and distance queries. Full label and distance maps are
diagnostics. They are never materialized just to obtain the source-center
label.

## Production anchor/gauge labels

Finite determinant grids can clip critical-curve components, so an infinite
positive-x parity ray is not a reliable production label. Use
`system.light_curve(..., include_labels=True)` for photometry with center
labels. The low-level `MicrolensingSimulation` also provides
`labeled_caustics`, `dynamic_labeled_caustics`, `dynamic_labeled_maps`, and
`light_curve_with_labels`. These methods use the validated production
construction.

- nine reproducibly offset anchors and nine distinct gauge probes just inside
  the requested source boundary.
- finite anchor-to-query paths and a consistent half-open shared-vertex rule.
- redundant anchor-pair constraints followed by a majority vote at the source
  center and each gauge.
- per-query rejection of votes intersecting a critical-curve component clipped
  by the determinant-grid boundary.
- safe gauge consensus to remove the arbitrary global binary XOR between
  adjacent epochs.

Before extraction, the default local 3×3 sign-island cleanup removes only
unresolved one- or few-pixel detA speckles. The setting
`minimum_determinant_sign_pixels=4` controls this threshold. Set it to zero
for a literal determinant contour audit.

The zero/one naming is conventional. A transition in `center_label`, reported
as `center_crossing`, is the physical crossing observable. `center_vote_count`
and `center_valid_count` expose the consensus instead of hiding an uncertain
decision.

Nine anchors and nine gauges are validated paper defaults, not hard-coded
limits. `anchor_count` and `gauge_count` are independent positive integers.
Increasing them adds redundant paths and decreasing them reduces label work.
The configured `minimum_alignment_gauges` must not exceed `gauge_count`.

High-level `MicrolensingSystem` calls do not require a `CausticConfig`. When it
is omitted, labels inherit both the map method's far-field approximation and
the light-curve temporal batch. Construct this configuration only to change
label-specific controls, to override the label batch, or when using the
low-level simulation API.

```python
caustic_config = mc.CausticConfig(
    far_field_approx=ipm.far_field_approx,
    temporal_batch_size=32,  # optional label-specific override
    discovery_downsample_ratio=16,  # 8192-pixel detA -> 512-pixel discovery
)

frames = simulation.dynamic_labeled_caustics(
    lens_grid,
    source_grid.region,
    times_days,
    config=caustic_config,
)
labels = [frame.labels.center_label for frame in frames]
crossings = [frame.labels.center_crossing for frame in frames]
```

The determinant grid controls caustic resolution and is independent of the
magnification-map grid. The paper calculation uses an 8192-pixel long axis.
Smaller grids are useful for examples and tests. In the source-scouted path,
a coarse determinant grid first finds sign changes and unusually small
absolute determinant values. Only those candidates are retained for sparse
evaluation at the requested determinant resolution. The default
`discovery_downsample_ratio=16` therefore uses a 512-pixel discovery grid with
an 8192-pixel determinant grid. Advanced users can lower the ratio for denser
discovery or raise it for a cheaper coarse pass. The final determinant-grid
resolution does not change.

## Sharing work with maps and light curves

For fused tiled IPM, `dynamic_labeled_maps` reuses the already-built temporal
far-field states for detA evaluation and caustic endpoint tracing. It does not
rebuild the far field. The high-level light-curve call consumes those maps
immediately, retaining only the final fluxes and lightweight caustic products.

```python
result = system.light_curve(
    duration_days=3650,
    map_cadence_days=25,
    rays=10_000_000,
    temporal_batch_size=30,
    include_labels=True,
)

flux = result.flux
labels = result.labels.crossing_labels
events = result.labels.crossing_events
label_times = result.labels.times_days
```

`center_distance_uas` is always finite. Because caustics are extracted over a
finite source field, production caps it at the radius of the largest centered
circle contained in that field. When no in-field caustic exists, the returned
distance is therefore `R_src`, while `center_distance_censored` is true:

```python
distance_uas = result.labels.center_distances_uas
distance_censored = result.labels.center_distance_censored
```

At censored epochs, `distance_uas == R_src` means only
`d_caustic >= R_src`. It does not place a caustic on the source boundary.

Map-only and light-curve-only calls still skip every caustic calculation.
CPU, Apple, float64, and non-Triton execution use the same predicates through
the portable chunked implementation. CUDA float32 uses compact two-pass
marching squares plus fused crossing/distance kernels. Temporal and spatial
batches reduce losslessly after a CUDA OOM. No physical resolution parameter
is changed.

## Diagnostic maps and numerical audit

Pass `diagnostic_grid=` to materialize the full anchor/gauge majority map and
`include_distance_map=True` to additionally store nearest-caustic distance.
These options are intended for validation and animations, not center-only
production labels.

Regular-grid parity and winding maps use an exact scanline accumulation. A
segment crossing updates one contiguous row prefix, followed by an integer
cumulative sum. CUDA float32 performs the half-open crossing tests, binary
searches the actual sampled x axis, and accumulates the integer prefix updates
in one Triton kernel. This preserves the grid coordinates and crossing rule
without point-by-segment output tensors. CUDA float32 anchor/gauge maps use the
same finite-path voting rule as the production labels through chunked Triton
queries. CUDA float32 distance maps use the exact point-to-segment Triton
reduction. CPU, Apple, float64, and non-Triton execution retain the portable
chunked Torch calculation. These implementations differ only in execution
strategy.

Winding maps still require a complete, consistently oriented caustic field.
Extracting that complete field can dominate runtime because the determinant
must cover the requested lens plane. Once those segments exist, the winding
map itself uses the scanline path above.

For CUDA float32 order-four far fields, complete-field determinant evaluation
loads the two regular-grid coordinate axes directly inside the Triton query.
It therefore uses one launch without materializing full coordinate meshes.
Compact Triton marching squares then extracts only the zero-crossing segments.
Other dtypes and backends evaluate the same grid in bounded coordinate chunks.

Canonical shared vertices, half-open predicates, and invalid-component vote
masks make the label query stable in the solver dtype. A second float64 label
replay is therefore not part of the production or public configuration.

Signed winding numbers require consistently oriented closed segments. Parity
labels do not require orientation, but both quantities require a complete
field unless the anchor/gauge invalid-component handling is used.

Imported segment collections that have lost their traversal order can be
registered before a signed winding query:

```python
field = mc.CausticField.from_unordered_closed_segments(
    critical_segments,
    mapped_caustic_segments,
    lens_grid,
    closure_tolerance_uas=1.0e-5,
)
winding = field.winding_map(source_grid, orientation="positive")
```

Connectivity is recovered in the lens plane, paired mapped endpoints remain
attached, and each closed component receives one positive global orientation.
This scientific operation is part of the core package rather than notebook
plotting code.

The optional Weisenbach CCF validation constructs winding maps from both its
independently generated caustic curves and the microcaustics complete field.
CCF does not export a native winding product or preserve the same signed
determinant-side orientation convention. The maintained external assertion is
therefore equality of winding modulo two, which is the orientation-independent
binary topology. It does not require equality of the raw signed integer values.
