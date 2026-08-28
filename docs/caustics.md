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
`labeled_caustics`, `dynamic_labeled_caustics`, `dynamic_labeled_maps`, or
`light_curve_with_labels` instead. These methods use the validated production
construction:

- nine reproducibly offset anchors and nine distinct gauge probes just inside
  the requested source boundary.
- finite anchor-to-query paths and a consistent half-open shared-vertex rule.
- redundant anchor-pair constraints followed by a majority vote at the source
  center and each gauge.
- per-query rejection of votes intersecting a critical-curve component clipped
  by the determinant-grid boundary. And
- safe gauge consensus to remove the arbitrary global binary XOR between
  adjacent epochs.

Before extraction, the default local 3×3 sign-island cleanup removes only
unresolved one/few-pixel detA speckles (`minimum_sign_component_pixels=4`).
Set `determinant_cleanup="none"` for a literal determinant contour audit.

The zero/one naming is conventional. A transition in `center_label`, reported
as `center_crossing`, is the physical crossing observable. `center_vote_count`
and `center_valid_count` expose the consensus instead of hiding an uncertain
decision.

Nine anchors and nine gauges are validated paper defaults, not hard-coded
limits. `anchor_count` and `gauge_count` are independent positive integers.
increasing them adds redundant paths and decreasing them reduces label work.
The configured `minimum_safe_gauges` must not exceed `gauge_count`.

```python
caustic_config = mc.CausticConfig(
    far_field_approx=ipm.far_field_approx,
    temporal_batch_size=40,
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
smaller grids are useful for examples and tests.

## Sharing work with maps and light curves

For fused tiled IPM, `dynamic_labeled_maps` reuses the already-built temporal
far-field states for detA evaluation and caustic endpoint tracing. It does not
rebuild the far field. `light_curve_with_labels` consumes those maps
immediately, retaining only the final fluxes and lightweight caustic products:

```python
result = simulation.light_curve_with_labels(
    lens_region,
    source_grid,
    lens_grid,
    times_days,
    source,
    distances,
    method=ipm,
    map_schedule=mc.DynamicConfig(temporal_batch_size=40),
    caustic_config=caustic_config,
)

flux = result.light_curve.flux
labels = result.crossing_labels
events = result.crossing_events
```

`center_distance_uas` is always finite. Because caustics are extracted over a
finite source field, production caps it at the radius of the largest centered
circle contained in that field. When no in-field caustic exists, the returned
distance is therefore `R_src`, while `center_distance_censored` is true:

```python
distance_uas = result.center_distances_uas
distance_censored = result.center_distance_censored
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

`CausticConfig(float64_label_fallback=True)` replays the small label-query set
in float64 after float32 caustic extraction. It is off by default. Canonical
shared vertices, half-open predicates, and invalid-component vote masks are the
production fix, while float64 replay remains an explicit audit option.

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
