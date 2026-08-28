# Coordinates, units, and normalization

## Array and coordinate order

Two-dimensional arrays use `[y, x]` order. Geometry tuples such as
`PlaneGrid.shape`, `field_of_view_uas`, `center_uas`, and `pixel_scale_uas` use the same
`(y, x)` ordering. Plotting functions convert these values into conventional
horizontal `x` and vertical `y` axes.

Lens-plane and source-plane angular coordinates are in microarcseconds unless
a class explicitly states otherwise. Macro-image rendering uses arcseconds.
physical source geometries use meters. Time axes are observer-frame days.

## Lens and source time

Point-lens motion and source trajectories are evaluated in observer time. A
macroimage arrival delay shifts intrinsic source evolution as
`source_time = observer_time - arrival_time_delay_days`. It does not shift the
local lens-motion clock.

## Magnification

`MagnificationMap.values` contains absolute linear magnification, not a
magnitude residual and not a mean-normalized map. IRS and IPM use the same
contract. Map comparisons retain normalization bias unless a caller explicitly
requests a shape-only light-curve comparison.

Magnitude residuals follow
`candidate - reference = -2.5 log10(mu_candidate / mu_reference)`. Positive
residuals are fainter than the reference. `rmse_mmag` is the RMS of that
residual multiplied by 1000. Fractional NRMSE is the RMS of the pixelwise
fractional residual over common finite, positive pixels.

## Provenance

Result metadata records the actual backend, numerical method, reuse settings,
padding, tuning decisions, and relevant geometry. Persist that metadata with
saved arrays so a map can be registered and reproduced.
