# Coordinates, units, and normalization

## Array and coordinate order

Two-dimensional arrays use `[y, x]` order. Geometry tuples such as
`PlaneGrid.shape`, `field_of_view_uas`, `center_uas`, and `pixel_scale_uas` use the same
`(y, x)` ordering. Plotting functions convert these values into conventional
horizontal `x` and vertical `y` axes.

Lens-plane and source-plane angular coordinates are in microarcseconds unless
a class explicitly states otherwise. Macro-image rendering uses arcseconds.
Physical source geometries use meters. Time axes are observer-frame days.

`MacroLens.shear_angle_deg` is the physical shear position angle in the input
sky frame. Tile-scouted and full-field calculations retain that frame. The
high-level rectangular strategy may use the shear eigenframe internally to
avoid replacing the conventional narrow rectangle by a much larger
axis-aligned bounding box. In that case the package transforms point-lens
positions and velocities, the source position angle, and the trajectory
together. It never rotates or interpolates an already rendered disk image.
The realization metadata records `coordinate_frame` and
`sky_to_local_rotation_deg`.

Pixelated source brightness used for photometry is in `Jy m^-2` of projected
source-plane area. `LightCurve.flux` and `LightCurve.unlensed_flux` are in Jy.
AB magnitudes therefore use the standard 3631 Jy zero point directly. No
reference magnitude is fitted to a simulated light curve.

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
