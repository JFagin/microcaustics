# Extending microcaustics

The package separates lens physics, map solvers, sources, variability,
trajectories, macro models, observation sampling, and output consumers. Prefer
implementing the smallest relevant protocol over subclassing a simulation.

## Custom finite sources

For a brightness function, use `CallableSource` with `source_grid_shape`,
`field_of_view_uas`, and `bands_angstrom`. For an image array, `StaticSource`
infers the shape. Both obtain physical geometry from the system distances.

For a custom class, implement the public pixelated-source interface. Expose a `SourceGeometry` and
return brightness tensors with shape `[time, y, x, band]` in `Jy m^-2` of
projected source-plane area. Existing map and
light-curve routines then work without knowing the source physics. This covers
tabulated images, expanding transients, corona models, spectral-line emission,
and differentiable Torch models.

## Variability and transfer functions

Wrap user time series with `TabulatedDrivingSignal`, or supply a callable PSD or
variability model. Attach the signal to a source that defines its response,
not to the microlensing system. For simple multiplication, wrap a physical
model or pixelated source with `ModulatedSource`. For a custom physical response, expose its
`driving_signal` and implement `with_driving_signal(signal)` to return a copy
with a bound or constant driver. Sources without a driver remain valid and
raise an error only when explicitly asked to apply one.
Microlensing map cadence and intrinsic source cadence remain
independent. Sources implementing the transfer-function protocol can use the
standalone steady and microlensed response interfaces.

## Stellar populations

Implement `MassFunction` to provide sampling and a mean mass, or construct
`PointMassField` directly. Binary objects, compact dark matter, planets, and
mixed populations require no solver changes.

## Macro models and resolved systems

Known image properties may be supplied through the advanced
`microcaustics.multi_image.MacroImageConfig` interface.
External macro solvers can implement `MacroModel` or use `CallableMacroModel`.

## Instruments and outputs

Observation samplers and macro-image renderers accept callbacks so survey
cadences, noise models, PSFs, and detector effects are not hard-coded. A map
observer can stream selected frames into any storage or analysis system.

Keep external research comparators in the optional validation suite rather
than importing them from runtime code.
