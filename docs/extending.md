# Extending microcaustics

The package separates lens physics, map solvers, sources, variability,
trajectories, macro models, observation sampling, and output consumers. Prefer
implementing the smallest relevant protocol over subclassing a simulation.

## Custom finite sources

Implement the public pixelated-source interface. Expose a `SourceGeometry` and
return brightness tensors with shape `[time, band, y, x]`. Existing map and
light-curve routines then work without knowing the source physics. This covers
tabulated images, expanding transients, corona models, spectral-line emission,
and differentiable Torch models.

## Variability and transfer functions

Wrap user time series with `TabulatedSignal`, or supply a callable PSD or
variability model. Microlensing map cadence and intrinsic source cadence remain
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
The optional caustics adapter accepts models supported by that package.

## Instruments and outputs

Observation samplers and macro-image renderers accept callbacks so survey
cadences, noise models, PSFs, and detector effects are not hard-coded. A map
observer can stream selected frames into any storage or analysis system.

Keep external research comparators in the optional validation suite rather
than importing them from runtime code.
