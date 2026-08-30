# Physical system interface

This guide explains how to construct complete microlensing calculations from
physical inputs. Most users can work with `MicrolensingSystem` while lower-level
source, lens, solver, batching, caustic, and multi-image interfaces remain
available for specialized calculations.

## What the interface handles

The ordinary workflow should not require a user to calculate source-plane
support, lens-plane aperture, stellar count, Einstein radii, or angular
velocities by hand. It should still expose every physical assumption and every
derived value.

The high-level interface must support all current workflows.

- Static and dynamic source-independent magnification maps
- Finite-source light curves with optional intrinsic variability
- Fast source-center caustic labels
- Full binary, distance, winding, determinant, critical-curve, and caustic maps
- Independent static-map and light-curve batching
- Multi-image systems with independent stellar fields and shared sources
- Built-in, callable, and pixelated sources
- Sampled stellar populations and directly supplied stars
- Scout, full-field, and rectangular integration regions
- Torch eager, Torch compiled, Triton, CPU, CUDA, and MPS execution

## Physical specifications and realized objects

The interface separates physical specifications from realized tensors.

- `StellarPopulation` describes a mass function, spatial distribution, and
  kinematics.
- `PointMassField` remains the realized positions, Einstein radii, masses, and
  angular velocities used by the solvers.
- `ThinDiskModel`, `GaussianModel`, and other source specifications
  contain physical parameters without requiring a grid.
- `ExpandingPhotosphereSource` derives its fixed grid from the maximum
  photosphere radius and is accepted directly by the same system interface.
- Existing pixelated source classes remain the realized source objects.
- `MicrolensingRealization` contains the resolved source grid, stellar field,
  integration geometry, simulation, and provenance.

Users may bypass every physical builder and continue supplying existing
low-level objects directly.

## Stellar populations

The Salpeter convenience constructor is

```python
population = mc.StellarPopulation.salpeter(
    mean_mass_solar=0.3,
    mass_ratio=100,
    kinematics=mc.IsotropicKinematics(
        dispersion_km_s=180,
        bulk_velocity_km_s=(0, 0),
    ),
)
```

The constructor derives the Salpeter mass limits from the requested mean mass
and mass ratio. The general population specification accepts any existing
`MassFunction`, a kinematics protocol, and an optional exact count. The system
seed controls reproducibility.

The default realistic spatial distribution is uniform within a full circular
stellar aperture. Rectangle and direct-position models remain available for
controlled experiments.

## Kinematics

Physical kinematics are resolved into the observer-frame angular velocities
already stored by `PointMassField`.

`IsotropicKinematics` accepts a one-dimensional proper velocity dispersion
and an optional Cartesian bulk velocity in km/s. Both are converted with the
lens angular-diameter distance and observer-frame time dilation. The derived
stellar aperture includes both the requested bulk drift and a configurable
multiple of the random dispersion over the simulated duration.

The built-in models currently include the following.

- `StaticKinematics`
- `IsotropicKinematics`
- `SkyProjectedKinematics`

The sky-projected models combine sky position, CMB motion, lens and source
peculiar velocities, and stellar dispersion. Their sampled form may defer the
redshift-dependent conversion until the enclosing system resolves its
cosmological distances. Custom models can implement the same public protocol.

## Automatic source support

Built-in physical source models report a recommended projected support. The
thin-disk default uses the radius enclosing 99.9 percent of the reddest-band
flux and applies a 1.05 numerical margin. The support calculation includes the
requested inclination, orientation, bands, redshift, and relativistic mode.

The source grid policy controls resolution, enclosed-flux fraction, margin,
and reference band. Custom sources may report their own support. A callable or
pixelated source without finite support must provide an explicit source grid.

A physical source is optional. A user may request a source-independent map by
supplying only its source-plane size and resolution.

Physical models also report their true circular major-axis support separately
from the rectangular pixel grid. This avoids treating known zero-valued grid
corners as emitting source area when deriving the stellar aperture. An
arbitrary pixelated or callable source may provide `source_support_radius_uas`
when its finite support is known. Otherwise the complete source-grid rectangle
is used conservatively. A source-independent map always uses that full
rectangle.

## Stellar aperture and integration region

The physical stellar aperture and numerical integration region are distinct.

For a sampled realistic population, the package always constructs the full
circular stellar field from the source support, macro eigenvalues, mass
moments, light-loss tolerance, safety scale, duration, and motion. This
physical stellar realization remains unchanged when integration methods are
compared. Its coordinate components may be expressed in the shear eigenframe
for the rectangular strategy.

The integration region chooses where rays or IPM cells originate.

- `IntegrationDomain.SCOUT` starts from the complete bounding domain and keeps
  only conservatively selected cells.
- `IntegrationDomain.FULL` evaluates every cell in the complete bounding
  domain.
- `IntegrationDomain.RECTANGLE` evaluates the conventional smaller launch
  rectangle. Built-in physical sources are evaluated in a consistently
  shear-aligned numerical frame, including their position angle and
  trajectory, so no source raster is rotated or resampled.

All stars in the full circular population contribute to all three methods.
Selecting a rectangle never truncates the stellar field.

The stellar-aperture `light_loss` tolerance and rectangular-integration
`rectangle_light_loss` tolerance are recorded separately. The latter defaults
to the former, but they are not the same quantity. The optional
`stellar_motion_sigma_margin` controls how many random-velocity standard
deviations are included in the aperture over the requested duration.

## Directly supplied stars

`MicrolensingSystem` accepts either `stellar_population` or `stars`. These
arguments are mutually exclusive. When `stars` is supplied, the package uses
exactly those objects and never adds, removes, or rescales them.

An automatic conservative bounding calculation derives a finite integration
domain from the requested source-plane map, macro model, and supplied stars.
An explicit lens region remains an escape hatch for regression fixtures,
external-code reproduction, moving direct-star fields, and deliberately
truncated experiments.

## Microlensing system and realization

The recommended observational interface is

```python
system = mc.MicrolensingSystem.from_redshifts(
    lens_redshift=0.0395,
    source_redshift=1.695,
    H0=70.0,
    Om0=0.3,
    macro=macro,
    source=source_model,
    stellar_population=population,
    duration_days=3650,
    seed=1001,
    runtime=runtime,
)
```

Common realized values are available directly on the system.

```python
system.realized_stars
system.resolved_source_grid
system.resolved_lens_region
system.stellar_aperture
system.metadata()
```

Use `system.with_seed(new_seed)` to create an independent stellar realization
without repeating the physical setup. Compatible compiled kernels are reused.

Advanced integrations can use `system.realization` to access the underlying
low-level simulation and every derived object together.

Convenience methods delegate to the existing numerical implementation.

```python
system.magnification_map(...)
system.dynamic_maps(...)
system.light_curve(...)
system.light_curve_with_labels(...)
system.dynamic_labeled_maps(...)
system.transfer_functions(...)
```

The high-level layer performs setup once and adds no per-frame numerical work.
Use `keep_maps_at_days=(...)` on a light-curve call to return a small selected
set of maps directly. Streaming observers remain available when every frame
must be exported. `system.warmup(...)` performs and returns a representative
call so compatible compiled kernels can be reused explicitly.

## Numerical settings

`production_ipm_config`, `production_dynamic_config`, and `CausticConfig`
group the validated numerical choices without adding astrophysical assumptions.
Each returns the same existing, independently replaceable configuration
objects used by the low-level interface.

The default dynamic calculation uses 10 million rays, scout ratio 2,
refinement 2, virtual refinement 4, the validated Taylor far field, a
40-frame temporal batch, a ten-frame endpoint-union scout refresh, and the
frame-zero scout-ratio normalization correction. Source-center labels use nine
anchors and nine gauges on an 8192-square determinant grid.

Every nested setting remains independently replaceable. Existing low-level
configuration constructors remain public.

## Batching and multi-image systems

Independent batching accepts resolved realizations. Compatible requests are
grouped by shape and backend while retaining independent stars, far-field
states, sources, and metadata.

Multi-image systems contain one resolved realization per macro image. Each image
has an independent stellar population, integration region, trajectory, and
microlensing calculation. The physical source and intrinsic variability may
be shared. Known or inferred cosmological delays remain supported.

Fast center labels and complete diagnostic label maps remain separate opt-in
products.

## Lower-level access

`MicrolensingSimulation`, `PointMassField`, `PlaneRegion`, `PlaneGrid`, source
classes, solver configurations, batching requests, and multi-image classes are
public. Population and far-field tutorials use some of these objects because
they teach the numerical components directly. Validation notebooks also keep
explicit control over grids and integration domains.
