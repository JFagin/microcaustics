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
import microcaustics as mc

population = mc.StellarPopulation.salpeter(
    mean_mass_solar=0.3,
    mass_ratio=100,
    kinematics=mc.SkyProjectedKinematics(
        ra_deg=340.126125,
        dec_deg=3.358611,
        stellar_dispersion_km_s=170,
        peculiar_velocity_dispersion_km_s=235,
        include_cmb_dipole=True,
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

`SkyProjectedKinematics` is the physical default demonstrated in the guides.
It includes the projected CMB dipole, lens and source peculiar velocities,
and a one-dimensional stellar proper-velocity dispersion. The derived stellar
aperture includes the resulting bulk drift and a configurable multiple of the
random dispersion over the simulated duration.

`IsotropicKinematics` accepts a one-dimensional dispersion and one already
combined Cartesian bulk velocity. It is useful for controlled studies, but a
dynamic system warns because it cannot verify the separate CMB, lens, and
source contributions.

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

A physical source is optional. A centered square source-independent map uses
`map_width_uas` and `map_pixels` on the map method. `PlaneGrid` remains the
advanced interface for rectangular or off-center fields.

The same width and pixel arguments are accepted by `summary()` and
`metadata()`. After exactly one source-independent geometry has been used,
both inspection methods select it automatically. When several geometries have
been used, pass the two arguments explicitly to remove any ambiguity.

Direct point-lens catalogs use positions in microarcseconds and masses in
solar masses. Their angular Einstein radii are derived from the system
distances. The numerical lens plane is automatic by default. It encloses the
macro preimage, all supplied lenses, and an Einstein-radius guard. Set
`lens_plane_uas` to one size for a centered square or two sizes for a centered
rectangle. An explicit `PlaneRegion` remains available for off-center
validation geometry.

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

The internal stellar velocities are evolved with specular reflection at the
circular aperture so the finite population does not diffuse away or change
the local macro parameters. Coherent bulk motion is applied after this
confined internal evolution. The fixed-source lens mapping includes the
corresponding uniform smooth-term offset, making coherent translation of the
explicit stars equivalent to translation of the complete local lens pattern.

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
macro = mc.MacroLens(convergence=0.391, shear=0.391)
source_model = mc.GaussianModel(
    sigma_uas=(0.05, 0.08),
    bands_angstrom={"g": 4770.0, "i": 7625.0},
    total_flux=(20e-6, 15e-6),
    source_grid_shape=256,
)
runtime = mc.RuntimeConfig(device="auto", backend="auto")
system = mc.MicrolensingSystem(
    lens_redshift=0.0395,
    source_redshift=1.695,
    H0=70.0,
    Om0=0.3,
    macro=macro,
    source=source_model,
    stellar_population=population,
    duration_days=3650,
    seed=0,
    runtime=runtime,
    caustic_grid_shape=256,
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

Use `system.with_seed(new_seed)` to create an independent stellar realization.
Its physical setup is recalculated when needed, while compatible compiled
kernels are reused.

Advanced integrations can use `system.realization` to access the underlying
low-level simulation and every derived object together.

Convenience methods delegate to the existing numerical implementation.

```python
static_map = system.magnification_map(rays=262_144)
dynamic_maps = tuple(system.dynamic_maps((0.0, 25.0), rays=262_144))
curve = system.light_curve(
    duration_days=25.0,
    map_cadence_days=25.0,
    source_cadence_days=1.0,
    rays=262_144,
)
labeled_curve = system.light_curve(
    duration_days=25.0,
    map_cadence_days=25.0,
    include_labels=True,
    rays=262_144,
)
labeled_maps = tuple(
    system.dynamic_labeled_maps((0.0, 25.0), rays=262_144)
)
```

Sources with response-delay maps additionally support steady and microlensed
transfer functions as described in the relativistic-source guide.

The high-level layer performs setup once and adds no per-frame numerical work.
Use `keep_maps_at_days=(...)` on a light-curve call to return a small selected
set of maps directly. Streaming observers remain available when every frame
must be exported. `system.warmup_light_curve(...)` performs and returns a representative
call so compatible compiled kernels can be reused explicitly.

The light-curve call also accepts `temporal_batch_size`, `scout_refresh_frames`,
`rays`, and an independent `source_cadence_days`. It returns apparent AB
`magnitude`, original Jy `flux`, and optional `labels` with their own time axis.
Retained maps use integer indices. `result.maps[0]` corresponds to
`result.map_times_days[0]`. Unavailable retention requests warn and are omitted.

## Numerical settings

`production_ipm_config`, `production_dynamic_config`, and `CausticConfig`
group the validated numerical choices without adding astrophysical assumptions.
Each returns the same existing, independently replaceable configuration
objects used by the low-level interface.

The default dynamic calculation uses 10 million rays, scout ratio 2,
refinement 2, virtual refinement 4, the validated Taylor far field, a
49-frame temporal batch, a ten-frame endpoint-union scout refresh, and the
frame-zero scout-ratio normalization correction. Source-center labels use nine
anchors and nine gauges on an 8192-square determinant grid.

On CUDA float32 runs, temporal caustic segments remain in packed ragged form
through fused Triton crossing and distance reductions. The implementation falls
back first to the padded Triton reduction and then to the portable Torch path if
the preferred kernel is unavailable.

Every nested setting remains independently replaceable. Existing low-level
configuration constructors remain public.

## Batching and multi-image systems

Independent batching accepts resolved realizations. Compatible production
Triton requests use ownership-tagged ragged GPU queues across systems while
retaining independent stars, scout cells, far-field states, sources, and
metadata. Mixed multi-image inputs are partitioned by compatible macroimage
contract and reconstructed in their original order.

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
