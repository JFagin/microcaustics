# Point-mass populations

For ordinary physical systems, describe the population rather than its
already sampled tensors:

```python
kinematics = mc.SkyProjectedKinematics(
    ra_deg=340.126125,
    dec_deg=3.358611,
    stellar_dispersion_km_s=170,
    peculiar_velocity_dispersion_km_s=235,
    include_cmb_dipole=True,
)
population = mc.StellarPopulation.salpeter(
    mean_mass_solar=0.3, mass_ratio=100, kinematics=kinematics,
)
system = mc.MicrolensingSystem(
    macro=macro,
    distances=distances,
    source=source,
    stellar_population=population,
    duration_days=3650,
    light_loss=0.01,
    safety_scale=1.5,
    seed=0,
)
```

The sampled lens and source velocities inherit the system seed. Set `seed`
inside `SkyProjectedKinematics` only when that physical velocity realization
must remain fixed while the stellar realization changes.

With no seed on either object, the first velocity query draws from PyTorch's
existing random stream without reseeding it. The draw is retained by the
kinematic object so that subsequent epochs and geometry calculations use
the same bulk motion. Construct a new kinematic object for a fresh unseeded
bulk-motion realization. Shared macroimage populations deliberately share
this motion, while their stars have independent positions and dispersions.
`system.metadata()` records the realized lens, source, CMB, and combined bulk
velocities, as well as the coordinate basis and stellar dispersion.

The sky position projects the CMB dipole. The sampled peculiar-velocity model
adds independent lens and source terms, while the stellar dispersion supplies
an independent proper motion for every compact object. Supply explicit lens
and source velocity pairs instead when they are measured.

The source peculiar velocity is already included in the effective stellar
drift. Do not add that same motion again through a source trajectory.
An additional trajectory should describe a separate relative displacement.

`IsotropicKinematics` remains available for controlled calculations that use
one already-combined bulk velocity. Dynamic systems warn when this reduced
model, stationary kinematics, zero stellar dispersion, or an omitted CMB,
lens, or source term is used.

This path samples the complete circular stellar field. Its aperture is derived
from the source support, the full macro-lens matrix including shear angle, the
mass-function moments, the light-loss tolerance, the safety scale, and a
conservative motion allowance. `stellar_motion_sigma_margin` controls that
allowance, while `rectangle_light_loss` can independently change the
conventional rectangular launch region. Scout, full-field, and rectangular
integration domains all retain this same population. They differ only in
which lens-plane cells or rays are evaluated.

The lower-level builders below remain useful when a study needs an explicit
region or directly controlled star arrays.

Users may provide physical masses and positions directly with
`PointMassField`, or draw a reproducible uniform population:

```python
catalog = mc.PointMassField(
    x_uas=[-2.0, 0.5, 1.7],
    y_uas=[0.2, -1.1, 1.4],
    mass_solar=[0.3, 1e-3, 30.0],
)
```

The containing `MicrolensingSystem` derives each angular Einstein radius from
its lens and source distances. Use `einstein_units_to_uas` and
`uas_to_einstein_units` when an external catalog or simulation is expressed
in units of the Einstein radius of a selected mean microlens mass.

```python
macro = mc.MacroLens(
    convergence=0.45,
    shear=0.38,
    smooth_matter_fraction=0.2,
)
lens_region = mc.PlaneRegion((800.0, 800.0))
motion = system.realize().stellar_population.kinematics  # bound system seed
stars = mc.PointMassField.sample_uniform(
    lens_region,
    macro,
    distances,
    mc.kroupa_mass_function(0.08, 10.0),
    seed=0,
    velocity_dispersion_uas_per_day=(
        motion.component_dispersion_uas_per_day(distances),
    ) * 2,
    velocity_mean_uas_per_day=(
        motion.mean_velocity_uas_per_day(distances)
    ),
)
print(mc.compact_convergence(stars, lens_region))
```

An explicit `seed` reproduces all sampled masses, positions, and velocities.
Passing a CPU `torch.Generator` instead lets a caller manage and advance a
larger random stream. If neither is supplied, sampling uses PyTorch's evolving
global CPU stream, so successive calls draw different populations. Compound
simulations should record separate seeds for the stellar population, intrinsic
source variability, and observation noise. This avoids unintended correlations
while preserving exact reproducibility.

The inferred object count uses the analytic mean of the mass function. A finite
draw generally has a slightly different realized convergence, which is why it
is returned by the explicit `compact_convergence` calculation rather than being
silently rescaled. Supplying `count=` overrides count inference, and supplying
arrays directly bypasses all built-in assumptions.

For this lower-level builder, the spatial region is exactly the region
requested by the caller. It does not add a light-loss convention or safety
factor. Enlarge the `PlaneRegion` explicitly when a calculation requires a
guard field.
