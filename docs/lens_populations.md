# Point-mass populations

For ordinary physical systems, describe the population rather than its
already sampled tensors:

```python
population = mc.StellarPopulation.salpeter(
    mean_mass_solar=0.3,
    mass_ratio=100,
    kinematics=mc.IsotropicKinematics(
        dispersion_km_s=180,
        bulk_velocity_km_s=(0, 0),
    ),
)
system = mc.MicrolensingSystem(
    macro=macro,
    distances=distances,
    source=source,
    stellar_population=population,
    duration_days=3650,
    light_loss=0.01,
    safety_scale=1.5,
    seed=1001,
)
```

For the complete observer-frame velocity construction, include the sky
position and lens/source peculiar motions:

```python
kinematics = mc.SkyProjectedKinematics(
    ra_deg=340.126125,
    dec_deg=3.358611,
    peculiar_velocity_dispersion_km_s=235,
    stellar_dispersion_km_s=170,
    seed=2001,
)
population = mc.StellarPopulation.salpeter(
    mean_mass_solar=0.3,
    mass_ratio=100,
    kinematics=kinematics,
)
```

This projects the CMB dipole onto the local east/north axes and combines it
with lens and source peculiar velocities. Supplying the two peculiar-velocity
pairs directly avoids sampling them. The stellar dispersion still generates
an independent proper motion for every compact object.

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
stars = mc.PointMassField.sample_uniform(
    lens_region,
    macro,
    distances,
    mc.kroupa_mass_function(0.08, 10.0),
    seed=1001,
    velocity_dispersion_uas_per_day=(2.0e-4, 2.0e-4),
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
