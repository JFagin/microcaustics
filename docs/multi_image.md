# Resolved multi-image simulations

`MultiImageSystem` coordinates several independent macroimage
calculations behind one physical source. Define the shared source, redshifts,
stellar-population prescription, and numerical settings once. Each named
`MacroLens` then generates its own seeded star field and dynamic map sequence.

```python
images = {
    "A": mc.MacroLens(convergence=0.39, shear=0.40, shear_angle_deg=120.32),
    "B": mc.MacroLens(convergence=0.38, shear=0.39, shear_angle_deg=143.24),
    "C": mc.MacroLens(convergence=0.74, shear=0.73, shear_angle_deg=51.57),
    "D": mc.MacroLens(convergence=0.64, shear=0.62, shear_angle_deg=80.21),
}

system = mc.MultiImageSystem.from_redshifts(
    lens_redshift=0.0395,
    source_redshift=1.695,
    H0=70.0,
    Om0=0.3,
    images=images,
    source=source,
    stellar_population=mc.StellarPopulation.salpeter(
        mean_mass_solar=0.3,
        mass_ratio=100,
        kinematics=kinematics,
    ),
    arrival_time_delays_days={"A": 0.0, "B": 7.4, "C": 2.1, "D": 11.8},
    integration_domain="scout",
    light_loss=0.01,
    safety_scale=1.5,
    seed=1001,
)

result = system.light_curves(observation_times_days, include_labels=True)
flux = result.flux_tensor()  # [image, time, band]
curve_b = result["B"].light_curve
```

A scalar seed is a reproducible base seed. Stable independent image seeds are
derived from each image name, so reordering the mapping does not change a
realization. Supply `seed={"A": ..., "B": ...}` when exact per-image seeds
are preferred. `methods`, `schedules`, `trajectories`,
`caustic_configs`, `integration_domain`, `stellar_population`, and `runtime`
may each be either one shared value or a mapping keyed by image name.

Each resolved image owns its own:

- local convergence, shear, and smooth-matter fraction.
- point-mass realization and stellar velocities.
- lens and source map fields.
- IPM or IRS configuration and dynamic schedule.
- source-plane bulk trajectory.
- numerical runtime or device.
- cosmological arrival-time delay.

The source object, source-plane distances, and band definitions are shared.
This keeps intrinsic variability coherent while allowing independent
microlensing patterns and bulk velocities.

Bands remain arbitrary because the interface accepts any `PixelatedSource`.
There is no built-in assumption of LSST `ugrizy`.

The lower-level `microcaustics.multi_image.MultiImageSimulation` and
`MacroImageConfig` interfaces remain available when an application already
owns fully constructed simulations, lens grids, or heterogeneous runtimes.
They are intentionally outside the top-level namespace and are not required for ordinary
multi-image calculations.

## Time convention

`arrival_time_delay_days` is an observer-frame arrival delay relative to the
user's chosen reference image. At observer time `t`, macroimage `i` evaluates
the shared source at

```text
source emission time = t - arrival_time_delay_days[i].
```

Only source evolution is shifted. Stellar motion, dynamic magnification maps,
and source-plane trajectories are evaluated at observer time `t`. Negative
relative delays are valid. `TimeShiftedSource` exposes the same convention as
a standalone source adapter.

The map magnification already includes the supplied local macro lens. The
multi-image layer does not apply an additional hidden macro-magnification,
extinction, zeropoint, or flux normalization.

Known or measured delays can be entered directly on each image, as above. A
global macro-lens solver can supply the same values as a mapping without
rebuilding the local microlensing configurations:

```python
system = system.with_arrival_time_delays(
    {"A": 0.0, "B": 7.4, "C": 2.1, "D": 11.8},
    require_all=True,
)
```

Partial mappings override only named images. This keeps the package flexible:
the delays may be inferred by a macro model, taken from observations, fixed in
a simulation study, or omitted entirely. The microlensing layer never assumes
that local convergence and shear alone determine global arrival delays.

Any lens from the `caustics` package can provide the global image positions,
local lens properties, and relative Fermat delays. Build the lens normally,
then pass it to the generic adapter:

```python
solutions = mc.solve_caustics_macroimages(
    my_caustics_lens,
    source_x_arcsec=beta_x,
    source_y_arcsec=beta_y,
    parameters=model_parameters,
    time_delay_parameters=time_delay_parameters,
    image_names=("A", "B", "C", "D"),
)

system = mc.MultiImageSystem.from_macroimage_solutions(
    solutions,
    distances=distances,
    source=source,
    stellar_population=stellar_population,
    duration_days=3650,
    seed=1001,
)
```

This works with individual profiles, composite single-plane lenses, and
multiplane models as long as the object exposes the standard `caustics`
`raytrace`, `jacobian_lens_equation`, and `time_delay` methods. Models from
other packages can implement `MacroModel` directly or use
`CallableMacroModel` with three callables.

For an EPL plus external-shear macro model, the package
also includes a convenience builder. Install the optional dependencies and
solve the relative Fermat delays:

```bash
pip install "microcaustics[macro]"
```

```python
macro = mc.EPLShearConfig(
    lens_redshift=0.0394,
    source_redshift=1.695,
    einstein_radius_arcsec=0.87,
    axis_ratio=0.70,
    position_angle_rad=1.16,
    shear_gamma1=-0.048,
    shear_gamma2=0.051,
)
solutions = mc.solve_epl_shear_macroimages(
    macro,
    source_x_arcsec=0.01,
    source_y_arcsec=-0.015,
    # Names correspond to the arrival-time-sorted solutions.
    image_names=("D", "A", "C", "B"),
)
system = mc.MultiImageSystem.from_macroimage_solutions(
    solutions,
    distances=distances,
    source=source,
    stellar_population=stellar_population,
    duration_days=3650,
    seed=1001,
)
```

Each solution provides position, parity, signed macro magnification, and local
convergence/shear. `solution.local_macro_lens()` converts those quantities to
the microlensing model. Users may ignore global solving entirely and supply
measured delays and local lens parameters directly.

## Multi-rate resolved-quasar workflow

Intrinsic variability and reprocessing often require a much finer time grid
than the moving microlens field. Generate maps sparsely and return daily fluxes
with two map--source contractions per source epoch:

The paper example uses `broken_power_law_driving_signal` with a 200-day PSD
break. This choice belongs to the source model. Any custom `DrivingSignal` can
be substituted without changing the multi-image or microlensing calculation.

```python
curves = system.multirate_light_curves(
    map_times_days=range(0, 3651, 25),
    flux_times_days=range(0, 3651),
    source=reprocessing_source,
    include_labels=True,
)
```

The implementation retains at most two maps and never materializes daily
interpolated maps. Caustic labels are returned only at the sparse map epochs,
available through each image's `label_times_days`. At observer time `t`, image `i` evaluates the source at
`t - arrival_time_delay_days[i]`, while maps and source trajectories stay at
observer time `t`.

Microlensing-weighted transfer functions use the same sparse map cadence:

```python
responses = system.transfer_functions(
    map_times_days=range(0, 3651, 25),
    source=reprocessing_source,
    delay_edges_days=torch.linspace(0, 30, 121),
)
```

`mean_delays_days` contains the internal reprocessing lag. The convenience
property `observer_mean_delays_days` adds the image's macro arrival delay.

Finally, an arbitrary visit sequence can sample the fine truth light curves.
For Rubin data, use an OpSim cadence and the standard random-plus-systematic
magnitude uncertainty:

```python
cadence = mc.sample_random_rubin_wfd_cadence("baseline.db", seed=7)
observations = mc.observe_multi_image_light_curves(
    curves,
    cadence,
    seed=11,
)
```

All photometric source models return flux density in Jy, so observations and
plots use AB magnitudes directly. No fitted reference magnitude is required.

Bands are read from the cadence and source. `ugrizy` is not hard-coded into the
multi-image simulator. General macro-image rendering is a separate optional
layer built directly on `caustics.LensSource`. It supports arbitrary caustics
lenses, source/lens-light stacks, pixelated package sources, PSFs, and detector
callbacks. See [`macro_image_rendering.md`](macro_image_rendering.md).

## Cadence, memory, and maps

A common time sequence gives directly stackable resolved light curves. A
mapping permits different cadences:

```python
result = system.light_curves(
    {"A": times_a, "B": times_b},
)
```

`result.flux_tensor()` requires common bands and times and raises clearly when
per-image cadences cannot be stacked. Individual curves remain available by
name in either case.

Macroimages execute in configured order. This bounds peak memory, preserves
each image's compiler and tuning behavior, and supports different devices.
`system.dynamic_maps(times)` streams `MultiImageMapFrame` objects in
image-major and then time order without retaining map cubes.

## Optional caustic labels

Supply `lens_grid` and optionally `caustic_config` on every image, then request
labels:

```python
result = system.light_curves(
    times,
    source,
    distances,
    include_labels=True,
)
events_a = result["A"].crossing_events
```

The current production anchor/gauge labels describe the fixed center of the
source grid. To prevent a silent mismatch, the high-level interface requires
`trajectory=None` when labels are requested. Relative bulk motion can be
encoded in each image's point-mass velocities, which is the validated
map-centered production convention. Light curves without labels support any
`SourceTrajectory` implementation.

Diagnostic full label and distance maps remain opt-in through
`diagnostic_grid` and `include_distance_map`. Ordinary center labels do not
materialize them.
