# Thin disks and relativistic source calculations

`ThinDiskSource` is a physical, static continuum disk implementing the same
signed-spin ISCO, Page--Thorne radial dissipation, color correction, and
observed-frequency convention as the validated paper implementation. It is an
ordinary `PixelatedSource`, so it can be used with IRS, IPM, external maps, or
without microlensing.

```python
disk = mc.ThinDiskSource.from_lensing_distances(
    geometry,
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    distances=distances,
    source_redshift=1.7,
    spin=0.7,
    inclination_deg=30.0,
    position_angle_deg=15.0,
    color_correction=1.0,
    relativity="approximate",
)
```

Pixels use package-standard array order `(y, x)` and represent the projected
source plane. Brightness is in `Jy m^-2` of projected source plane, so the
ordinary finite-source integrator supplies the pixel area exactly once and
returns Jy. This scaling also avoids float32 underflow that would result from
storing cgs flux per physical square meter. Wavelengths in `SourceGeometry` are
observer-frame angstroms and may describe any filters.

The two currently published modes are explicit:

- `relativity="none"` uses no photon frequency shift while retaining the
  Page--Thorne radial flux profile.
- `relativity="approximate"` adds the straight-screen circular-orbit
  gravitational/Doppler shift used by the legacy approximation.

Neither option is mislabeled as full GR. For ordinary full-GR calculations,
`KerrDiskModel` owns the mutually consistent observer screen, primary-image
Kerr trace, redshift and delay maps, axial-lamppost response, and pixel
geometry:

```python
driver = mc.broken_power_law_driving_signal(
    torch.arange(-200.0, 3651.0),
    break_timescale_days=200.0,
    seed=12,
    extrapolation="hold",
)
disk = mc.KerrDiskModel(
    black_hole_mass_solar=10**9.08,
    eddington_ratio=0.34,
    bands={"u": 3671.0, "g": 4827.0, "r": 6223.0},
    spin=0.74,
    inclination_deg=10.0,
    position_angle_deg=175.0,
    source_redshift=1.695,
    lamp_fraction=0.1,
    corona_height_above_isco_rg=20.0,
    driving_signal=driver,
    resolution=1024,
    source_margin=1.05,
)
source = disk.pixelate(distances, runtime=runtime)
```

Omit `driving_signal` for a static relativistic disk. Use
`disk.with_driving_signal(signal)` to retain every other choice. Resolution,
enclosed-flux support, outer margin, lamppost sampling, and compile behavior
remain explicit numerical controls.

The full-GR frequency-shift map is already part of the returned source. No
microlensing calculation is required:

```python
g = source.transfer.gfactor       # g = nu_observed / nu_emitted
hit = source.transfer.hit         # primary rays that reach the disk
g_map = torch.where(hit, g, torch.nan)
```

The source-model tutorial plots this map in angular coordinates. The SIM5
validation tutorial independently regenerates the same package transfer before
reading the optional external SIM5 result.

The other registered GR and reverberation maps are available from the same
source without retracing rays:

```python
radius_rg = source.transfer.radius_rg
azimuth_rad = source.transfer.emission_azimuth_rad
observer_delay_days = source.transfer.relative_delay_days
continuum_lag_days = source.delay_days
heating_response_temperature4 = source.response_temperature4
```

`relative_delay_days` is the disk-to-observer propagation term and has its own
relative zero. `delay_days` adds the lamp-to-disk travel time and is reset to
zero at the earliest physically responsive disk element. The latter is the
causal delay used by the continuum transfer functions. The source-model
tutorial renders the radius, azimuth, observer-delay, and total-continuum-lag
maps in their common source-plane coordinates.

The lower-level observer-transfer components remain public for inspecting the
trace, importing an external transfer, or developing a different illumination
model. The next section documents that advanced interface.

## Observer-transfer contract

`ObserverTransfer` separates photon propagation from emission physics. It
stores package-standard `[y, x]` maps of emission radius, frequency shift,
observer solid angle, hit status, and optional relative delay and emission
azimuth. `TransferredThinDiskSource` converts any compatible transfer into a
physical Page--Thorne source:

```python
screen = mc.ObserverScreen.uniform(
    (1024, 1024),
    half_size_rg=80.0,
    gravitational_radius_m=gravitational_radius_m,
    observer_distance_m=observer_distance_m,
)
screen = screen.rotated(position_angle_deg=15.0)

trace = mc.trace_primary_equatorial(
    screen,
    spin=0.7,
    inclination_deg=30.0,
    disk_outer_rg=80.0,
)

coordinates = mc.add_observer_coordinates(
    trace,
    screen,
    black_hole_mass_solar=1.0e9,
    spin=0.7,
    inclination_deg=30.0,
    source_redshift=1.7,
    coordinate_dtype=torch.float64,
)

source = mc.TransferredThinDiskSource(
    geometry,
    coordinates.transfer,
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    spin=0.7,
    source_redshift=1.7,
)
```

Consequently the same disk-emission implementation can consume the analytic
Kerr backend, a SIM5 validation product, or a user ray tracer. Delay and
azimuth remain available for reverberation and non-axisymmetric extensions,
but axisymmetric static brightness does not pay to recompute them.

Choose the observer-screen extent from the emission model rather than an
unrelated plotting constant. `thin_disk_flux_radius_rg` evaluates the
Page--Thorne profile, optional axial-lamp heating, the reddest requested
rest-frame wavelength, and a configurable enclosed-flux fraction:

```python
outer_rg = mc.thin_disk_flux_radius_rg(
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    spin=0.7,
    observed_wavelength_angstrom=9700.0,
    source_redshift=1.7,
    lamp_fraction=0.1,
    corona_height_above_isco_rg=10.0,
    flux_fraction=0.999,
    safety_factor=1.05,
)
screen = mc.ObserverScreen.uniform(shape, outer_rg, ...)
```

Use the same radius for the Kerr disk boundary, source pixel scale, and
physical source field. A moving source may require a larger magnification-map
field to cover its trajectory, but that does not change the disk model or its
pixel scale. This distinction prevents both truncated transfer-function tails
and accidental zero padding during finite-source photometry.

When a transfer includes `relative_delay_days`, it may be used immediately by
`DelayedModulatedSource.from_observer_transfer(...)`. For a physical additive
heating map, `ThermalReprocessingSource` evaluates the nonlinear temperature
response and its analytic linear transfer function. The built-in axial Kerr
lamppost tracer supplies conservative proper-area illumination and the
lamp-to-disk part of the delay:

```python
lamp = mc.axis_lamppost_profile(
    spin=0.7,
    height_above_isco_rg=10.0,
    disk_outer_rg=80.0,
)
variable_disk = mc.ThermalReprocessingSource.from_axis_lamppost(
    geometry,
    coordinates.transfer,
    signal,
    lamp,
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    spin=0.7,
    source_redshift=1.7,
    lamp_fraction=0.1,
)
```

`ObserverScreen.rotated` applies the position-angle convention without
reimplementing coordinate algebra. `height_above_isco_rg` is converted with
the package Kerr ISCO calculation. Supplying `lamp_fraction` evaluates the
Novikov--Thorne radiative efficiency and the corresponding heating
normalization internally. The lower-level `source_height_rg` and
`irradiation_efficiency` arguments remain available for externally specified
models.

External illumination models can supply the same two arrays directly. This is
also the extension point for a future finite or non-axisymmetric corona.

`axis_lamppost_profile` supports eager Torch on CPU, CUDA, and Apple devices.
Set `compile_solver=True` to cache one static-shape `torch.compile` callable per
device, dtype, launch resolution, quadrature order, and compile mode. The ray
result records `compile_warmup_s` separately from steady execution time, and a
failed compiler toolchain falls back to the numerically identical eager path
unless `fallback_to_eager=False` is requested.

`ObserverScreen` uses pixel-center sampling and calculates each pixel's true
solid angle from its impact-parameter extent, physical gravitational radius,
and source angular-diameter distance. `TransferredThinDiskSource` then applies
the invariant-intensity factor `(1 + z)^-3`. This prevents photon tracers from
silently mixing luminosity and angular-diameter distances or using different
flux normalizations.

The primary analytic tracer uses separated Kerr equations, fixed-count Carlson
integrals, manifestly real quartic roots, and an isolated-pinhole repair that
recomputes the circular-orbit frequency shift from the repaired radius. Its
extracted tensors match the paper implementation exactly in float64. On the
three stored 1024-square SIM5 validation screens, the migrated tracer retains
identical hit masks. The paper comparison had unit hit-mask intersection over
union and integrated flux ratios within `4e-7` of unity.

Observer azimuth and finite-observer travel time use analytic polar integrals
and logarithmically sampled radial Gauss--Legendre integration. Calculations
are chunked over hit rays only. Float32 uses a compact float64 repair queue for
rare two-real-root and poorly conditioned polar rays. Callers may instead
promote the entire coordinate stage to float64 independently of the primary
image. Returned delays are converted from `GM/c^3` to
observer-frame days using the supplied black-hole mass. The implementation
matches the validated paper coordinates to floating-point rounding on the
frozen regression fixture.

Low-level differentiable functions are available from
`microcaustics.relativity` provides `kerr_isco_radius`, `page_thorne_flux_factor`,
`circular_disk_gfactor`, `circular_disk_zamo_lorentz_factor`, and the explicitly
named approximate frequency shift.
