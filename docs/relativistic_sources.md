# Thin disks and relativistic source calculations

`ThinDiskModel` specifies a physical, static continuum disk implementing the same
signed-spin ISCO, Novikov--Thorne radial dissipation, color correction, and
observed-frequency convention as the validated paper implementation. It is an
object that chooses its own pixel grid. Its resolved source can be used with
IRS, IPM, external maps, or without microlensing.

```python
model = mc.ThinDiskModel(
    black_hole_mass_solar=1e9, eddington_ratio=0.1,
    bands_angstrom={"g": 4800, "i": 7500}, source_grid_shape=256,
    spin=0.7, inclination_deg=30, position_angle_deg=15,
    relativity="approximate",
)
disk = model.pixelate(source_redshift=1.7, H0=70, Om0=0.3)
```

Pixels use package-standard array order `(y, x)` and represent the projected
source plane. Brightness is in `Jy m^-2` of projected source plane, so the
ordinary finite-source integrator supplies the pixel area exactly once and
returns Jy. This scaling also avoids float32 underflow that would result from
storing cgs flux per physical square meter. Wavelengths in `SourceGeometry` are
observer-frame angstroms and may describe any filters.

The two currently published modes are explicit:

- `relativity="none"` uses no photon frequency shift while retaining the
  selected radial flux profile.
- `relativity="approximate"` adds the straight-screen circular-orbit
  gravitational/Doppler shift used by the straight-screen approximation.

Neither option is mislabeled as full GR. For ordinary full-GR calculations,
`KerrDiskModel` owns the mutually consistent observer screen, primary-image
Kerr trace, redshift and delay maps, axial-lamppost response, and pixel
geometry:

```python
driver = mc.broken_power_law_driving_signal(
    cadence_days=0.1,
    break_timescale_days=200.0,
    alpha_L=1.0,
    alpha_R=3.0,
    standard_deviation=0.1,
    seed=0,
)
disk = mc.KerrDiskModel(
    black_hole_mass_solar=10**9.08,
    eddington_ratio=0.34,
    bands_angstrom={"u": 3671.0, "g": 4827.0, "r": 6223.0},
    spin=0.74,
    inclination_deg=10.0,
    position_angle_deg=0.0,
    lamp_fraction=0.1,
    corona_height_above_isco_rg=20.0,
    driving_signal=driver,
    source_grid_shape=1024,
    source_margin=1.05,
)
source = disk.pixelate(source_redshift=1.695, H0=70.0, Om0=0.3)
```

## Viscous flux profiles and radiative efficiency

`ThinDiskModel` and `KerrDiskModel` use the relativistic Novikov--Thorne
profile by default. The Shakura--Sunyaev profile is also built in:

```python
disk = mc.KerrDiskModel(
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    bands_angstrom={"g": 4800.0, "i": 7500.0},
    viscous_flux_profile="shakura-sunyaev",
)
```

When `radiative_efficiency=None`, each built-in profile uses its corresponding
analytic efficiency. It may instead be set explicitly to
`"novikov-thorne"`, `"shakura-sunyaev"`, a scalar in `(0, 1]`, or a callable
of `(spin, isco_rg)`.

Custom profiles are callables of `(radius_rg, spin, isco_rg)`. Additional
parameters belong to the callable itself, so the package interface does not
need a new keyword for every physical model:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class WindSuppressedFlux:
    wind_index: float
    transition_rg: float
    name: str = "wind-suppressed"

    def __call__(self, radius_rg, spin, isco_rg):
        base = mc.shakura_sunyaev_flux_factor(radius_rg, spin, isco_rg)
        return base * (1 + radius_rg / self.transition_rg) ** (-self.wind_index)

    def metadata(self):
        return {
            "wind_index": self.wind_index,
            "transition_rg": self.transition_rg,
        }

disk = mc.ThinDiskModel(
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    bands_angstrom={"g": 4800.0, "i": 7500.0},
    viscous_flux_profile=WindSuppressedFlux(0.35, 50.0),
    radiative_efficiency=0.1,
)
```

The optional `name` and `metadata()` members are recorded in the resolved
source metadata for reproducibility. They are not required for evaluation.
The existing `temperature_slope_beta` remains a separate radial-temperature
tilt whose normalization preserves the selected profile's bolometric viscous
power.

Omit `driving_signal` for a static relativistic disk. Use
`disk.with_driving_signal(signal)` to retain every other choice. Resolution,
enclosed-flux support, outer margin, lamppost sampling, and compile behavior
remain explicit numerical controls.

This standalone path needs no lens redshift or microlensing system. When
lensing is needed, pass the disk model directly to the system instead.
Advanced callers may still supply `distances`, `grid`, and `runtime` to
`pixelate`. Source-only cosmological inputs and explicit distances cannot
be combined.

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
field_of_view_uas = source.transfer.metadata["source_field_of_view_uas"]
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
physical thin-disk source:

```python
import torch
import microcaustics as mc

black_hole_mass_solar = 1.0e9
source_redshift = 1.7
distances = mc.LensingDistances.from_redshifts(0.5, source_redshift)
gravitational_radius_m = float(
    mc.gravitational_radius_m(black_hole_mass_solar)
)
screen = mc.ObserverScreen.uniform(
    (1024, 1024),
    half_size_rg=80.0,
    gravitational_radius_m=gravitational_radius_m,
    observer_distance_m=float(distances.source_m),
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
    black_hole_mass_solar=black_hole_mass_solar,
    spin=0.7,
    inclination_deg=30.0,
    source_redshift=source_redshift,
    coordinate_dtype=torch.float64,
)

pixel_scale_m = 160.0 * gravitational_radius_m / 1024
geometry = mc.SourceGeometry(
    shape=(1024, 1024),
    pixel_scale_m=(pixel_scale_m, pixel_scale_m),
    wavelengths_angstrom=(4770.0, 6231.0, 7625.0),
    band_names=("g", "r", "i"),
)
source = mc.TransferredThinDiskSource(
    geometry,
    coordinates.transfer,
    black_hole_mass_solar=black_hole_mass_solar,
    eddington_ratio=0.1,
    spin=0.7,
    source_redshift=source_redshift,
)
```

Consequently the same disk-emission implementation can consume the analytic
Kerr backend, a SIM5 validation product, or a user ray tracer. Delay and
azimuth remain available for reverberation and non-axisymmetric extensions,
but axisymmetric static brightness does not pay to recompute them.

Choose the observer-screen extent from the emission model rather than an
unrelated plotting constant. `thin_disk_flux_radius_rg` evaluates the selected
viscous profile, optional axial-lamp heating, the reddest requested
rest-frame wavelength, and a configurable enclosed-flux fraction:

```python
shape = (1024, 1024)
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
screen = mc.ObserverScreen.uniform(
    shape,
    half_size_rg=outer_rg,
    gravitational_radius_m=float(mc.gravitational_radius_m(1.0e9)),
    observer_distance_m=float(
        mc.LensingDistances.from_redshifts(0.5, 1.7).source_m
    ),
)
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
selected profile's radiative efficiency and corresponding heating
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

Compilation warnings are enabled by default. Set `warn_on_compile=False` on
`KerrDiskModel`, `trace_primary_equatorial`, or `axis_lamppost_profile` to
suppress them; a runtime with `RuntimeConfig(warn_on_compile=False)` also
suppresses compilation warnings during high-level Kerr source construction.

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
`microcaustics.relativity`, including `kerr_isco_radius`,
`novikov_thorne_flux_factor`, `shakura_sunyaev_flux_factor`, their analytic
radiative efficiencies, `circular_disk_gfactor`,
`circular_disk_zamo_lorentz_factor`, and the explicitly named approximate
frequency shift.
