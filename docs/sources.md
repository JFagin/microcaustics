# Source models and driving signals

## Optional source driving

Attach a driver to the source that defines its response. A driver is optional
and does not require manually constructing a time array.

```python
driver = mc.broken_power_law_driving_signal(
    cadence_days=0.1,
    break_timescale_days=200,
    alpha_L=1,
    alpha_R=3,
    standard_deviation=0.1,
)

source = mc.KerrDiskModel(
    black_hole_mass_solar=1e8,
    eddington_ratio=0.1,
    bands_angstrom={"g": 4800, "r": 6200},
    driving_signal=driver,
)
```

Pass `source=source` when constructing `MicrolensingSystem` or
`MultiImageSystem`. Neither system accepts a separate `driving_signal`.
The source's driver inherits the system's independent variability seed
unless explicitly overridden. Multi-image systems share one intrinsic driver
and apply the image arrival delays to source time only.

Call `system.light_curve(..., apply_driving_signal=False)` for constant mean
driver heating, or use `apply_driving_signal=True` for intrinsic variability.
Omitting the switch uses the source's driver when one is configured. Disabling the driver does not
disable other evolution such as photospheric expansion. Custom multiplicative
drivers use unit baseline unless their metadata supplies `mean_amplitude`.

A source without a driver needs no special handling. Omitting the switch or
setting it to `False` evaluates that source normally. Setting it to `True`
raises an error before realization. The built-in supernova model does not
accept a driving signal. Its radius, luminosity, and temperature already
describe its intrinsic evolution. Use an explicit `ModulatedSource` wrapper
if an additional brightness modulation is scientifically intended.

To construct a steady Kerr disk with no driver, omit `driving_signal` from the
disk constructor. This retains the existing static thin-disk model. To compare
a reverberating disk against its own mean lamp heating, retain its driver and
set `apply_driving_signal=False` instead.

The generation grid defaults to 1000 days of prehistory and a 7300-day maximum
duration. Fivefold FFT padding generates a longer stochastic series before
cropping. It does not repeat the retained realization. Sampling is lazy and
occurs once per bound driver, so a shorter light-curve request uses the same
samples. The 0.1-day driver grid and daily source sampling do not require daily
or sub-daily microlensing maps.

Set `history_days`, `max_duration_days`, and `padding_factor` on the driver for
other workloads. Queries outside the declared horizon raise an actionable
error. History must cover reverberation and macroimage delays. Changing the
generation grid changes the realization even with the same seed.

A per-call `source=` override or a `with_source(...)` replacement uses its own
geometry and driver in both single-image and multi-image calculations. It
never inherits the previous source's driver. Independent-system batches apply
the same switch to every input and reject `True` if any source lacks a driver.

Sampled drivers reuse their interpolation tables on each requested device and
dtype. Treat sample tensors as immutable and construct a new driver to replace
them. Tables that require gradients are not cached. Concurrent curves sharing
one bound driver also share one generated realization, including unseeded runs.

Explicit sampled and callable signals remain available for custom sources and
standalone continuum reverberation calculations, as described below.
Explicit sample times cannot be combined with the automatic cadence or horizon
arguments. The requested sample times already define that grid.

## Physical source interface

Microlensing solvers never depend on a particular source class. A pixelated
source supplies a `SourceGeometry` and returns brightness with shape
`[time, y, x, band]`. Band names and wavelengths are arbitrary. LSST `ugrizy`
is an application choice, not a package constraint.

For photometry, source brightness is observed spectral flux density per
projected source-plane area in `Jy m^-2`. Multiplication by physical pixel area
therefore produces light curves in Jy. Built-in disks and supernovae already
satisfy this convention. `GaussianSource.total_flux` and
`GaussianModel.total_flux` are integrated flux densities in Jy. A
`StaticSource` or `CallableSource` supplied by a user must follow the same
convention. Dimensionless profiles remain useful for morphology and
magnification-only calculations, but need a physical normalization before
absolute photometry is requested. `flux_to_magnitude` converts Jy directly to
AB magnitudes using the fixed 3631 Jy definition.

Custom sources can specify their angular extent, pixel count, and wavelengths
directly. The system supplies its distances when it resolves the source.

```python
source = mc.CallableSource(
    my_brightness_function,
    source_grid_shape=256,
    field_of_view_uas=(8.0, 8.0),
    bands_angstrom={"blue": 4800.0, "red": 7500.0},  # observed wavelengths in Angstrom
)
```

The `bands_angstrom` mapping uses the same name and units as the built-in
physical source models.

The callable must return physical surface brightness in `Jy m^-2` when the
result will be converted to absolute fluxes or AB magnitudes.
It may accept only `times_days`, or `times_days, *, geometry` when it needs
physical coordinates or `geometry.pixel_area_m2` for normalization. The
geometry keyword is detected once during setup, not on every brightness call.
The system does not normalize or reinterpret the supplied brightness.

Static images use the same convention and infer the pixel dimensions from
their `[y, x, band]` array.

```python
source = mc.StaticSource(
    image,
    field_of_view_uas=8.0,
    bands_angstrom={"blue": 4800.0, "red": 7500.0},
)
```

Scalar extents describe square fields. Tuples use `(height, width)` in
microarcseconds, while shapes use `(ny, nx)`. Custom sources keep their
declared pixels even if the magnification map uses another resolution.

Explicit shared geometry is still available through `SourceGeometry(...)`.
Give either `field_of_view_uas` or physical `pixel_scale_m`, never both.
Angular geometry remains unresolved until the system supplies its distances.
For a standalone calculation with no microlensing system, call
`source.pixelate(source_redshift=1.5, H0=70, Om0=0.3)` before evaluating a
callback that needs physical pixel sizes. An explicit `distances` object is
still accepted for a custom cosmology. The same source specification can
be reused at different redshifts without changing the original object.

Built-in lightweight sources include `StaticSource`, `GaussianSource`, an
elliptical Gaussian with an optional smooth central hole, physical thin disks,
and configurable expanding photospheres. `CallableSource` remains the shortest
bridge to a completely custom disk, transient, binary source, or external
radiation calculation.

The physical wrappers `GaussianModel`, `ThinDiskModel`, and `KerrDiskModel`
choose their own source support and pixel geometry. They can therefore be
passed directly to `MicrolensingSystem`. `GaussianModel` accepts `sigma_uas`
or `sigma_m` directly, with optional `center_uas` or `center_m` and
`hole_radius_uas` or `hole_radius_m`. Its `position_angle_deg` is in degrees,
as for the disk models. Do not supply both units for the same quantity. Give
the trajectory and duration to the system when a source also moves:

```python
system = mc.MicrolensingSystem(
    lens_redshift=0.04,
    source_redshift=1.7,
    macro=macro,
    source=source_model,
    stellar_population=population,
    trajectory=trajectory,
    duration_days=3650.0,
)
```

The source parameters determine its native angular support. The system then
enlarges the map field, when necessary, to cover the complete trajectory.

## Expanding supernovae

`ExpandingPhotosphereSource` does not hard-code a Type Ia model. Its temporal
physics is any object implementing the four-method `PhotosphereEvolution`
protocol. The included analytic model requires every scientific parameter
explicitly:

```python
evolution = mc.PowerLawExponentialPhotosphere(
    peak_time_rest_days=17.5,
    peak_luminosity_watts=1.3e36,
    rise_power=2.2,
    decline_time_rest_days=31.0,
    photosphere_velocity_km_s=11_500.0,
    initial_radius_m=8.0e8,
    temperature_floor_k=3_200.0,
    temperature_ceiling_k=16_500.0,
)

source = mc.ExpandingPhotosphereSource(
    bands_angstrom={"blue": 4800.0, "middle": 6200.0, "red": 7500.0},
    maximum_observer_time_days=180.0,
    evolution=evolution,
    source_grid_shape=256,
    source_margin=1.05,
    appearance=mc.PhotosphereAppearance(position_angle_deg=25.0),
)
```

Pass this model directly to a system. It supplies the source redshift and
converts its angular-diameter distance to a luminosity distance. Explicit
source redshifts must agree with the system. A known `luminosity_distance_m`
can still override the brightness distance.

The largest radius over `maximum_observer_time_days` fixes the source grid.
Shorter light-curve requests keep that grid unchanged. Longer source-time
requests raise an error explaining how to extend the horizon. For a standalone
source, use `source.pixelate(source_redshift=0.65, H0=70, Om0=0.3)` before
evaluating brightness. No lens parameters are needed. Source evolution and
map cadence remain independent.
For a non-monotonic or sharply varying custom radius law, pass a conservative
`maximum_photosphere_radius_m` directly instead of relying on initialization
sampling to size the fixed field.

`PhotosphereAppearance` controls the built-in projected ellipse, limb
darkening, chromatic evolution, and optional UV suppression. For models that
need more structure, pass differentiable Torch `spatial_profile` and
`spectral_modifier` callables. For radiation-hydrodynamic calculations,
implement `PhotosphereEvolution` using tabulated or differentiable luminosity,
radius, and temperature laws. The microlensing interfaces do not change.

`paper_type_ia_supernova_source(...)` is a clearly labeled reproducibility
preset for the paper prototype. Its values are not selected by the general
constructor, and it is not intended as a precision radiative-transfer or
distance-inference model.

Intrinsic variability is an independent layer:

```python
signal = mc.TabulatedDrivingSignal(
    times_days=torch.arange(0.0, 101.0),
    values=my_positive_amplitudes,
    extrapolation="error",
)
variable_source = mc.ModulatedSource(static_source, signal)
```

The wrapped source may also be a `ThinDiskModel`, `GaussianModel`, or other
physical model. The system handles pixelization, so there is no need to
calculate distances or call `.pixelate()` before adding coherent modulation.

The table may contain one achromatic amplitude or one amplitude per band.
`CallableDrivingSignal` accepts an arbitrary differentiable PyTorch callable.
Signals are multiplicative and must be finite and non-negative. More involved
reverberation models can implement the same `PixelatedSource` protocol without
passing through this simple wrapper.

A custom source with an attached `driving_signal` can implement
`with_driving_signal(signal)` to return a copy with that signal. This lets the
system bind the inherited variability seed and lets the driver switch provide
a constant mean amplitude without changing the source's physical response.
Custom source evolution remains the responsibility of `brightness(...)`.

The resolved-quasar example driver is a lognormal realization of a smoothly broken power-law
PSD, not a damped random walk:

```python
signal = mc.broken_power_law_driving_signal(
    regularly_sampled_times,
    break_timescale_days=200.0,
    alpha_L=1.0,
    alpha_R=3.0,
    standard_deviation=0.3,
    seed=0,
)
```

This convenience function reproduces the paper convention. Random Fourier
phases, five-times padding, trimming, normalization, and a positive lognormal
transform. `BrokenPowerLawPSD` can also be evaluated independently.

No PSD shape is required by the package. Supply any callable that maps positive
frequencies in inverse days to non-negative power:

```python
def my_psd(frequency_per_day):
    return model_power(frequency_per_day)

signal = mc.driving_signal_from_psd(
    regularly_sampled_times,
    my_psd,
    fourier_sampling="gaussian",  # or "random_phase"
    amplitude_transform="lognormal",
    seed=0,
)
```

A PSD may return `[frequency]` or `[band, frequency]`. Mean amplitudes and
standard deviations may likewise be scalar or per-band. Users who already have
a realization can use `TabulatedDrivingSignal`. Fully custom time-domain or
differentiable variability remains available through `CallableDrivingSignal`
or by implementing the small `DrivingSignal` protocol. A DRW generator remains
available as an optional convenience model but is not the resolved-quasar default.

For a known response-delay map, `DelayedModulatedSource` evaluates the same
driving-signal protocol at `t - delay[y, x]` in one batched call:

```python
delayed = mc.DelayedModulatedSource.from_observer_transfer(
    static_source,
    signal,
    observer_transfer,
)
```

This adapter deliberately describes delayed multiplicative variability, not a
specific thermal-reprocessing law.

`ThermalReprocessingSource` supplies the physical nonlinear Planck layer for
continuum reverberation. The caller provides an additive heating map in
temperature-to-the-fourth per unit driver and a total delay map:

```python
reprocessed = mc.ThermalReprocessingSource(
    geometry,
    observer_transfer,
    signal,
    response_temperature4=lamp_heating_map,
    delay_days=total_lamp_to_disk_to_observer_delay,
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    spin=0.7,
    source_redshift=1.7,
)

weights = reprocessed.linear_response_weights()
psi = reprocessed.transfer_function(delay_edges_days, magnification=mu)
```

When only the first moment is needed, use the direct mean-delay operations:

```python
steady_mean = mc.steady_mean_response_delays(reprocessed)
microlensed_mean = mc.microlensed_mean_response_delays(
    reprocessed,
    magnification_map,
    distances,
)
mean_series = mc.microlensed_mean_response_delays_batch(
    reprocessed,
    magnification_maps,
    distances,
)
```

These operations do not construct delay bins. Repeated no-gradient calls also
reuse the source's invariant linear-response weights, and the batch operation
uses bounded spatial chunks for long dynamic-map sequences. Request a full
transfer function only when its delay-dependent shape is required.

Transfer functions are also first-class standalone data products. They do not
require a low-level `microcaustics.multi_image.MultiImageSimulation`:

```python
steady = mc.steady_transfer_function(
    reprocessed,
    delay_edges_days,
)

microlensed = mc.microlensed_transfer_function(
    reprocessed,
    magnification_map,
    distances,
    delay_edges_days,
)

series = simulation.transfer_functions(
    lens_region,
    source_grid,
    map_times_days,
    reprocessed,
    distances,
    delay_edges_days,
    method=ipm,
    response_batch_size=16,
)
```

Built-in thermal sources batch full transfer-function accumulation across map
epochs. By default the response batch follows the dynamic temporal batch;
`response_batch_size` can lower its memory footprint independently, while
`response_spatial_chunk_size` bounds the temporary pixel-by-band product.
Custom transfer-function sources without a batched implementation continue to
use the scalar streaming path.

Each product contains the binned response, band names, bin edges, mean lags,
normalization and source provenance. The time-dependent interface streams maps
and retains only response products unless a map observer explicitly saves a
frame. Any custom source implementing `TransferFunctionSource` can use the same
interfaces. The built-in thermal disk is not required.

The time-domain source evaluates
`T^4 = T_viscous^4 + response_temperature4 * driver(t-delay)` before the
redshifted Planck calculation. The linear response is the analytic derivative
of that same nonlinear source, not a separate normalization convention.
Extended coronae and line emission can implement `PixelatedSource` directly
while reusing the transfer's delay and emission-azimuth maps.

Source evolution and dynamic magnification-map cadence are deliberately
separate. A source can be evaluated daily while maps are generated less often,
or vice versa, as long as the caller chooses and validates the interpolation
appropriate to the application.

## Automatic physical source grids

`ThinDiskModel` and `GaussianModel` separate physical source parameters from
pixelization. They choose a conservative angular field when passed to
`MicrolensingSystem`:

```python
source = mc.ThinDiskModel(
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    bands_angstrom={"blue": 4800.0, "red": 9700.0},
    inclination_deg=45.0,
    source_grid_shape=1024,
    enclosed_flux_fraction=0.999,
    source_margin=1.05,
)
```

The default viscous flux profile is `"novikov-thorne"`; select
`"shakura-sunyaev"` for the Newtonian thin-disk profile. A custom callable
receives `(radius_rg, spin, isco_rg)` and may carry additional parameters in a
closure, `functools.partial`, or callable object. Custom profiles require an
explicit `radiative_efficiency`, supplied as a value, one of the two built-in
names, or a callable of `(spin, isco_rg)`. This keeps the high-level API fixed
without restricting additional model parameters. See the disk-flux-profile
notebook for a parameterized example.

The reddest thin-disk band sets the common outer support. The Gaussian model
uses the widest requested band and includes ellipticity, orientation, and an
offset center in its bounding grid. Advanced callers may override the derived
grid, but ordinary quasar and supernova calculations should not do so.

The physical support radius is also used when constructing the complete
stellar aperture. For a custom pixelated source whose grid includes known
zero-valued corners, pass `source_support_radius_uas` to `MicrolensingSystem`
to expose its true circular major-axis support. If it is omitted, the package
conservatively treats the entire rectangular grid as requested map area.

`ExpandingPhotosphereSource` already derives a fixed grid from the maximum
photosphere radius over its configured observer interval. It can therefore be
passed directly to `MicrolensingSystem`. Existing `StaticSource`,
`CallableSource`, and other pixelated sources are also accepted. Their geometry
defines the angular source field unless an explicit matching grid is supplied.
