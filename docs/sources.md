# Source models and driving signals

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

Custom pixelated sources can define their geometry directly in observable
angular units. The constructor performs the source-plane distance conversion,
so notebook and application code does not need a manual microarcsecond-to-metre
constant:

```python
geometry = mc.SourceGeometry.from_angular(
    distances,
    shape=(512, 512),
    field_of_view_uas=(8.0, 8.0),
    bands=("blue", "red"),
)
source = mc.CallableSource(geometry, my_brightness_function)
```

The callable must return physical surface brightness in `Jy m^-2` when the
result will be converted to absolute fluxes or AB magnitudes.

Built-in lightweight sources include `StaticSource`, `GaussianSource`, an
elliptical Gaussian with an optional smooth central hole, physical thin disks,
and configurable expanding photospheres. `CallableSource` remains the shortest
bridge to a completely custom disk, transient, binary source, or external
radiation calculation.

The physical wrappers `GaussianModel`, `ThinDiskModel`, and `KerrDiskModel`
choose their own source support and pixel geometry. They can therefore be
passed directly to `MicrolensingSystem`. `GaussianModel.from_angular` accepts
observational widths in microarcseconds without manual unit conversion. Give
the trajectory and duration to the system when a source also moves:

```python
system = mc.MicrolensingSystem.from_redshifts(
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
    redshift=0.65,
    wavelengths_angstrom=(4_800.0, 6_200.0, 7_500.0),
    band_names=("blue", "middle", "red"),
    maximum_observer_time_days=180.0,
    evolution=evolution,
    resolution=256,
)
```

The source grid is fixed by the largest radius over the requested interval.
the photosphere can evolve daily while dynamic maps use an independently
chosen cadence. Arbitrary bands and resolutions are supported. A known
luminosity distance can be supplied directly, or the built-in flat-Lambda-CDM
distance calculation can be configured.
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

The table may contain one achromatic amplitude or one amplitude per band.
`CallableDrivingSignal` accepts an arbitrary differentiable PyTorch callable.
Signals are multiplicative and must be finite and non-negative. More involved
reverberation models can implement the same `PixelatedSource` protocol without
passing through this simple wrapper.

The resolved-quasar example driver is a lognormal realization of a smoothly broken power-law
PSD, not a damped random walk:

```python
signal = mc.broken_power_law_driving_signal(
    regularly_sampled_times,
    break_timescale_days=200.0,
    low_frequency_slope=1.0,
    high_frequency_slope=3.0,
    standard_deviation=0.3,
    seed=17,
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
    seed=17,
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
)
```

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
    bands={"blue": 4800.0, "red": 9700.0},  # Angstrom
    inclination_deg=45.0,
    resolution=1024,
    enclosed_flux_fraction=0.999,
    source_margin=1.05,
)
```

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
