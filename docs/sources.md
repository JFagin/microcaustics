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
To obtain both products in one map calculation, set
`include_microlensing_only=True`. The comparison is returned as
`curve.microlensing_only_flux` and `curve.microlensing_only_magnitude`.

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
other workloads. Queries outside the configured time range raise an error.
History must cover reverberation and macroimage delays. Changing the
generation grid changes the realization even with the same seed.

A per-call `source=` override or a `with_source(...)` replacement uses its own
geometry and driver in both single-image and multi-image calculations. It
never inherits the previous source's driver. Independent-system batches apply
the same switch to every input and reject `True` if any source lacks a driver.

Sampled drivers reuse their interpolation tables on each requested device and
dtype. Do not modify these sampled arrays in place. Construct a new driver to
replace them. Tables that require gradients are not cached. Concurrent curves sharing
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

Several standalone physical sources can also be resolved together:

```python
sources = mc.batched_pixelate_sources(
    disk_models,
    distances,
    batch_size=3,
    runtime=runtime,
)
```

On CUDA, compatible driven Kerr disks calculate observer delays together in
fixed-size GPU batches. Other source types and incompatible settings use their ordinary serial
calculation. This changes scheduling only. The requested model order and each
source's independent physical parameters are preserved, CUDA memory exhaustion
reduces the active source batch without changing numerical settings, and later
calls reuse the same compiled kernel specialization.

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

This convenience function draws Gaussian real and imaginary Fourier coefficients
(random amplitudes and phases), generates a five-times-longer series, and retains
an interior segment. It standardizes that latent sequence before applying the
positive lognormal transform. The requested PSD describes the latent sequence,
not the exact PSD after exponentiation. `BrokenPowerLawPSD` can also be evaluated
independently. This replaces the previous fixed-amplitude random-phase default,
so seeded driver realizations change, but stellar random draws do not.

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

The observer transfer and heating maps are achromatic, so the same pixelated
disk can be evaluated at any set of observed wavelengths without rebuilding
them. The mean-driver source retains the mean lamppost heating while disabling
only its stochastic fluctuations:

```python
spectrum_source = reprocessed.with_bands({
    f"lambda_{wavelength:05d}": float(wavelength)
    for wavelength in range(3000, 11001, 20)
})
mean_spectrum_source = spectrum_source.at_driver_mean()

spectrum = mc.light_curve_from_maps(
    maps,
    spectrum_source,
    map_times_days,
    distances,
    batch_size=1,
)
```

Here the result is an ordinary multiband `LightCurve`. `batch_size` bounds the
temporary wavelength-by-pixel cube without changing the calculation.

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

Transfer functions can also be calculated on their own. They do not
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
epochs. By default the response batch follows the dynamic temporal batch.
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

For many wavelength channels, keep the physical model on the system and select
the requested bands at light-curve time. Include the reddest requested
wavelength in the configured model so its automatic support is large enough:

```python
spectral_bands = {
    f"lambda_{w:05d}": float(w) for w in range(3000, 11001, 20)
}
spectrum = system.light_curve(
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    bands_angstrom=spectral_bands,
    band_batch_size=32,
    include_microlensing_only=True,
)
```

For several products from one map sequence, prefer
`LightCurveRequest(bands_angstrom=...)` with `system.light_curves`. Requests can
independently set `flux_cadence_days` and `apply_driving_signal=False`. No
manual distance calculation, pixelation, or constant-driver source is needed.
The final partial wavelength batch is padded internally and trimmed from the
returned `[time, band]` arrays.

## Empirical quasar spectra and bandpasses

`QuasarSpectrum` adds static broad and narrow emission lines, host-galaxy
light, intrinsic dust attenuation, and intergalactic absorption to the
physical disk continuum. `QuasarSpectrumPopulation` samples its configurable
parameters with an isolated random generator, so its seed does not alter the
disk, driver, stars, or observing cadence.
The empirical templates and their luminosity-dependent scaling follow
[Temple et al. (2021)](https://arxiv.org/abs/2109.04472), with the physical
continuum construction adapted from the spectral workflow in
[Fagin et al. (2024)](https://arxiv.org/abs/2410.18423).

```python
model = mc.QuasarSpectrumPopulation(
    host_fraction=mc.ClippedNormal(0.244, 0.075, 0.0, 0.7),
    ebv=mc.LogNormal(-2.0, 0.5),
).sample(seed=42)

result = system.light_curve(
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    bandpasses="lsst",
    spectral_model=model,
    wavelength_samples=32,
    wavelength_batch_size=16,
    return_spectrum=True,
    include_microlensing_only=True,
    spectrum_wavelengths=np.linspace(3250.0, 10900.0, 1000),
)
```

`wavelength_samples` controls the log-spaced physical continuum grid across the
combined filter support (32 by default). Normalizing host or line templates
may require additional disk wavelengths outside that range.
`wavelength_batch_size` controls how many spatial disk channels are
evaluated together. Emission-line templates and response curves are evaluated
on a separate fine one-dimensional grid using log-flux interpolation of the
continuum. Consequently, increasing
`spectrum_wavelengths` does not increase the number of spatial source images.
Increase `wavelength_samples` and check convergence for unusually sharp
chromatic caustic features or custom filters with narrow structure.
Omitting `spectrum_wavelengths` retains 1000 evenly spaced samples across the
combined filter support.
The returned `LightCurve` contains six integrated bands in `flux`, a shared-map
mean-driver result in `microlensing_only_flux`, component band fluxes in
`component_flux`, and the optional `[time, wavelength]` product in `spectrum`.

The continuum and its bandpass integrals are evaluated at every requested
source epoch, including daily intrinsic variability between 25-day maps.

### Response-based source evolution

By default, `source_evolution="exact"` evaluates the nonlinear thermal disk at
every source epoch. The `linear_response`, `quadratic_response`, and
`cubic_response` modes approximate its response to the driving signal, reducing
the cost of variable-source calculations, particularly for spectra.

For a stationary `ThermalReprocessingSource`, `source_evolution="linear_response"`
multiplies the mean disk and its unnormalized delay response by each
magnification map and sums over pixels. It interpolates these results between
map epochs and combines the response with the driving signal. Only the spatial
disk calculation is approximated. The driving signal and output light curve
keep their original time sampling. The unnormalized kernel is essential because microlensing changes
both its shape and response amplitude. The default CDF/PCHIP reconstruction
conserves total response mass. Fine fractional deposition and averaging four
shifted CDF grids reduce artifacts from individual pixel delays. The same
polynomials generate transfer-function products and convolve with the driver,
without plot-only smoothing.

The option `source_evolution="cubic_response"` adds
one sixth of the third brightness derivative convolved with the cubed driver
fluctuation. It uses the same signed CDF reconstruction as quadratic response.
Temperature and Planck terms are shared across all three derivatives, driver
powers share integration work, and the final flux uses one combined matrix
multiplication. The ordinary-deposition option also supports cubic evolution.
For large float32 CUDA response projections, the Triton path
groups pixels by delay bin and shares their reads between signed response
terms and neighboring bins. This changes only how the existing deposition
is computed, not its weights or the CDF reconstruction.
The source model, stellar mapping, and exact-evolution default are unchanged.
Higher order is not guaranteed to improve large excursions or errors dominated
by numerical response reconstruction. The notebook compares driver standard
deviations of 0.1, 0.3, and 0.5 before recommending a mode for that example.

The notebook `examples/notebooks/source_models/06_source_evolution_comparison.ipynb`
derives the linear, quadratic, and cubic flux approximations and generates a matched
comparison with exact evolution, including per-band errors and complete-call
timings.

```python
approximate = system.light_curve(
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    bandpasses="lsst",
    wavelength_samples=32,
    wavelength_batch_size=16,
    include_microlensing_only=True,
    source_evolution="linear_response",  # opt in; "exact" is the default
    response_delay_bin_days=None,  # automatic native CDF spacing
)
```

The response modes select the coarse CDF spacing once near the observer-frame
light-crossing time of a source pixel, bounded by the narrowest unlensed
response width and the tabulated driver's cadence. Set a positive
`response_delay_bin_days` to select it explicitly. Output transfer-function
edges do not set the native resolution. Tabulated drivers are integrated
analytically over each PCHIP piece. Scalar callable drivers use three-point
Gaussian quadrature. The common polynomial intervals, one quarter of the
coarse spacing, must not exceed the tabulated driver cadence.
No ordinary-response comparison or convergence audit runs per map.

`RuntimeConfig(response_reconstruction="ordinary")` restores the previous
two-node, moment-preserving response deposition, with 0.1-day bins when no
spacing is supplied. The same setting selects direct thermal transfer
histograms. Neither option changes `source_evolution="exact"`, which remains
the default evolution mode. CDF preserves response mass, but its centroid is
only approximately preserved. Transfer products retain direct pixel mean lags.

The same controls work on individual `LightCurveRequest` objects, with
optional caustic labels, and with `TimeShiftedSource` arrival delays. Source
trajectories must be stationary in these modes. A moving source currently
raises an error rather than reusing invalid response weights.

This is an approximation to the nonlinear Planck response, not merely a
coarser time grid. In one matched Q2237 B full-field test (1024×1024-pixel disk,
10 million rays per map, 147 maps, 3651 daily epochs, 32 continuum wavelengths,
RTX 5070 Ti), warmed complete calls were about 1.37–1.38 s exact versus
1.01–1.02 s linear response after compiling the map-response projection.
Across that test the maximum six-band error was 2.01 mmag, and the maximum
within 25 days of the day-1300 crossing was
1.96 mmag. A best-shift fit to the crossing-window variable component gave
0.13–0.24 day across the six bands relative to exact, even though the binned
response itself preserves its mean delay. The shift reflects the physical
first-order approximation and can matter for continuum-lag studies.

With a sampled `QuasarSpectrum` and the same 32-wavelength continuum,
warmed calls were 1.57–1.65 s exact versus 1.10–1.15 s with linear
response (about 1.43 times faster). Both runs returned a spectrum at every
daily epoch. Neither reduced the intrinsic cadence to the 25-day map cadence.
These timings include map generation and spectral post-processing but not
caustic labels or first-call compilation.

`source_evolution="quadratic_response"` adds the second-order thermal
coefficient. In the full-spectrum test its maximum six-band error was 0.09 mmag
and the best-fit shift was 0.01–0.02 day. With the same compiled projection,
it took 1.16–1.19 s, about 1.33 times faster than exact in this case.
These historical measurements used ordinary 0.25-day response bins. They are for one
realization and are not general error bounds. Compare with `"exact"` for
other disks, drivers, and caustics. The response modes cache one spatial
weight per sampled disk wavelength and use more GPU memory than the exact
streaming path. Moderate wavelength sets use one compiled map-response calculation.
Larger sets are processed in chunks within a GPU-memory limit. Check available
memory before using hundreds of wavelengths.

Thermal disks share the retarded driving signal and temperature field across
wavelength chunks. On compiled backends, regular grids use tiled Planck emission
and map-weighted flux reductions without retaining a full
time-by-pixel-by-wavelength cube. Irregular grids remain supported.
NVIDIA CUDA float32 evaluates wavelengths together in a Triton kernel. Requests needing
gradients, float64, or another GPU backend retain the compiled PyTorch path.
This also applies to source-only photometry and time-shifted multi-image sources.
Temporal tails repeat the last valid evaluation time internally and are trimmed
from the result, so padding never extends the driving-signal horizon.

For a stationary mean-driver disk, the code multiplies the static brightness by
each map, sums over pixels once, and interpolates the resulting band fluxes. This is algebraically
equivalent to the original map interpolation, not an approximation to intrinsic
variability. Cached static brightness uses a fixed memory budget. Larger spectra retain
the wavelength-streaming path. `include_labels=True` shares the same map stream
and preserves the caustic labels through spectral post-processing.

The physical continuum determines the empirical $M_i(z=2)$ luminosity scaling
unless `absolute_i_magnitude` is supplied. In a lensed macroimage, host light is
omitted by default because a local microlensing model does not specify the host
aperture or its macro magnification. Emission lines are static and receive the
local macro magnification, but are not microlensed. `host_lensing="macro"` and
`host_lensing="unlensed"` are explicit approximations. The unlensed
`system.source_light_curve(...)` path includes the host directly. Pass a
`BandpassSet` instead of `"lsst"` for custom response curves.

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

The default viscous flux profile is `"novikov-thorne"`. Select
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
