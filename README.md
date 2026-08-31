# microcaustics

`microcaustics` is a Python package for static and dynamic gravitational
microlensing simulations. Version 1.0.0 packages the validated numerical
methods used for the accompanying methods paper behind a documented public
interface.

![Ten-year Q2237 image-B-like dynamic magnification map with source-plane caustics](docs/assets/q2237_image_b_dynamic_magnification.gif)

The animation is a complete 147-frame output of the first scientific
tutorial. It shows a ten-year, $10^7$-ray, $1024^2$ dynamic IPM calculation with the
caustic network overlaid. Its map normalization, colorbar, and indexed GIF
palette are fixed globally across all frames.

The portable reference layer includes the following capabilities.

- exact chunked point-mass ray tracing and analytic Jacobians.
- source-independent uniform-grid inverse ray shooting (IRS).
- full-field and nested-scout IPM.
- a direct-cell Triton IPM rasterizer plus an exact portable overlap reference.
- a readable local-exact complex-Taylor far-field approximation.
- fused batches of unrelated static maps and moving-star map sequences on CUDA, with
  portable frame-by-frame equivalents.
- critical curves, caustics, binary label maps, and distance maps.
- physically calibrated multiband pixelated/callable sources and analytic Gaussians.
- composable tabulated or callable intrinsic driving signals.
- physical Page--Thorne thin disks with non-GR, approximate-GR, and primary
  full-Kerr observer transfers.
- nonlinear thermal reverberation with arbitrary delayed heating maps and
  linear transfer-function extraction.
- reproducible point-mass populations with extensible mass functions.
- moving-source finite-source light curves.
- coherent resolved multi-image light curves with independent lens fields,
  trajectories, numerics, and cosmological arrival delays.

The fused dynamic IPM production path is available. It combines temporal
Taylor far-field queries, a shared conservative cell queue, biquadratic interpolation,
and direct-cell rasterization without materializing triangle arrays. Optional
caustic/label products reuse the same per-epoch far-field states, but remain opt-in so
users do not pay for them when only maps or light curves are requested.

## Installation

Create a Python 3.10+ environment, then install PyTorch **before** installing
`microcaustics`. Use the live
[PyTorch installation selector](https://pytorch.org/get-started/locally/) to
choose the correct command for the operating system and compute platform. In
particular, select a CUDA build rather than the CPU build for an NVIDIA GPU.
The exact CUDA wheel command changes between PyTorch releases.

Verify the PyTorch installation before continuing.

```bash
python -c "import torch; print('torch', torch.__version__); print('CUDA', torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS')"
```

Install the tagged release directly from GitHub after PyTorch is available.

```bash
python -m pip install "microcaustics @ git+https://github.com/JFagin/microcaustics.git@v1.0.0"
```

Clone the repository when you want to run the notebooks, edit the source, or
contribute changes. A complete development installation is available from the
repository root.

```bash
python -m pip install -e ".[dev,docs,notebooks,test,validation]"
```

For an editable installation without development tools, use
`python -m pip install -e .`. Plotting, notebooks, astronomy dependencies, and
external validation packages remain optional extras. The standalone scripts
are indexed in [`examples/README.md`](examples/README.md).

The numerical core supports Python 3.10 and newer. The optional `caustics`
interoperability layer requires Python 3.11 or newer because that is the
minimum version supported by `caustics>=1.7`.

### Triton acceleration

The production fused IPM path requires an NVIDIA CUDA device, float32, and an
importable Triton package. This path is significantly faster than compiled PyTorch.

- **Linux.** Upstream Triton officially supports Linux. Install PyTorch first.
  If `import triton` is still unavailable, follow the
  [official Triton installation and compatibility guidance](https://github.com/triton-lang/triton#compatibility).
  Keep the Triton minor version compatible with the installed PyTorch release.
- **Native Windows.** Upstream Triton does not officially support Windows.
  Use the community-maintained
  [`triton-windows` distribution](https://github.com/triton-lang/triton-windows)
  and follow its current PyTorch/Triton compatibility table. For example, only
  when that table assigns Triton 3.7 to the installed PyTorch release.

  ```powershell
  python -m pip uninstall -y triton
  python -m pip install -U "triton-windows>=3.7,<3.8"
  ```

  Do not blindly reuse that example for a different PyTorch minor release.
  Current wheels bundle the minimal CUDA toolchain, but require a current
  NVIDIA driver and the Microsoft Visual C++ Redistributable documented by the
  project.
- **Windows through WSL2.** Install
  [WSL](https://learn.microsoft.com/windows/wsl/install), enable NVIDIA's
  [CUDA support for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
  and then follow the Linux path above. This stays within upstream Triton's
  supported operating system.
- **macOS or CPU.** Triton is not used. The package selects its portable
  eager/compiled PyTorch implementation. Apple MPS is supported where PyTorch
  supports the requested operations.

Check the detected runtime.

```bash
microcaustics doctor
```

For a timing or validation run that must not silently fall back to PyTorch,
request `RuntimeConfig(device="cuda", backend="triton", dtype="float32",
strict_backend=True)`. The default CUDA memory ceiling is 95 percent, and
lossless OOM recovery reduces chunks or splits batches when necessary. See the
[portability and troubleshooting guide](docs/portability_and_troubleshooting.md)
for cache, compiler, first-call timing, and out-of-memory guidance.

## Tutorial notebooks

The notebooks are the recommended introduction to the package because they
show complete scientific workflows, physical coordinate conventions, and
accelerator timing. Install the notebook dependencies and launch the
collection with the following commands.

```bash
python -m pip install -e ".[notebooks,science]"
jupyter lab examples/notebooks
```

Start with
[Getting started with maps and light curves](examples/notebooks/getting_started/01_q2237_production_light_curve_and_gif.ipynb).
It generates a ten-year dynamic magnification sequence, finite-source light
curves with and without intrinsic variability, source-center caustic labels,
and the animation shown above. The calculation uses the documented production
IPM and far-field settings rather than a simplified plotting surrogate.

The complete notebook guide is organized into five tracks.

| Track | Notebooks |
|---|---|
| Getting started | [Static maps and numerical methods](examples/notebooks/getting_started/00_static_maps_and_numerical_methods.ipynb), [Q2237 production light curves](examples/notebooks/getting_started/01_q2237_production_light_curve_and_gif.ipynb), and [dynamic maps and light curves](examples/notebooks/getting_started/02_dynamic_maps_and_light_curves.ipynb) |
| Methods | [Stellar populations and mass functions](examples/notebooks/methods/00_stellar_populations_and_mass_functions.ipynb), [far-field approximation](examples/notebooks/methods/01_far_field_approximation.ipynb), and [caustics and labels](examples/notebooks/methods/02_caustics_and_labels.ipynb) |
| Source models | [Relativistic disks and reverberation](examples/notebooks/source_models/00_relativistic_disks_and_reverberation.ipynb), [expanding supernovae](examples/notebooks/source_models/01_expanding_supernovae.ipynb), and [custom sources and variability](examples/notebooks/source_models/02_custom_sources_and_variability.ipynb) |
| Workflows | [Multi-image light curves and observations](examples/notebooks/workflows/00_multi_image_light_curves_and_observations.ipynb), [realistic strong-lens images](examples/notebooks/workflows/01_realistic_strong_lens_image.ipynb), [end-to-end lensed quasars](examples/notebooks/workflows/02_end_to_end_lensed_quasar.ipynb), [streaming and export](examples/notebooks/workflows/03_streaming_and_exporting_results.ipynb), and [simulation datasets](examples/notebooks/workflows/04_simulation_datasets.ipynb) |
| Validation | [Weisenbach IPM](examples/notebooks/validation/00_weisenbach_ipm_visual_validation.ipynb), [SIM5 GR](examples/notebooks/validation/01_sim5_gr_visual_validation.ipynb), [accuracy and performance](examples/notebooks/validation/02_accuracy_and_performance.ipynb), and [analytic single-point lens](examples/notebooks/validation/03_single_point_lens_validation.ipynb) |

The validation notebooks can use independently generated Weisenbach and SIM5
products when those external codes are available. The remaining notebooks run
only with the public `microcaustics` interface and their documented optional
dependencies.

## Getting started with maps and light curves

This example follows the first tutorial notebook. It defines one Q2237 image
B-like physical system, generates an independent magnification map, and then
generates a ten-year light curve with source-center caustic labels. The
requested light-curve time axis automatically expands the stellar aperture
and map geometry for the full trajectory. The package derives the source grid, circular stellar field,
number of stars, Einstein radii, stellar velocities, and lens-plane bounds.

### 1. Define the lens, source, and stellar population

```python
import torch
import microcaustics as mc
import microcaustics.plotting as mcp

lens_redshift = 0.0395
source_redshift = 1.695

macro = mc.MacroLens(
    convergence=0.391,
    shear=0.391,
    shear_angle_deg=141.73,
    smooth_matter_fraction=0.0,
)

source = mc.KerrDiskModel(
    black_hole_mass_solar=10.0**9.08,
    eddington_ratio=0.34,
    bands_angstrom={"u": 3671, "g": 4827, "r": 6223,
                     "i": 7546, "z": 8691, "y": 9712},
    spin=0.74,
    inclination_deg=10.0,
    position_angle_deg=175.0,
    lamp_fraction=0.1,
    corona_height_above_isco_rg=20.0,
    source_grid_shape=1024,          # pixels per source-plane axis
    enclosed_flux_fraction=0.999,
    source_margin=1.05,
)

kinematics = mc.SkyProjectedKinematics(
    ra_deg=340.126125,
    dec_deg=3.358611,
    peculiar_velocity_dispersion_km_s=235.0,
    stellar_dispersion_km_s=170.0,
    seed=2001,
)
population = mc.StellarPopulation.salpeter(
    mean_mass_solar=0.3,
    mass_ratio=100,
    kinematics=kinematics,
)

system = mc.MicrolensingSystem(
    lens_redshift=lens_redshift,
    source_redshift=source_redshift,
    H0=70.0,                         # km s^-1 Mpc^-1
    Om0=0.3,                         # present-day matter density
    macro=macro,
    source=source,
    stellar_population=population,
    integration_domain="scout",     # "scout", "full", or "rectangle"
    light_loss=0.01,                 # stellar-aperture truncation tolerance
    safety_scale=1.5,                # expand the derived circular star field
    stellar_motion_sigma_margin=5.0, # motion allowance over the full duration
    seed=1001,
    caustic_grid_shape=8192,         # detA pixels per axis for labels
)
```

The system evaluates a flat matter-plus-cosmological-constant geometry in
Torch float32. Supply an explicit `LensingDistances` object for a different
expansion history. The
stellar-realization and peculiar-velocity seeds are separate, so both random
processes remain independently reproducible.

A single integer seed is enough for a reproducible calculation. The package
derives stable independent streams for stars, source variability,
observations, and each macroimage. A mapping can override only the desired
components, for example `seed={"base": 1001, "stars": 42}`.

Before allocating stars or compiling a kernel, inspect the derived geometry
and resource scale with `system.summary(duration_days=3650)`. The returned
dictionary can also be validated programmatically, and the call performs no
simulation.

`source_grid_shape` is the number of source pixels per axis, not a physical source
size. Quasar models derive their angular field from the black-hole, accretion,
wavelength, inclination, redshift, and enclosed-flux settings.
Expanding-supernova models derive it from the largest photospheric radius over
the requested evolution.
When a source trajectory is supplied to `MicrolensingSystem`, its duration is
also included automatically so the derived map field covers the complete
path. An explicit `PlaneGrid` is only needed for a source-independent map or
an intentionally pixelated custom source.

### 2. Generate and plot one magnification map

An independent static map uses the complete `k=1` source scout. This avoids the
dynamic sequence's one-time `k=1` to `k=2` normalization correction.

```python
magnification_map = system.magnification_map(
    time_days=0.0,
    rays=10_000_000,
)

figure, ax = mcp.plot_magnification_map(
    magnification_map,
    log10=True,
    scale_bar_uas=1.0,
    show_axes=False,
)
```

The operation infers the static `k=1` IPM preset. Common experiments remain
plain keyword changes, for example
`system.magnification_map(rays=5_000_000, refinement=3,
virtual_refinement=4, far_field=True)`. Use `method="irs"` for inverse ray
shooting, or set `integration_domain` on the system to `"scout"`, `"full"`,
or `"rectangle"`.

For a source-independent map, omit `source=` and supply only
`source_grid=mc.PlaneGrid((1024, 1024), (height_uas, width_uas))`. The tuple
order is `(y, x)`, and the field of view is in microarcseconds.

### 3. Generate a production light curve with labels

The shortest call uses the complete validated production path. This means
`N=10^7`, `k=2`, true refinement `r=2`, virtual refinement `v=4`, the
local-exact complex-Taylor far field, temporal batching, conservative
endpoint-union scout reuse, and aligned source-center labels.

```python

microlensing_only = system.light_curve_with_labels(
    duration_days=3650,
    map_cadence_days=25,
    keep_maps_at_days=(0.0,),
)

map_at_day_zero = microlensing_only.maps[0.0]
print(microlensing_only.light_curve.flux)  # 147 map epochs by 6 bands
print(microlensing_only.crossing_labels)   # source-center parity
print(microlensing_only.crossing_events)   # label transitions
print(microlensing_only.center_distances_uas)
print(map_at_day_zero.values)         # [1024, 1024]

figure, ax = mcp.plot_light_curve(
    microlensing_only.light_curve,
    bands=("i",),
    normalize=True,
    show_unlensed=True,
    title="Q2237 image B-like light curve",
)
```

Intrinsic variability can be evaluated daily while the dynamic maps remain
on the faster 25-day cadence. The two PSD slopes are explicit user controls.

```python
driver_times = torch.arange(0.0, 3651.0, 1.0)
driver = mc.broken_power_law_driving_signal(
    driver_times,
    break_timescale_days=200.0,  # PSD break timescale
    alpha_L=1.0,                 # low-frequency PSD slope
    alpha_R=3.0,                 # high-frequency PSD slope
    standard_deviation=0.10,     # fractional driving variability
    seed=3001,
    extrapolation="hold",        # hold the endpoint over small delay offsets
)

variable_system = system.with_source(source.with_driving_signal(driver))
combined = variable_system.light_curve_with_labels(
    duration_days=3650,
    map_cadence_days=25,         # microlensing map cadence
    source_cadence_days=1,       # intrinsic source cadence
)
```

Here is the same production calculation with its principal numerical controls
made explicit. These are ordinary configurations, so users can change any
value for accuracy, speed, memory, or convergence studies.

```python
far_field = mc.FarFieldApproxConfig(
    enabled=True,
    cells_per_axis=16,           # far-field spatial partition
    nodes_per_cell_axis=8,       # evaluation nodes per partition cell
    exact_radius_cells=1.0,      # larger values trace more nearby stars exactly
    taylor_order=4,              # higher values improve the far-star expansion
    center_translation_order=10, # accuracy when translating cell expansions
)

method = mc.production_ipm_config(
    rays=10_000_000,            # N
    scout_ratio=2,               # k, lower values use a denser scout
    refinement=2,                # r, true lens-equation refinement
    virtual_refinement=4,        # v, interpolated polygon refinement
    scout_halo_pixels=0.0,       # optional source-pixel coverage halo
    scout_dilation_cells=1,      # conservative neighboring-cell expansion
    dual_scout_scalar_correction=True,  # one-time k=1 to k=2 correction
    cell_chunk_size=524_288,     # lower this to reduce peak memory
    far_field_approx=far_field,
)

schedule = mc.production_dynamic_config(
    temporal_batch_size=30,      # optimized for the combined LC + label path
    light_curve_batch_size=None,  # optional sources sharing this map sequence
    fused_temporal_ipm=True,     # disable only for implementation validation
    scout_refresh_frames=10,     # lower for more frequent scout refreshes
)

labels = mc.CausticConfig(
    far_field_approx=far_field,
    # Labels inherit the shared 30-frame map batch. LC-only calls use the
    # 49-frame production preset because they omit detA and marching squares.
    anchor_count=9,              # more points add alignment redundancy
    gauge_count=9,
    minimum_determinant_sign_pixels=4,  # remove unresolved sign islands
    minimum_alignment_gauges=3, # minimum trusted temporal alignment set
)

explicit_result = system.light_curve_with_labels(
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    method=method,
    schedule=schedule,
    caustics=labels,
    keep_maps_at_days=(0.0,),
)
```

The short labeled call uses a shared 30-frame map and label batch. An LC-only
call uses the 49-frame production preset. The expanded example repeats the
settings intentionally so every effective value is visible. Lower-level label
query and Triton chunk controls remain available in `CausticConfig` for
unusual workloads.

Only the requested day-zero map is retained. The other full-resolution maps
are streamed through the finite-source photometry and label calculation rather
than stored as a large cube. The first tutorial extends this same workflow to
daily intrinsic variability, a 25-day microlensing cadence, and a fixed-scale
GIF of all 147 maps.

## Warmup, reuse, and profiling

Compiled Torch and Triton kernels are cached by compatible execution shape.
Reusing one `MicrolensingSystem` also reuses its seeded stellar realization.
An explicit warmup can separate first-call compilation from production work.

```python
# Use a representative temporal batch so the production batch shape is warm.
system.warmup_light_curve(labels=True)

# Subsequent compatible calls reuse the realization and warmed kernels.
next_result = system.light_curve_with_labels(
    duration_days=3650,
    map_cadence_days=25,
)
```

`system.light_curves(...)` batches multiple sources or trajectories through a
shared map sequence. This is what `light_curve_batch_size` controls.
`batched_system_maps(...)` batches unrelated static systems without sharing
their stars. `batched_system_light_curves(...)` runs complete independent
systems concurrently. All three interfaces reuse compatible compiled kernels
instead of recompiling for every realization.

Independent stellar realizations of the same macroimage can be generated
together. Each seed produces a new star field. Compatible Triton or
compiled-Torch kernels are reused.

```python
systems = [system.with_seed(seed) for seed in range(1001, 1009)]
maps = mc.batched_system_maps(
    systems,
    batch_size=8,  # automatically reduced if necessary to avoid an OOM
)
```

The same systems can produce independent dynamic light curves. Each retains
its own stars, source, trajectory, variability, maps, and labels.

```python
batch = mc.batched_system_light_curves(
    systems,
    map_times_days=range(0, 3651, 25),
    flux_times_days=range(0, 3651),
    curves_per_batch=3,  # complete independent systems, not shared maps
    include_labels=True,
)
curves = batch.light_curves
print(batch.seconds_per_curve, batch.executed_batch_sizes)
```

CUDA memory exhaustion reduces only the active concurrency and retries the
same numerical calculation. The best concurrency depends on the star count,
map geometry, labels, and GPU. It can be measured explicitly with
`mc.tune_system_light_curve_batch(..., candidates=(1, 2, 3, 4))`. Tuning is
not run silently during production.

Different convergence, shear, redshift, stellar, trajectory, variability, and
disk parameter values also reuse an existing compatible kernel specialization.
They do not need to share one stellar field and may be evaluated sequentially
or through the applicable batching interface. Keep the numerical shapes and
configuration fixed for maximum throughput. Changing source resolution, band
count, temporal batch size, dtype, backend, or the IPM and far-field shape
controls can require one new compiled specialization. A substantially
different stellar count can also produce a new compiled-Torch shape variant.
This compilation is cached and reused by later compatible realizations.

Timing collection is disabled by default because GPU synchronization can
reduce throughput. Request synchronized end-to-end or component timing only
when it is needed.

```python
runtime = mc.RuntimeConfig(profiling="total")
runtime = mc.RuntimeConfig(profiling="detailed")
```

With `profiling="off"`, `result.timing.collected` is false and no timing-only
accelerator barriers are inserted. Use `benchmark_callable` when comparing
first-call and warmed performance across repeated calls.

`integration_domain="scout"` is the fast production path. It evaluates only
cells that can map into the source field. `"full"` evaluates every cell in the
square bounding the complete circular stellar field. `"rectangle"` evaluates
the conventional light-loss rectangle while retaining the same complete
circular star population. The latter can use a separate
`rectangle_light_loss=` value on `MicrolensingSystem`.

For dynamic scout calculations, `scout_refresh_frames` is the reuse interval.
Endpoint union retains the cells selected at both ends of each interval, which
protects motion between refreshes. The far-field Taylor coefficients themselves
are rebuilt at every epoch in the validated production mode. There is no hidden
temporal coefficient reuse. Disable the far-field approximation with
`FarFieldApproxConfig(enabled=False)` for a direct point-mass reference calculation.

The main accuracy and throughput controls in the example are listed below.

| Control | Meaning |
|---|---|
| `rays` | Requested base lens-plane cell budget, $N$ |
| `integration_domain` | `"scout"`, `"full"`, or `"rectangle"` integration region |
| `scout_ratio` | Scout coarsening factor, $k$, used only by the scout domain |
| `refinement` | True lens-equation refinement, $r$ |
| `virtual_refinement` | Interpolated polygon refinement, $v$ |
| `FarFieldApproxConfig` | Local partition, exact-neighbor radius, Taylor order, and node resolution |
| `scout_refresh_frames` | Number of dynamic epochs sharing an endpoint-union scout selection |
| `temporal_batch_size` | Number of consecutive maps processed by one fused temporal batch |
| `light_curve_batch_size` | Number of sources or trajectories sampled from one shared map sequence |
| `curves_per_batch` | Number of complete independent systems run concurrently on one GPU |
| `cell_chunk_size` | Maximum spatial work chunk before automatic memory backoff |
| `caustic_grid_shape` | Resolution of determinant, critical-curve, and source-center-label products |

The production far-field configuration uses a `16 x 16` partition to decide
which stars are evaluated exactly. Each partition cell contains an `8 x 8`
Taylor evaluation lattice. The latter is the grid shown in the far-field
schematic and should not be confused with the older interpolation method.

The production constructors are presets rather than separate restricted code
paths. Every value above may be overridden, and the same controls work with
independent-map batches, multi-image calculations, custom sources, and directly
supplied stellar fields. Automatic tuning and lossless CUDA out-of-memory
backoff remain available when explicit batch or chunk values are omitted.

The default dynamic method is the validated `N=10^7`, `k=2`, `r=2`, `v=4`
scout IPM with the one-time `k=1` normalization correction. The stellar field
is always the complete circular field. `integration_domain="full"` evaluates
all cells in its bounding square, while `"rectangle"` changes only the launch
region and never removes stars. Directly supplied `PointMassField` objects and
explicit source or lens grids remain supported for controlled experiments.

Physical models such as `KerrDiskModel`, `ThinDiskModel`, and `GaussianModel`
choose their own pixel grids. Existing pixelated, callable, expanding-supernova, and custom
sources can still be supplied directly. A source is optional when only a
magnification map is required.

Point lenses may be supplied directly or sampled from any object implementing
the `MassFunction` protocol. Population builders expose their lens region and
report the realized compact convergence. They do not apply hidden light-loss
rectangles or safety factors. See
[`docs/lens_populations.md`](docs/lens_populations.md).

High-level operations choose the production scout automatically. An
independent static map uses `k=1`, while a moving sequence uses `k=2` plus the
one-time `k=1` normalization repair. Call `production_ipm_config()` only when
an advanced experiment needs an explicit numerical configuration.
Both use `N=10_000_000, r=2, v=4` by default. These are not fixed constants. The implementation
supports arbitrary sensible ray budgets and scout ratios, as well as
`virtual_refinement >= refinement >= 1`. CUDA float32 automatically uses the
direct-cell Triton rasterizer for `v <= 16`. CPU, Apple, float64, and larger
`v` use the same numerical method through the exact portable rasterizer. See
[`docs/ipm.md`](docs/ipm.md) for the meaning and tradeoffs of `N`, `k`, `r`,
and `v`.
The validated scout uses `scout_halo_pixels=0` and
`scout_dilation_cells=1`. The former expands the source aperture, whereas the
latter retains a one-cell support ring in the lens plane. Both can be changed
explicitly for convergence or safety studies.

Compatible unrelated static star fields can be evaluated together with
`batched_magnification_maps`. This is genuine independent-map batching. Each
request has its own stars, far-field state, scout mask, result, and metadata.
The [stellar-population notebook](examples/notebooks/methods/00_stellar_populations_and_mass_functions.ipynb)
demonstrates the interface across several stellar mass functions.
The dedicated [far-field notebook](examples/notebooks/methods/01_far_field_approximation.ipynb)
explains every `FarFieldApproxConfig` control, shows the production cell/node geometry,
and demonstrates direct-raytrace accuracy and runtime validation before a
setting is changed.

Dynamic calculations use `DynamicConfig` for static-map reuse, temporal and
multi-source batching, lossless CUDA OOM backoff, and optional endpoint-union
scout reuse.
See [`docs/dynamic.md`](docs/dynamic.md). Approximate reuse is always recorded
in map metadata rather than being silently enabled.

Portable exact deflection/Jacobian blocks, Taylor far-field queries, and IRS
deposition have real cached `torch.compile` implementations. Result metadata
separates the requested backend from what actually executed. The portable IPM
polygon clipper is deliberately retained as an exact Python reference and is
reported as `torch-compile-partial`. CUDA float32 uses the fully fused Triton
IPM rasterizer.

Temporal batches, IPM cell chunks, IRS ray chunks, and caustic Jacobian chunks
can optionally be selected by warmed, memory-bounded steady-state tuning.
Tuning is disabled by default and never changes physical or resolution
parameters. See [`docs/tuning.md`](docs/tuning.md) and
[`examples/automatic_tuning.py`](examples/automatic_tuning.py).

Runnable scripts live in [`examples`](examples). Reusable plotting helpers and
additional notebook guidance are described in
[`docs/plotting.md`](docs/plotting.md). The core numerical package does not
import Matplotlib or Jupyter.

The final tutorials also build labeled machine-learning training sets for a
fixed Q2237 image and for prior-sampled lens/disk parameters. Their companion
scripts use ordinary Python multiprocessing. No distributed launcher is
required, and each GPU retains one persistent worker and compiler cache.

```bash
python examples/generate_q2237_training_set.py --count 1000 --gpus 0 1 2 3
python examples/generate_random_training_set.py --count 1000 --gpus 0 1 2 3
```

Every example is written independently, so interrupted jobs resume without
recomputing completed curves. The first curve on each GPU includes any missing
compilation. Later curves reuse the same compiled kernels while physical lens,
star, disk, and variability parameters change.

Operational guides cover [method selection](docs/choosing_methods.md),
[coordinates and normalization](docs/conventions.md),
[portability and troubleshooting](docs/portability_and_troubleshooting.md),
[extension protocols](docs/extending.md), and
[streaming data products](docs/data_products.md). The final notebooks turn
method selection and export into executable CUDA-aware workflows.

Custom bands, pixelated sources, and intrinsic variability are described in
[`docs/sources.md`](docs/sources.md). Source evaluation remains independent of
map cadence and of the selected microlensing solver. The configurable
`ExpandingPhotosphereSource` supports user evolution objects, arbitrary bands,
custom spatial and spectral Torch functions, and independently sampled source
and map cadences. The paper's Type Ia-like prototype is only an explicit named
reproducibility preset.

Resolved lensed systems use the same physical interface. Supply a mapping of
image names to local `MacroLens` objects, followed by the distances, source,
stellar population, and measured arrival delays only once.

```python
system = mc.MultiImageSystem(
    images={"A": macro_a, "B": macro_b, "C": macro_c, "D": macro_d},
    lens_redshift=lens_redshift,
    source_redshift=source_redshift,
    H0=70.0,
    Om0=0.3,
    source=source,
    stellar_population=stellar_population,
    arrival_time_delays_days={"A": 0.0, "B": 7.4, "C": 2.1, "D": 11.8},
    seed=1001,
)
microlensing_only = system.light_curves(
    duration_days=3650,
    map_cadence_days=25,
    include_labels=True,
)

variable_system = system.with_source(source.with_driving_signal(driver))
microlensing_and_variability = variable_system.light_curves(
    duration_days=3650,
    map_cadence_days=25,
    source_cadence_days=1,
    include_labels=True,
)
maps = system.magnification_maps()  # one independent static map per image
```

Each image receives an independent stellar realization. Shared numerical
settings may be replaced by per-image mappings when necessary. The high-level
interface applies cosmological time delays only to intrinsic source evolution
while keeping lens motion in observer time. Global macro-model results can be
passed directly to `MultiImageSystem.from_macroimage_solutions`.
The same object provides `dynamic_maps`, `caustics`, `labeled_caustics`,
`multirate_light_curves`, and microlensing-weighted `transfer_functions`.
See
[`docs/multi_image.md`](docs/multi_image.md) and
[`examples/multi_image_light_curves.py`](examples/multi_image_light_curves.py).
The resolved-quasar workflow additionally supports optional EPL+shear macro
image/delay solving, sparse dynamic maps with fine-cadence source evolution,
microlensing-weighted transfer functions, and Rubin OpSim sampling. See
[`examples/multirate_lensed_quasar.py`](examples/multirate_lensed_quasar.py).
The survey notebook reads a local OpSim SQLite database from
`MICROCAUSTICS_LSST_OPSIM`. Rubin's
[`rubin_sim` data guide](https://rubin-sim.lsst.io/data-download.html) describes
the baseline-database download. A deterministic illustrative cadence is used
when that variable is not set.
Instrument-independent macro-image scenes can use `caustics.LensSource`. See
[`docs/macro_image_rendering.md`](docs/macro_image_rendering.md).

The validated relativistic-source layer is documented in
[`docs/relativistic_sources.md`](docs/relativistic_sources.md). Primary-image
Kerr light bending, capture, circular-disk frequency shifts, emission azimuth,
and observer-frame relative delays are available. The axial Kerr lamppost
tracer provides conservative illumination and lamp-to-disk delays for the
nonlinear thermal-reprocessing and transfer-function source layer.

Critical curves, caustics, parity labels, signed winding numbers, and distance
queries are documented in [`docs/caustics.md`](docs/caustics.md). Full label
maps are optional diagnostics. Center-only queries do not materialize them.

## Citation

If `microcaustics` contributes to published work, please cite the accompanying
methods paper rather than the GitHub repository. The DOI and publication
details will be added after publication.

Questions, bug reports, and collaboration inquiries may be sent to Joshua
Fagin at [faginjoshua@gmail.com](mailto:faginjoshua@gmail.com) or opened as a
GitHub issue.

## Author

`microcaustics` was created and is maintained by
[Joshua Fagin](https://orcid.org/0000-0001-8723-6136). He can be reached at
[faginjoshua@gmail.com](mailto:faginjoshua@gmail.com).

Contributors include the following authors.

- [Connor Stone](https://orcid.org/0000-0002-9086-6398)
- [James Hung-Hsu Chan](https://orcid.org/0000-0001-8797-725X)
- [Sophia Miskiewicz](https://orcid.org/0000-0003-0631-9701)
- [Henry Best](https://orcid.org/0009-0009-6932-6379)
- [Matthew O'Dowd](https://orcid.org/0009-0000-4476-5003)
- [Laurence Perreault-Levasseur](https://orcid.org/0000-0003-3544-3939)
- [Yashar Hezaveh](https://orcid.org/0000-0002-8669-5733)

## Selected scientific references

The package builds on work in microlensing, relativistic accretion-disk
emission, continuum reverberation, and time-domain survey simulation. The
references below are grouped by the parts of the package they most directly
motivate.

### Microlensing and numerical methods

- Wambsganss, Paczyński, and Katz (1990),
  [*A Microlensing Model for QSO 2237+0305*](https://doi.org/10.1086/168546).
- Mediavilla et al. (2006),
  [*A Fast and Very Accurate Approach to the Computation of Microlensing Magnification Patterns Based on Inverse Polygon Mapping*](https://doi.org/10.1086/508796),
  and Mediavilla et al. (2011),
  [*New Developments on Inverse Polygon Mapping*](https://doi.org/10.1088/0004-637X/741/1/42).
- Meena, Arad, and Zitrin (2022),
  [*An Efficient Method for Simulating Light Curves of Cosmological Microlensing and Caustic Crossing Events*](https://doi.org/10.1093/mnras/stac1511).
- Zheng et al. (2022),
  [*An Improved GPU-based Ray-shooting Code for Gravitational Microlensing*](https://doi.org/10.3847/1538-4357/ac68ea).
- Jiménez-Vicente and Mediavilla (2022),
  [*Fast Multipole Method for Gravitational Lensing*](https://doi.org/10.3847/1538-4357/ac9e59).
- Weisenbach (2025),
  [*Efficient Generation of Microlensing Magnification Maps with GPUs*](https://doi.org/10.1093/mnras/staf994)
  and
  [*A GPU Code for Finding Microlensing Critical Curves and Caustics*](https://doi.org/10.1093/mnras/staf1202).
- Stone et al. (2024),
  [*Caustics: A Python Package for Accelerated Strong Gravitational Lensing Simulations*](https://doi.org/10.21105/joss.07081).

### Relativistic disks and flexible AGN source models

- Best et al. (2024),
  [*Resolving the Vicinity of Supermassive Black Holes with Gravitational Microlensing*](https://doi.org/10.1093/mnras/stae1182).
- Best et al. (2025),
  [*Amoeba: An AGN Model of Optical Emissions Beyond steady-state Accretion discs*](https://doi.org/10.1093/mnras/staf499).
- Chan et al. (2024),
  [*Reverberation Mapping of Lamp-post and Wind Structures in Accretion Thin Disks*](https://doi.org/10.48550/arXiv.2409.15669).
- Page and Thorne (1974),
  [*Disk-Accretion onto a Black Hole: Time-Averaged Structure of Accretion Disk*](https://doi.org/10.1086/152990).
- Bursa (2018),
  [*SIM5: Library for Ray-tracing and Radiation Transport in General Relativity*](https://ui.adsabs.harvard.edu/abs/2018ascl.soft11011B).
- Gralla and Lupsasca (2020),
  [*Null geodesics of the Kerr exterior*](https://doi.org/10.1103/PhysRevD.101.044032).
### Dynamic light curves, variability, and LSST-like workflows

- Fagin et al. (2025),
  [*Predicting High-magnification Events in Microlensed Quasars in the Era of LSST using Recurrent Neural Networks*](https://doi.org/10.3847/1538-4357/adaebb).
- Fagin et al. (2025),
  [*Joint Modeling of Quasar Variability and Accretion Disk Reprocessing using Latent Stochastic Differential Equations*](https://doi.org/10.3847/1538-4357/addabc).
- Fagin et al. (2024),
  [*Latent Stochastic Differential Equations for Modeling Quasar Variability and Inferring Black Hole Properties*](https://doi.org/10.3847/1538-4357/ad2988).
