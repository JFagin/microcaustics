# Plotting and notebooks

Install the optional plotting interface with:

```bash
pip install "microcaustics[plot]"
```

The functions in `microcaustics.plotting` accept public result objects and
return `(figure, axes)`. They do not call `show()` and do not rerun a
microlensing calculation, which makes them safe to compose into a larger
figure or use in a batch job.

```python
import microcaustics.plotting as mcp
from matplotlib import pyplot as plt

with plt.style.context(mcp.publication_style()):
    figure, axes = mcp.plot_magnification_map(
        magnification_map,
        log10=True,
        caustics=caustic_field,
        scale_bar_uas=1.0,
        show_axes=False,
    )
figure.savefig("magnification_map.pdf", bbox_inches="tight")
```

Maps are drawn with their physical source- or lens-plane coordinates rather
than array indices. Helpers are also provided for map residuals, label and
distance maps, anchor/gauge layouts, sources, transfer functions, rendered
macroimages, survey observations, and timing breakdowns.

The standard magnification, label, distance, and winding diagnostics can be
drawn together without reproducing notebook-specific layout code:

```python
figure, axes = mcp.plot_caustic_diagnostics(
    magnification_map,
    binary_label_map,
    distance_to_caustic_map,
    winding_number_map,
    caustics=caustic_field,
    scale_bar_uas=1.0,
)
```

The binary and winding panels use discrete integer colorbars. All four panels
share the same physical orientation and field extent.

Image panels can follow the compact paper convention directly:
`show_axes=False` removes redundant ticks and labels while retaining a thin
panel frame, `add_scale_bar` adds a
physical source/lens/image-plane scale, and `panel_colorbar` creates a colorbar
that is exactly as tall as its panel. `plot_label_map` uses a discrete norm and
integer ticks, which is important for binary and signed-winding products.

For animated maps, use `save_fixed_palette_gif` after rendering every frame
with one fixed Matplotlib `vmin`/`vmax`. The helper quantizes all RGB frames
against one global indexed palette. This prevents an otherwise fixed
scientific colorbar from appearing to change because GIF encoders choose a
different palette per frame. Static pixels are encoded once from frame zero
with smooth error-diffusion dithering and their exact palette indices are
reused in every frame. Only pixels that are identical throughout the input
are frozen. This keeps colorbars smooth and stationary without hiding changes
in the data. Set `freeze_static=False` to disable this treatment.

For moving pixels, the default ordered dithering stays fixed to pixel
coordinates. For a single-colormap animation, pass
`palette_rgb` with the colormap and annotation colors and set `dither=False`.
This preserves the gradient and thin caustic lines that automatic palette
sampling might miss. GIFs remain limited to 256 colors.

For a self-contained notebook preview, use `display(mcp.gif_preview(gif_path))`.
This embeds a continuous excerpt below GitHub's observed display-size cutoff.
Only the end of the excerpt is shortened when necessary. Resolution, colors,
frame cadence, and playback timing are unchanged. Set `start_frame` and
`stop_frame` to choose an interval. The full GIF is left intact and can be
linked separately. This helper also requires IPython.

For a variable source, `animate_standardized_source_bands` uses the temporal
statistics returned by `standardize_source_over_time`. Pass `bands="i"` for a
single-band animation, or omit `bands` to show them all. Band selection does
not change the source calculation or its normalization. The animation keeps
the full source colormap and neutral annotation colors. Moving disk pixels
are undithered, while the static colorbar is encoded only once.

Use `mean_source_isophotes` to stream a variable source over time and measure
one enclosed-flux boundary per band without retaining the full time cube:

```python
summary = mcp.mean_source_isophotes(
    source,
    daily_times,
    source_plane_grid,
    fraction=0.95,
    batch_size=8,
)
for image, level, radius in zip(
    summary.images,
    summary.levels,
    summary.area_equivalent_radii_uas,
    strict=True,
):
    axis.contour(summary.x_uas, summary.y_uas, image, levels=[level])
    print(f"area-equivalent R95 = {radius:.3f} microarcsec")
```

The contours retain the actual projected source shape. The reported radius is
only the radius of a circle with the same enclosed pixel area.

When several source models share one dynamic map sequence,
`MicrolensingSystem.light_curves` can retain selected evaluated maps directly:

```python
variable, baseline = system.light_curves(
    map_times,
    requests=(variable_request, baseline_request),
    flux_times_days=daily_times,
    keep_maps_at_days=map_times,
)
maps = variable.maps
assert maps[0] is baseline.maps[0]
```

Selected maps can also be retained when using `map_observer`. This does not trigger another map
calculation.

## Timing accelerated calculations

GPU work is asynchronous and compiled paths have two different timings worth
reporting. Use the public benchmark helper with the selected runtime's
synchronization function:

```python
import microcaustics as mc
import microcaustics.plotting as mcp

benchmark = mc.benchmark_callable(
    make_map,
    warmup=2,
    repeats=5,
    synchronize=simulation.runtime.synchronize,
)
mcp.print_benchmark(benchmark, label="IPM map")
```

The first call may include Triton or PyTorch compilation, allocator setup, and
cache population. `print_benchmark` therefore calls it the *first call in this
process*, not pure compile time. The warmed median is the appropriate number
for repeated production calculations. It also prints the operating system,
PyTorch and CUDA versions, GPU model, and CUDA compute capability.

Install and launch the complete notebook collection with:

```bash
pip install "microcaustics[notebooks,science,macro]"
jupyter lab examples/notebooks
```

The complete collection uses `science` for cosmology and numerical reference
calculations. Its strong-lens rendering examples use `macro`, which requires
Python 3.11 or newer.

The Weisenbach-IPM and SIM5 notebooks are optional visual validation guides.
They read reference results generated by those codes and skip
cleanly when the corresponding paths are not configured.

See the [notebook guide](https://github.com/JFagin/microcaustics/blob/main/examples/notebooks/README.md)
for the full collection of tutorials.

The relativistic disk layer also works independently of microlensing.
`source_models/00_relativistic_disks_and_reverberation.ipynb` constructs full-Kerr lamppost
transfer functions and daily multiband continuum-reverberation light curves
without creating a magnification map.

Response plots show bin mass divided by bin width without filtering. Gaussian
response smoothing is no longer supported. Use the numerical
[CDF/PCHIP reconstruction](relativistic_sources.md#transfer-function-delay-reconstruction)
when a different output delay grid is needed, and check the native-grid
resolution separately.

The dataset notebook `workflows/04_simulation_datasets.ipynb` demonstrates
`plot_light_curve_dataset` and `plot_labeled_map_gallery`, which preserve the
separate photometry and caustic-label time axes. See the
[example-script guide](https://github.com/JFagin/microcaustics/blob/main/examples/README.md)
for multi-GPU dataset generation.
`validation/03_single_point_lens_validation.ipynb` compares the IPM map, critical
curve, and caustic directly with the analytic point-lens solution.

The survey notebooks accept a Rubin OpSim SQLite database through
`MICROCAUSTICS_LSST_OPSIM`. Rubin documents the current baseline-data download
workflow in the
[rubin_sim data guide](https://rubin-sim.lsst.io/data-download.html). Once the
database is local, `RubinOpSimCadenceIndex` provides fast repeated WFD, DDF,
and sky-position queries without requiring `rubin_sim` at simulation time.
See [`observations.md`](observations.md).
