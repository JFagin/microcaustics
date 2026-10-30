# Microimages and centroid motion

The optional microimage output renders
the stellar microimages around **one macroimage**, not the galaxy-scale quad.

## Request images with the light curve

Pass a callback to an existing system or simulation. No configuration object
is needed for the default output.

```python
centroids = []

def save_frame(index, frame):
    centroids.append(frame.centroid_uas.detach().cpu())

curve = system.light_curve(
    times_days=[0, 25, 50],
    microimage_observer=save_frame,
)
```

For a low-level `simulation.light_curve`, supply its usual lens region,
source grid, times, source, distances and IPM method, then add the same callback.
The callback receives every evaluated epoch and every light-curve band by default.
`index` is the original epoch index. The returned `LightCurve` is unchanged.
Without a callback or configuration, the ordinary calculation is unchanged.

Select fewer images without changing the light curve itself.

```python
from microcaustics import MicroimageConfig

curve = system.light_curve(
    times_days=[0, 25, 50],
    microimages=MicroimageConfig(times_days=[0, 50], bands=["i"]),
    microimage_observer=save_frame,
)
```

Omit either selection to use all epochs or all bands. Times must match evaluated
epochs at the precision of the light-curve time array. Decimal times such as
0.1 days are supported. Unavailable times and bands raise an error, rather than
being snapped to nearby epochs or simulated separately. A configuration without a callback also raises
an error. Callback exceptions propagate to the caller.

## Shared calculation and memory

IPM already computes how each mapped triangle overlaps the source pixels.
The renderer weights those overlaps by source brightness and assigns the flux
to the original lens cell. It reuses the scouting, traced nodes and overlaps.
The light curve and microimages use the same source brightness evaluation.
Only selected bands are rendered. Unselected epochs still produce ordinary
maps and photometry but skip the image weighting.

Arrays remain on the calculation device. Saving, plotting or transferring a
frame is the callback's responsibility. Do not retain every full-resolution
frame unless you need them. A callback that copies frames to the CPU will add
transfer time. `frame.rebin(factor)` combines cells on the device for compact
display, preserving total flux.

An original frame may share storage with other frames in its temporal batch.
Rebin it, or clone the tensors you want to retain, to release that batch sooner.

Temporal batching still applies. Sparse epoch selections currently rasterize
mixed batches one epoch at a time, while retaining shared ray tracing. All-epoch
output retains the fused temporal rasterizer. Start with a modest explicit
`temporal_batch_size` when rendering many wavelengths. Automatic tuning does
not yet estimate all additional image buffers.

## Units and interpretation

- `flux_jy` is flux density integrated over each selected lens cell, with shape
  `[cell, wavelength]`. Wavelengths follow `wavelengths_angstrom`.
- `cell_indices` indexes the flattened lens-plane `grid`. Unselected cells are
  absent. `to_dense()` explicitly allocates the entire lens grid.
- `total_flux_jy` sums the resolved image. The uniform scout normalization
  correction is reported separately as `normalization_correction_flux_jy`.
  Their sum should match the corresponding light-curve flux to numerical
  precision. The correction is not assigned a position or included in centroids.
- `centroid_uas` gives `[wavelength, (x, y)]` in local lens-plane microarcseconds.
  It uses cell centers, not a point-image solver. Zero-flux bands return NaN.
  Use the original frame, not a rebinned display frame, for astrometry.
- `time_days` is the lens epoch. Existing source response delays are retained.
  Additional microimage propagation delays are not included.

Flux agreement does not by itself establish centroid or morphology convergence.
Increase the ray density and check the scouting region for precision astrometry.
These images are noiseless and contain no telescope PSF or detector model.

```python
from microcaustics.plotting import plot_microimages

fig, ax = plot_microimages(frame, band=0, bounds_uas=(-140, 140, -140, 140), scale_bar_uas=50)
```

The colorbar gives cell-averaged intensity in microJy per square microarcsecond.
Image axes have no ticks and the scale label is below the bar.
The helper crops before allocating the image and preserves native cell resolution.
Without bounds it crops to the nonzero cells. Display binning occurs only when
`max_pixels` is explicitly supplied. `frame.crop(bounds_uas)` also returns a sparse
crop for animations. Calculate full-image fluxes and centroids before cropping.

## Current scope

The callback interface supports IPM with a centered stationary source on its
native map grid, exact source evolution, and equal map/photometry cadences.
This includes an evolving disk brightness pattern with its existing response
delays. It does not require the stellar field to be static.

Separate source/map cadences, moving source trajectories, linear/quadratic/cubic
response approximations, bandpass integration, band batching, a simultaneous
mean-driver comparison and integrated caustic labeling are not yet supported
by this interface. They raise an error rather than silently changing the source
or omitting images. Caustic labels can be calculated separately. Independent
system batching with ragged cell selections remains unsupported.

The notebook `examples/notebooks/workflows/05_microimages_and_astrometry.ipynb` generates six-band Q2237 B photometry, i-band images, centroid motion,
caustic-crossing labels and its GIF from scratch. It also checks flux closure.
No pre-existing animation or simulation output is loaded.
