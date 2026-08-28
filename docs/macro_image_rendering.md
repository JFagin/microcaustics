# General macro-image rendering

Macro-image rendering is a thin multiband layer over the
[`caustics`](https://github.com/Ciela-Institute/caustics) package. Microcaustics
does not implement a second strong-lens image simulator. `caustics.LensSource`
performs the ray tracing, source and lens-light evaluation, subpixel
quadrature, upsampling, and PSF convolution.

Install the optional macro-lensing dependencies:

```bash
python -m pip install "microcaustics[macro]"
```

Any caustics lens and light profiles can be used. The renderer is not tied to
an HST configuration, a particular mass profile, or a fixed set of bands:

```python
import caustics
import microcaustics as mc

cosmology = caustics.FlatLambdaCDM()
lens = caustics.SIE(
    cosmology=cosmology,
    z_l=0.5,
    z_s=1.5,
    x0=0.0,
    y0=0.0,
    q=0.72,
    phi=0.4,
    Rein=0.9,
)

host = caustics.Sersic(
    x0=0.04,
    y0=-0.02,
    q=0.65,
    phi=0.7,
    n=1.2,
    Re=0.18,
    Ie=1.0,
)
compact = caustics.StarSource(
    x0=0.04,
    y0=-0.02,
    theta_s=0.003,
    Ie=20.0,
    gamma=0.5,
)
source = caustics.LightStack((host, compact))

renderer = mc.CausticsMacroImageRenderer(
    lens,
    mc.ImagePlaneGrid((128, 160), (3.2, 4.0)),
    band_names=("blue", "red"),
    wavelengths_angstrom=(5000.0, 7500.0),
    upsample_factor=2,
    quadrature_level=2,
)
image = renderer.render(
    0.0,
    # A shared caustics source is allowed. Mappings provide band-specific ones.
    sources=source,
)
```

Use any caustics `Source`, including `Sersic`, `StarSource`, `Pixelated`, or a
`LightStack` of multiple components. A single model may be shared between
bands, or sequences/mappings can define independent morphologies and spectra.
The lens may be an analytic caustics lens, a composite `SinglePlane` model, or
a multiplane model supported by `LensSource`. Native caustics parameter values
can be supplied through `parameters` or `parameters_by_band`.

`render(time_days=...)` records the epoch and passes it to an optional
observation model, but it does not implicitly evolve arbitrary native caustics
sources. Supply epoch-specific caustics parameters, or use
`caustics_pixelated_sources` at the same epoch for a package source model.

## Package source models

`caustics_pixelated_sources` converts any package `PixelatedSource` at one
epoch into ordinary caustics `Pixelated` models:

```python
band_sources = mc.caustics_pixelated_sources(
    evolving_source,
    time_days=42.0,
    source_angular_diameter_distance_m=D_s,
    image_grid=renderer.grid,
    source_center_arcsec=(-0.02, 0.04),
)
image = renderer.render(42.0, sources=band_sources)
```

This supports thin disks, transferred GR disks, expanding photospheres, and
arbitrary user sources without teaching the macro renderer their physics.
Conversion from physical source-plane surface brightness to flux per rendered
pixel is enabled by default and follows surface-brightness conservation.

## Instrument independence

PSFs are passed directly to caustics and may be shared, per-band arrays, or a
band mapping. `upsample_factor`, `quadrature_level`, `psf_mode`, and
`chunk_size` retain their caustics meanings. When `upsample_factor > 1`, the
PSF array must use that same supersampled pixel scale.

`gaussian_psf_kernel` constructs a normalized kernel from an angular FWHM,
final image-pixel scale, and renderer oversampling factor. The lightweight
`PeakScaledPoissonReadNoise` model is convenient for normalized, high-S/N
demonstrations. It keeps the assumed peak electron count and read-noise
fraction explicit. Calibrated applications should continue to supply their
own instrument model.

Detector effects are deliberately outside the lens renderer. An optional
`observation_model` callback receives the noiseless multiband result and full
grid/band context, and may return an observed image, or an image plus variance
and metadata. This accommodates arbitrary exposure models, backgrounds,
Poisson/read noise, saturation, cosmic rays, or survey-specific pipelines
without naming an instrument in the physical renderer.
