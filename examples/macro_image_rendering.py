"""Render a multiband strong-lens scene with native caustics components."""

import caustics
import torch

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


def source(intensity: float, name: str):
    host = caustics.Sersic(
        x0=0.04,
        y0=-0.02,
        q=0.65,
        phi=0.7,
        n=1.2,
        Re=0.18,
        Ie=intensity,
        name=f"{name}_host",
    )
    compact = caustics.StarSource(
        x0=0.04,
        y0=-0.02,
        theta_s=0.004,
        Ie=15.0 * intensity,
        gamma=0.5,
        name=f"{name}_compact",
    )
    return caustics.LightStack((host, compact), name=f"{name}_source")


def lens_light(intensity: float, name: str):
    return caustics.Sersic(
        x0=0.0,
        y0=0.0,
        q=0.78,
        phi=0.4,
        n=4.0,
        Re=0.55,
        Ie=intensity,
        name=f"{name}_lens_light",
    )


grid = mc.ImagePlaneGrid((128, 160), (3.2, 4.0))
renderer = mc.CausticsMacroImageRenderer(
    lens,
    grid,
    band_names=("blue", "red"),
    wavelengths_angstrom=(5_000.0, 7_500.0),
    upsample_factor=2,
    psf_mode="conv2d",
)


def gaussian_psf(fwhm_arcsec: float) -> torch.Tensor:
    sampled_pixel_scale = (
        grid.pixel_scale_arcsec[0] / renderer.upsample_factor
    )
    sigma_pixels = fwhm_arcsec / sampled_pixel_scale / 2.35482
    axis = torch.arange(-8, 9, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma_pixels**2))


result = renderer.render(
    0.0,
    sources=(source(1.0, "blue"), source(0.8, "red")),
    lens_light=(lens_light(0.5, "blue"), lens_light(0.8, "red")),
    psf=(gaussian_psf(0.08), gaussian_psf(0.11)),
    retain_components=True,
)

print(result.values.shape)
print(result.metadata)
print(result.component_values.keys())
