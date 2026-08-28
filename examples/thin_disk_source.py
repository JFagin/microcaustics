"""Evaluate a physical multiband thin disk independently of microlensing."""

import torch

import microcaustics as mc

distances = mc.LensingDistances(8.0e24, 1.6e25, 9.0e24)
geometry = mc.SourceGeometry(
    shape=(256, 256),
    pixel_scale_m=(2.0e12, 2.0e12),
    wavelengths_angstrom=(3671.0, 4827.0, 6223.0, 7546.0),
    band_names=("u", "g", "r", "i"),
)
disk = mc.ThinDiskSource.from_lensing_distances(
    geometry,
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    distances=distances,
    source_redshift=1.7,
    spin=0.7,
    inclination_deg=30.0,
    position_angle_deg=15.0,
    relativity="approximate",
)
brightness = disk.brightness(torch.tensor([0.0]))
pixel_area_m2 = geometry.pixel_scale_m[0] * geometry.pixel_scale_m[1]
unlensed_flux = brightness[0].sum(dim=(0, 1)) * pixel_area_m2

print(brightness.shape)
print(dict(zip(geometry.band_names, unlensed_flux.tolist(), strict=True)))
