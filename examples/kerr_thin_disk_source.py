"""Generate a standalone full-Kerr continuum-disk image."""

import torch

import microcaustics as mc

mass_solar = 1.0e9
redshift = 1.7
angular_diameter_distance_m = 5.0e25
gravitational_radius_m = float(mc.gravitational_radius_m(mass_solar))
shape = (256, 256)
half_size_rg = 80.0
screen = mc.ObserverScreen.uniform(
    shape,
    half_size_rg,
    gravitational_radius_m=gravitational_radius_m,
    observer_distance_m=angular_diameter_distance_m,
    dtype=torch.float64,
)
trace = mc.trace_primary_equatorial(
    screen,
    spin=0.7,
    inclination_deg=45.0,
    disk_outer_rg=75.0,
)
pixel_scale_m = 2.0 * half_size_rg * gravitational_radius_m / shape[0]
geometry = mc.SourceGeometry(
    shape,
    (pixel_scale_m, pixel_scale_m),
    (4770.0, 7625.0),
    ("g", "i"),
)
source = mc.TransferredThinDiskSource(
    geometry,
    trace.transfer,
    black_hole_mass_solar=mass_solar,
    eddington_ratio=0.1,
    spin=0.7,
    source_redshift=redshift,
)
image = source.brightness(0.0, dtype=torch.float64)[0]
flux_jy = image.sum(dim=(0, 1)) * pixel_scale_m**2
print("shape:", tuple(image.shape))
print("hit fraction:", float(trace.transfer.hit.double().mean()))
print("repaired pixels:", trace.repaired_pixels)
print("flux [Jy]:", dict(zip(geometry.band_names, flux_jy.tolist(), strict=True)))
