"""Apply a generic driving signal across a relativistic delay map."""

import math

import torch

import microcaustics as mc

mass_solar = 1.0e9
redshift = 1.7
angular_diameter_distance_m = 5.0e25
gravitational_radius_m = float(mc.gravitational_radius_m(mass_solar))
shape = (128, 128)
half_size_rg = 70.0
screen = mc.ObserverScreen.uniform(
    shape,
    half_size_rg,
    gravitational_radius_m=gravitational_radius_m,
    observer_distance_m=angular_diameter_distance_m,
    dtype=torch.float64,
)
primary = mc.trace_primary_equatorial(
    screen,
    spin=0.7,
    inclination_deg=45.0,
    disk_outer_rg=65.0,
)
coordinates = mc.add_observer_coordinates(
    primary,
    screen,
    black_hole_mass_solar=mass_solar,
    spin=0.7,
    inclination_deg=45.0,
    source_redshift=redshift,
    coordinate_dtype=torch.float64,
)
pixel_scale_m = 2.0 * half_size_rg * gravitational_radius_m / shape[0]
geometry = mc.SourceGeometry(
    shape,
    (pixel_scale_m, pixel_scale_m),
    (4770.0,),
    ("g",),
)
static_disk = mc.TransferredThinDiskSource(
    geometry,
    coordinates.transfer,
    mass_solar,
    0.1,
    0.7,
    redshift,
)
signal = mc.CallableDrivingSignal(
    lambda time: 1.0 + 0.1 * torch.sin(2.0 * math.pi * time / 100.0),
    name="sinusoidal_example",
)
variable_disk = mc.DelayedModulatedSource.from_observer_transfer(
    static_disk,
    signal,
    coordinates.transfer,
)
images = variable_disk.brightness([0.0, 10.0], dtype=torch.float64)
delay = coordinates.transfer.relative_delay_days[coordinates.transfer.hit]
print("brightness shape:", tuple(images.shape))
print("delay range [days]:", float(delay.min()), float(delay.max()))
