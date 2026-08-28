"""Evaluate a user-configured expanding supernova photosphere."""

import math

import torch

import microcaustics as mc

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
    wavelengths_angstrom=(4_827.0, 6_223.0, 7_546.0),
    band_names=("g", "r", "i"),
    maximum_observer_time_days=180.0,
    evolution=evolution,
    resolution=256,
    source_fov_margin=1.08,
)

observer_times = torch.arange(0.0, 181.0)
brightness = source.brightness(observer_times)
pixel_area_m2 = math.prod(source.geometry.pixel_scale_m)
unlensed_flux_jy = brightness.sum(dim=(1, 2)) * pixel_area_m2

print(brightness.shape)
print(source.metadata())
print(unlensed_flux_jy.max(dim=0).values)
