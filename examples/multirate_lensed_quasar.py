"""Sparse dynamic maps, fine source evolution, and survey sampling."""

from __future__ import annotations

import numpy as np
import torch

import microcaustics as mc

distances = mc.LensingDistances(1.0e25, 2.0e25, 1.0e25)
kinematics = mc.SkyProjectedKinematics(
    ra_deg=340.126125,
    dec_deg=3.358611,
    stellar_dispersion_km_s=170.0,
    peculiar_velocity_dispersion_km_s=235.0,
    include_cmb_dipole=True,
    lens_redshift=0.08,
    source_redshift=0.18,
    seed=0,
)
bulk_x, bulk_y = kinematics.mean_velocity_uas_per_day(distances)
dispersion = kinematics.component_dispersion_uas_per_day(distances)


def stars(offset: float) -> mc.PointMassField:
    """Return a small moving field for one illustrative macroimage."""

    return mc.PointMassField(
        torch.tensor([-0.4 + offset, 0.45 + offset]),
        torch.tensor([0.3, -0.2]),
        mass_solar=torch.tensor([0.001791, 0.002579]),
        velocity_x_uas_per_day=bulk_x + dispersion * torch.tensor([0.7, -0.6]),
        velocity_y_uas_per_day=bulk_y + dispersion * torch.tensor([-0.4, 0.5]),
    )


geometry = mc.SourceGeometry(
    shape=(9, 9),
    pixel_scale_m=(1.0e11, 1.0e11),
    wavelengths_angstrom=(4800.0, 6200.0, 7500.0),
    band_names=("g", "r", "i"),
)
base_source = mc.GaussianSource(geometry, sigma_m=2.0e11)
driver = mc.broken_power_law_driving_signal(
    torch.arange(-20.0, 61.0),
    break_timescale_days=200.0,
    alpha_L=1.0,
    alpha_R=3.0,
    standard_deviation=0.1,
    seed=0,
    extrapolation="hold",
)
source = mc.ModulatedSource(base_source, driver)
method = mc.IPMConfig(
    rays=4096,
    scout_ratio=2,
    refinement=1,
    virtual_refinement=2,
    far_field_approx=mc.FarFieldApproxConfig(enabled=False),
)
system = mc.MultiImageSystem(
    images={
        "A": mc.MacroLens(convergence=0.2, shear=0.12),
        "B": mc.MacroLens(convergence=0.2, shear=0.18),
    },
    distances=distances,
    source=source,
    stars={"A": stars(0.0), "B": stars(0.1)},
    arrival_time_delays_days={"A": 0.0, "B": 7.4},
    methods=method,
    runtime=mc.RuntimeConfig(device="cpu", backend=mc.Backend.TORCH_EAGER),
    duration_days=50.0,
)

curves = system.light_curves(
    duration_days=50,
    map_cadence_days=25,
    source_cadence_days=1,
)
cadence = mc.SurveyCadence(
    time_days=np.array([3.0, 12.0, 26.0, 41.0, 49.0]),
    band_names=("g", "r", "i", "g", "r"),
    five_sigma_depth=np.array([24.7, 24.4, 24.0, 24.6, 24.2]),
)
observations = mc.observe_multi_image_light_curves(
    curves,
    cadence,
    seed=0,
)
print("truth", curves.flux_tensor().shape)
print("observations", observations.magnitude.shape)
print("arrival delays [days]", curves.arrival_time_delays_days)
