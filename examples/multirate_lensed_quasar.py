"""Sparse dynamic maps, fine source evolution, and survey sampling."""

from __future__ import annotations

import numpy as np
import torch

import microcaustics as mc


def stars(offset: float) -> mc.PointMassField:
    """Return a small moving field for one illustrative macroimage."""

    return mc.PointMassField(
            torch.tensor([-0.4 + offset, 0.45 + offset]),
            torch.tensor([0.3, -0.2]),
            torch.tensor([0.15, 0.18]),
            velocity_x_uas_per_day=torch.tensor([0.001, -0.0008]),
            velocity_y_uas_per_day=torch.tensor([-0.0004, 0.0006]),
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
    low_frequency_slope=1.0,
    high_frequency_slope=3.0,
    standard_deviation=0.1,
    seed=8,
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
system = mc.MultiImageMicrolensingSystem(
    images={
        "A": mc.MacroLens(convergence=0.2, shear=0.12),
        "B": mc.MacroLens(convergence=0.2, shear=0.18),
    },
    distances=mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
    source=source,
    stars={"A": stars(0.0), "B": stars(0.1)},
    arrival_time_delays_days={"A": 0.0, "B": 7.4},
    methods=method,
    runtime=mc.RuntimeConfig(device="cpu", backend=mc.Backend.TORCH_EAGER),
    duration_days=50.0,
)

curves = system.multirate_light_curves(
    map_times_days=[0.0, 25.0, 50.0],
    flux_times_days=torch.arange(0.0, 51.0).tolist(),
)
cadence = mc.SurveyCadence(
    time_days=np.array([3.0, 12.0, 26.0, 41.0, 49.0]),
    band_names=("g", "r", "i", "g", "r"),
    five_sigma_depth=np.array([24.7, 24.4, 24.0, 24.6, 24.2]),
)
observations = mc.observe_multi_image_light_curves(
    curves,
    cadence,
    zero_point_flux=1.0e23,
    seed=4,
)
print("truth", curves.flux_tensor().shape)
print("observations", observations.magnitude.shape)
print("arrival delays [days]", curves.arrival_time_delays_days)
