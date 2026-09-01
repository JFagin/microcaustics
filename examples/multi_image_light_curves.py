"""Generate coherent variable light curves for two resolved macroimages."""

from __future__ import annotations

import torch

import microcaustics as mc

runtime = mc.RuntimeConfig(device="cpu", backend=mc.Backend.TORCH_EAGER)
distances = mc.LensingDistances(1.0e25, 2.0e25, 1.0e25)
source_shape = (64, 64)
source_fov_uas = (2.0, 2.0)
pixel_scale_m = distances.uas_to_source_length(
    (source_fov_uas[0] / source_shape[0], source_fov_uas[1] / source_shape[1]),
    dtype=torch.float64,
)
geometry = mc.SourceGeometry(
    shape=source_shape,
    pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
    wavelengths_angstrom=(4800.0, 7500.0),
    band_names=("blue", "red"),
)
base = mc.GaussianSource(geometry, sigma_m=2.0e11)
driver = mc.CallableDrivingSignal(
    lambda times: 1.0 + 0.1 * torch.sin(times / 20.0),
    name="sinusoid",
)
source = mc.ModulatedSource(base, driver)


def stars(offset: float) -> mc.PointMassField:
    """Return a small illustrative star field translated per macroimage."""

    return mc.PointMassField(
        x_uas=torch.tensor([-0.5 + offset, 0.45 + offset]),
        y_uas=torch.tensor([0.3, -0.25]),
        mass_solar=torch.tensor([0.002579, 0.002037]),
        velocity_x_uas_per_day=torch.tensor([0.001, -0.0007]),
        velocity_y_uas_per_day=torch.tensor([-0.0004, 0.0008]),
    )


method = mc.IPMConfig(
    rays=4_096,
    scout_ratio=2,
    refinement=1,
    virtual_refinement=2,
    far_field_approx=mc.FarFieldApproxConfig(enabled=False),
)
system = mc.MultiImageSystem(
    images={
        "A": mc.MacroLens(convergence=0.25, shear=0.12),
        "B": mc.MacroLens(convergence=0.25, shear=0.18),
    },
    distances=distances,
    source=source,
    stars={"A": stars(0.0), "B": stars(0.15)},
    lens_region=mc.PlaneRegion((4.0, 4.0)),
    integration_domain="full",
    arrival_time_delays_days={"A": 0.0, "B": 8.0},
    methods=method,
    schedules=mc.DynamicConfig(temporal_batch_size=2),
    runtime=runtime,
    caustic_grid_shape=64,
)

result = system.light_curves(torch.arange(0.0, 51.0, 5.0).tolist())
print(result.image_names)
print(result.flux_tensor().shape)
print(result.arrival_time_delays_days)
