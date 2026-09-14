"""Generate a small source-independent IRS magnification map on the CPU."""

import torch

import microcaustics as mc

macro = mc.MacroLens(convergence=0.0, shear=0.0)
point_masses = mc.PointMassField(
    x_uas=torch.tensor([0.0]),
    y_uas=torch.tensor([0.0]),
    mass_solar=torch.tensor([0.005094]),
)
system = mc.MicrolensingSystem(
    macro=macro,
    distances=mc.LensingDistances(8.0e24, 1.6e25, 9.0e24),
    stars=point_masses,
    runtime=mc.RuntimeConfig(
        device="cpu",
        backend="torch-eager",
        dtype="float32",
    ),
)

magnification = system.magnification_map(
    map_width_uas=2.0,
    map_pixels=128,
    method=mc.IRSConfig(
        rays=1_000_000,
        sampling="cartesian",  # use "random" with a seed for Monte Carlo IRS
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    ),
)

print(magnification.values.shape)
print(magnification.metadata)
# Timing is opt-in. See docs/timing.md for first-call and warmed measurements.
