"""Generate a small source-independent IRS magnification map on the CPU."""

import torch

import microcaustics as mc

macro = mc.MacroLens(convergence=0.0, shear=0.0)
point_masses = mc.PointMassField(
    x_uas=torch.tensor([0.0]),
    y_uas=torch.tensor([0.0]),
    einstein_radius_uas=torch.tensor([0.3]),
)
system = mc.MicrolensingSystem(
    macro=macro,
    distances=mc.LensingDistances(8.0e24, 1.6e25, 9.0e24),
    stars=point_masses,
    source_grid=mc.PlaneGrid(shape=(128, 128), field_of_view_uas=(2.0, 2.0)),
    runtime=mc.RuntimeConfig(
        device="cpu",
        backend="torch-eager",
        dtype="float32",
    ),
)

magnification = system.magnification_map(
    method=mc.IRSConfig(
        rays=1_000_000,
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    ),
)

print(magnification.values.shape)
print(magnification.metadata)
print(f"steady-state time: {magnification.timing.steady_seconds:.3f} s")
