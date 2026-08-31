"""Generate moving-star maps and a multiband finite-source light curve."""

import torch

import microcaustics as mc

distances = mc.LensingDistances(
    lens_m=8.0e24,
    source_m=1.6e25,
    lens_to_source_m=9.0e24,
)
field = mc.PointMassField(
    x_uas=torch.tensor([-0.4, 0.5]),
    y_uas=torch.tensor([0.2, -0.3]),
    einstein_radius_uas=torch.tensor([0.18, 0.22]),
    velocity_x_uas_per_day=torch.tensor([2.0e-4, -1.0e-4]),
    velocity_y_uas_per_day=torch.tensor([1.0e-4, 1.5e-4]),
)
times = torch.tensor([0.0, 25.0, 50.0])
source = mc.GaussianSource(
    mc.SourceGeometry(
        shape=(32, 32),
        pixel_scale_m=(1.0e12, 1.0e12),
        wavelengths_angstrom=(4800.0, 7500.0),
        band_names=("blue", "red"),
    ),
    sigma_m=(3.0e12, 4.5e12),
)
system = mc.MicrolensingSystem(
    macro=mc.MacroLens(convergence=0.15, shear=0.08),
    distances=distances,
    stars=field,
    source=source,
    # The larger map includes the imposed source trajectory.
    source_grid=mc.PlaneGrid((64, 64), (2.0, 2.0)),
    duration_days=float(times[-1]),
    runtime=mc.RuntimeConfig(device="auto", backend="auto"),
)
ipm = mc.IPMConfig(
    rays=65_536,
    scout_ratio=2,
    refinement=2,
    virtual_refinement=4,
    tiled=True,
    cell_chunk_size=16_384,
    # Enable this optional frame-zero k=1 -> k=2 normalization correction
    # for a conservative high-accuracy production behavior.
    dual_scout_scalar_correction=True,
    far_field_approx=mc.FarFieldApproxConfig(
        cells_per_axis=4,
        nodes_per_cell_axis=8,
        taylor_order=4,
        center_translation_order=10,
    ),
)
light_curve = system.light_curve(
    times,
    method=ipm,
    trajectory=mc.LinearTrajectory(velocity_uas_per_day=(5.0e-4, 0.0)),
    schedule=mc.DynamicConfig(
        temporal_batch_size=3,
        fused_temporal_ipm=True,
        scout_refresh_frames=3,
    ),
)

print("magnification by band:")
print(light_curve.flux / light_curve.unlensed_flux)
print(f"streaming light-curve time: {light_curve.timing.steady_seconds:.4f} s")
