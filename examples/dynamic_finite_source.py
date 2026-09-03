"""Generate moving-star maps and a multiband finite-source light curve."""

import torch

import microcaustics as mc

distances = mc.LensingDistances(
    lens_m=8.0e24,
    source_m=1.6e25,
    lens_to_source_m=9.0e24,
)
kinematics = mc.SkyProjectedKinematics(
    ra_deg=340.126125,
    dec_deg=3.358611,
    stellar_dispersion_km_s=170.0,
    peculiar_velocity_dispersion_km_s=235.0,
    include_cmb_dipole=True,
    lens_redshift=0.05,
    source_redshift=0.12,
    seed=0,
)
bulk_x, bulk_y = kinematics.mean_velocity_uas_per_day(distances)
dispersion = kinematics.component_dispersion_uas_per_day(distances)
field = mc.PointMassField(
    x_uas=torch.tensor([-0.4, 0.5]),
    y_uas=torch.tensor([0.2, -0.3]),
    mass_solar=torch.tensor([0.001834, 0.002739]),
    velocity_x_uas_per_day=bulk_x + dispersion * torch.tensor([0.7, -0.6]),
    velocity_y_uas_per_day=bulk_y + dispersion * torch.tensor([0.4, -0.3]),
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
print("apparent AB magnitudes:")
print(light_curve.magnitude)
