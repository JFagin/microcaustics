"""Derive several source trajectories from one streamed map sequence."""

import torch

import microcaustics as mc

distances = mc.LensingDistances(8.0e24, 1.6e25, 9.0e24)
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
stars = mc.PointMassField(
    x_uas=torch.tensor([-0.5, 0.6]),
    y_uas=torch.tensor([0.2, -0.3]),
    mass_solar=torch.tensor([0.001274, 0.002264]),
    velocity_x_uas_per_day=bulk_x + dispersion * torch.tensor([0.7, -0.6]),
    velocity_y_uas_per_day=bulk_y + dispersion * torch.tensor([-0.4, 0.5]),
)
geometry = mc.SourceGeometry(
    shape=(24, 24),
    pixel_scale_m=(1.0e12, 1.0e12),
    wavelengths_angstrom=(5000.0,),
    band_names=("continuum",),
)
source = mc.GaussianSource(geometry, sigma_m=3.0e12)
system = mc.MicrolensingSystem(
    macro=mc.MacroLens(convergence=0.1, shear=0.05),
    distances=distances,
    stars=stars,
    source=source,
    # This map covers both demonstration trajectories, not only the source.
    source_grid=mc.PlaneGrid((64, 64), (2.0, 2.0)),
    duration_days=100.0,
)
requests = (
    mc.LightCurveRequest(
        source,
        distances,
        trajectory=mc.LinearTrajectory(velocity_uas_per_day=(2.0e-4, 0.0)),
        name="horizontal",
    ),
    mc.LightCurveRequest(
        source,
        distances,
        trajectory=mc.LinearTrajectory(velocity_uas_per_day=(0.0, 2.0e-4)),
        name="vertical",
    ),
)
curves = system.light_curves(
    torch.arange(0.0, 101.0, 25.0),
    requests,
    method=mc.IPMConfig(
        rays=65_536,
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        dual_scout_scalar_correction=True,
    ),
    schedule=mc.DynamicConfig(
        temporal_batch_size=5,
        light_curve_batch_size=2,
    ),
)

for curve in curves:
    print(curve.metadata["request_name"], curve.flux / curve.unlensed_flux)
