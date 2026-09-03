"""Stream dynamic tiled-IPM maps together with robust caustic labels."""

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
far_field_approx = mc.FarFieldApproxConfig(
    cells_per_axis=8,
    nodes_per_cell_axis=8,
)
system = mc.MicrolensingSystem(
    macro=mc.MacroLens(convergence=0.25, shear=0.1),
    distances=distances,
    stars=mc.PointMassField(
        x_uas=torch.tensor([-0.8, 0.4, 1.1]),
        y_uas=torch.tensor([0.5, -0.7, 0.2]),
        mass_solar=torch.tensor([0.003537, 0.002264, 0.001834]),
        # The shared drift contains projected CMB, lens, and source motion.
        # The independent offsets represent the stellar velocity dispersion.
        velocity_x_uas_per_day=bulk_x
        + dispersion * torch.tensor([0.8, -0.5, 0.3]),
        velocity_y_uas_per_day=bulk_y
        + dispersion * torch.tensor([-0.4, 0.6, 0.2]),
    ),
    duration_days=100.0,
    caustic_grid_shape=1024,
)
times_days = torch.arange(0.0, 101.0, 10.0)

frames = system.dynamic_labeled_maps(
    times_days,
    map_width_uas=2.0,
    map_pixels=256,
    method=mc.IPMConfig(
        rays=1_000_000,
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        far_field_approx=far_field_approx,
    ),
    schedule=mc.DynamicConfig(
        temporal_batch_size=8,
        scout_refresh_frames=10,
    ),
    caustics=mc.CausticConfig(
        far_field_approx=far_field_approx,
        temporal_batch_size=8,
    ),
)

for frame in frames:
    print(
        f"t={frame.time_days:6.1f} d  "
        f"segments={frame.caustics.caustics.segment_count:5d}  "
        f"label={frame.caustics.labels.center_label}  "
        f"crossing={frame.caustics.labels.center_crossing}"
    )
