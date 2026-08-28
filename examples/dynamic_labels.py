"""Stream dynamic tiled-IPM maps together with robust caustic labels."""

import torch

import microcaustics as mc

far_field_approx = mc.FarFieldApproxConfig(
    cells_per_axis=8,
    nodes_per_cell_axis=8,
)
system = mc.MicrolensingSystem(
    macro=mc.MacroLens(convergence=0.25, shear=0.1),
    distances=mc.LensingDistances(8.0e24, 1.6e25, 9.0e24),
    stars=mc.PointMassField(
        x_uas=torch.tensor([-0.8, 0.4, 1.1]),
        y_uas=torch.tensor([0.5, -0.7, 0.2]),
        einstein_radius_uas=torch.tensor([0.25, 0.2, 0.18]),
        velocity_x_uas_per_day=torch.tensor([2.0e-4, -1.0e-4, 1.5e-4]),
        velocity_y_uas_per_day=torch.tensor([-1.0e-4, 1.0e-4, 0.5e-4]),
    ),
    source_grid=mc.PlaneGrid((256, 256), (2.0, 2.0)),
    duration_days=100.0,
    caustic_grid_shape=1024,
)
times_days = torch.arange(0.0, 101.0, 10.0)

frames = system.dynamic_labeled_maps(
    times_days,
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
