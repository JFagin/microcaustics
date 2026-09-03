"""Tune lossless IPM work sizes, then stream a short dynamic sequence."""

from __future__ import annotations

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
system = mc.MicrolensingSystem(
    macro=mc.MacroLens(convergence=0.35, shear=0.2),
    distances=distances,
    stars=mc.PointMassField(
        x_uas=torch.tensor([-0.6, 0.2, 0.8]),
        y_uas=torch.tensor([0.3, -0.4, 0.1]),
        mass_solar=torch.tensor([0.002264, 0.001449, 0.001834]),
        velocity_x_uas_per_day=bulk_x
        + dispersion * torch.tensor([0.8, -0.5, 0.3]),
        velocity_y_uas_per_day=bulk_y
        + dispersion * torch.tensor([-0.4, 0.6, 0.2]),
    ),
)

method = mc.IPMConfig(
    rays=100_000,
    scout_ratio=2,
    refinement=2,
    virtual_refinement=4,
)
schedule = mc.DynamicConfig(
    tuning=mc.AutoTuningConfig(
        enabled=True,
        temporal_candidates=(1, 4, 8),
        spatial_candidates=(16_384, 32_768, 65_536),
        maximum_trial_frames=8,
    )
)

maps = system.dynamic_maps(
    torch.arange(8, dtype=torch.float64).tolist(),
    map_width_uas=2.0,
    map_pixels=128,
    method=method,
    schedule=schedule,
)

for frame in maps:
    print(
        frame.time_days,
        frame.metadata["autotune_temporal_batch_size"],
        frame.metadata["autotune_spatial_chunk_size"],
        frame.metadata["autotune_cache_hit"],
    )
