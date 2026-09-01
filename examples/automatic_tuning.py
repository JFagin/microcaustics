"""Tune lossless IPM work sizes, then stream a short dynamic sequence."""

from __future__ import annotations

import torch

import microcaustics as mc

system = mc.MicrolensingSystem(
    macro=mc.MacroLens(convergence=0.35, shear=0.2),
    distances=mc.LensingDistances(8.0e24, 1.6e25, 9.0e24),
    stars=mc.PointMassField(
        x_uas=torch.tensor([-0.6, 0.2, 0.8]),
        y_uas=torch.tensor([0.3, -0.4, 0.1]),
        mass_solar=torch.tensor([0.002264, 0.001449, 0.001834]),
        velocity_x_uas_per_day=torch.tensor([0.001, -0.0015, 0.0007]),
        velocity_y_uas_per_day=torch.tensor([-0.0005, 0.0008, -0.001]),
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
