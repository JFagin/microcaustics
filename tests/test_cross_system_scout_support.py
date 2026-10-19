"""CPU scheduling regression for cross-system map and label scout support.

Numerical kernels are stubbed, but the real cross-system scheduling and mask
expansion run here. Moving, disjoint masks make a per-interval map queue fail
even without a GPU, including batches that straddle refresh intervals.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc
from microcaustics.dynamic import _cross_system_tiled_ipm_maps


@pytest.mark.parametrize("batch_size", [4, 5])
@pytest.mark.parametrize("system_count", [2, 3])
@pytest.mark.parametrize("include_labels", [False, True])
def test_cross_system_maps_use_per_system_temporal_union(
    batch_size,
    system_count,
    include_labels,
):
    times = tuple(float(i) for i in range(11))
    refresh, ratio = 2, 2
    fine_nx = 16
    grid = mc.PlaneGrid((4, 4), (1.0, 1.0))
    region = mc.PlaneRegion((4.0, 4.0))
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        backend="torch-eager",
    )
    macro = mc.MacroLens(0.2, 0.1)
    simulations = [
        SimpleNamespace(owner=i, runtime=runtime, macro_lens=macro)
        for i in range(system_count)
    ]
    method = mc.production_ipm_config(rays=256)
    schedule = mc.DynamicConfig(
        temporal_batch_size=batch_size, scout_refresh_frames=refresh
    )
    requests = [(sim, region, grid, times, method, schedule) for sim in simulations]
    caustic_config = mc.CausticConfig()
    caustic_requests = (
        [(grid, caustic_config, None, False, None)] * system_count
        if include_labels
        else None
    )
    map_calls, label_calls, correction_calls = [], [], []

    def far_fields(simulation, _region, _config, _times, indices):
        return tuple((simulation.owner, i) for i in indices), {}

    def scout(_simulation, fields, *_args, **_kwargs):
        masks = torch.zeros((len(fields), system_count, 8), dtype=torch.bool)
        for row, (owner, index) in enumerate(fields):
            masks[row, owner, index % 8] = True
        return None, system_count * ratio, fine_nx, [], (), masks

    def correction(simulation, *_args, **_kwargs):
        correction_calls.append(simulation.owner)
        return torch.tensor(0.1 * (simulation.owner + 1)), {}

    def rasterize(_simulation, _region, _grid, _method, frame_times, **kwargs):
        map_calls.append((list(frame_times), kwargs))
        return tuple(SimpleNamespace(metadata={}) for _ in frame_times)

    def caustic_fields(_simulation, _grid, frame_times, _config, **kwargs):
        label_calls.append((list(frame_times), kwargs))
        return tuple(kwargs["far_fields"])

    def labels(fields, *_args, **_kwargs):
        return tuple(fields), None, None, None, None

    with (
        patch(
            "microcaustics.solvers.far_field.temporal_taylor_far_field_window",
            side_effect=far_fields,
        ),
        patch(
            "microcaustics.solvers.ipm._source_scout_union_temporal", side_effect=scout
        ),
        patch(
            "microcaustics.solvers.ipm.dual_scout_scalar_correction",
            side_effect=correction,
        ),
        patch("microcaustics.solvers.ipm.temporal_batch_ipm", side_effect=rasterize),
        patch(
            "microcaustics.caustics.production.caustic_fields_from_far_fields",
            side_effect=caustic_fields,
        ),
        patch(
            "microcaustics.caustics.production.label_caustic_fields", side_effect=labels
        ),
        patch(
            "microcaustics.dynamic._scheduled_map", side_effect=lambda item, **_: item
        ),
    ):
        maps, label_results = _cross_system_tiled_ipm_maps(
            requests, caustic_requests=caustic_requests
        )

    assert [len(row) for row in maps] == [len(times)] * system_count
    assert correction_calls == list(range(system_count))
    assert len(map_calls) == 3
    assert len(label_calls) == (3 if include_labels else 0)
    if include_labels:
        assert label_results == tuple(
            tuple((owner, i) for i in range(len(times)))
            for owner in range(system_count)
        )

    def endpoint_indices(index):
        return (
            index // refresh * refresh,
            min(len(times) - 1, (index // refresh + 1) * refresh - 1),
        )

    def expected_cells(owner, anchors):
        # Independent construction, deliberately not the production expander.
        return sorted(
            {
                (owner * ratio + y) * fine_nx + (i % 8) * ratio + x
                for i in anchors
                for y in range(ratio)
                for x in range(ratio)
            }
        )

    for batch, (frame_times, kwargs) in enumerate(map_calls):
        indices = list(
            range(batch * batch_size, min(len(times), (batch + 1) * batch_size))
        )
        anchors = {
            endpoint for index in indices for endpoint in endpoint_indices(index)
        }
        assert frame_times == [times[i] for _ in simulations for i in indices]
        assert kwargs["far_fields"] == [
            (owner, i) for owner in range(system_count) for i in indices
        ]
        for slot, (owner, index) in enumerate(kwargs["far_fields"]):
            actual = kwargs["selected_cell_indices_by_frame"][slot]
            assert sorted(actual.tolist()) == expected_cells(owner, anchors)
            assert kwargs["scalar_correction"][slot].item() == pytest.approx(
                0.1 * (owner + 1)
            )
            if include_labels:
                # Keep the narrower label support unchanged, matching serial
                # caustic generation rather than expanding it with map support.
                label_cells = label_calls[batch][1]["selected_cell_indices_by_frame"][
                    slot
                ]
                assert sorted(label_cells.tolist()) == expected_cells(
                    owner, endpoint_indices(index)
                )
