"""Experimental fused microimages must not alter the ordinary map."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc
from microcaustics.microimages import MicroimageRequest


def setup(device="cpu", parity="minimum", stars=0):
    distances = mc.LensingDistances(1e25, 2e25, 1e25)
    geometry = mc.SourceGeometry(
        (24, 32),
        field_of_view_uas=(3.0, 4.0),
        bands_angstrom={"blue": 4000.0, "red": 8000.0},
    ).resolve(distances)
    grid = mc.PlaneGrid(geometry.shape, (3.0, 4.0))
    y, x = grid.mesh(device=device)
    values = torch.stack(
        (torch.exp(-(x * x + y * y) / 0.3), torch.exp(-(x * x + y * y) / 0.8)), dim=-1
    )
    source = mc.StaticSource(image=values / geometry.pixel_area_m2, geometry=geometry)
    generator = torch.Generator().manual_seed(17)
    positions = torch.rand((2, stars), generator=generator) * 10 - 5
    field = mc.PointMassField(
        positions[0],
        positions[1],
        mass_solar=torch.full((stars,), 0.03),
        velocity_x_uas_per_day=torch.full((stars,), 0.001),
        velocity_y_uas_per_day=torch.full((stars,), -0.0005),
    ).resolve(distances)
    sim = mc.MicrolensingSimulation.create(
        mc.MacroLens(
            convergence=0.2,
            shear=0.1 if parity == "minimum" else 1.1,
            smooth_matter_fraction=1.0,
        ),
        field,
        runtime=mc.RuntimeConfig(
            device=device,
            backend="triton" if device == "cuda" else "torch-eager",
            strict_backend=True,
        ),
    )
    config = mc.IPMConfig(
        rays=4096,
        tiled=True,
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        dual_scout_scalar_correction=False,
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    )
    return sim, source, distances, grid, mc.PlaneRegion((12.0, 12.0)), config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("parity", ["minimum", "saddle"])
@pytest.mark.parametrize("stars", [0, 32])
def test_flux_closure_and_unchanged_map(device, parity, stars):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, config = setup(device, parity, stars)
    ordinary = sim.magnification_map(region, grid, method=config, time_days=20.0)
    rendered = sim.magnification_map(
        region,
        grid,
        method=config,
        time_days=20.0,
        microimages=MicroimageRequest(source, distances),
    )
    torch.testing.assert_close(rendered.values, ordinary.values, rtol=3e-4, atol=3e-5)
    assert rendered.metadata["microimage_output"] is True
    pixel_flux = (
        source.brightness([20.0], device=device)[0] * source.geometry.pixel_area_m2
    )
    expected = (rendered.values[..., None] * pixel_flux).sum((0, 1))
    torch.testing.assert_close(
        rendered.microimages.total_flux_jy, expected, rtol=3e-5, atol=1e-6
    )
    torch.testing.assert_close(
        rendered.microimages.to_dense().sum((0, 1)), expected, rtol=3e-5, atol=1e-6
    )
    assert ordinary.microimages is None
    assert rendered.microimages.grid.shape == tuple(
        rendered.metadata["base_cell_grid_shape"]
    )
    assert torch.all(rendered.microimages.flux_jy >= 0)


def test_correction_is_separate_and_delay_is_preserved():
    sim, source, distances, grid, region, config = setup(stars=8)

    class VariableSource:
        geometry = source.geometry

        def brightness(self, t, **kwargs):
            t = torch.as_tensor(t, **kwargs)
            self.last_time = t.clone()
            delays = torch.arange(32, **kwargs)[None, None, :, None] / 4
            return source.brightness(t, **kwargs) * (
                1 + 0.2 * torch.sin(t[:, None, None, None] - delays)
            )

    variable = VariableSource()
    result = sim.magnification_map(
        region,
        grid,
        method=replace(config, dual_scout_scalar_correction=True),
        time_days=8.0,
        microimages=MicroimageRequest(variable, distances, macro_delay_days=3.0),
    )
    assert variable.last_time.item() == 5.0
    image = result.microimages
    assert image.time_days == 8.0 and image.source_time_days == 5.0
    pixels = (
        variable.brightness([5.0], device="cpu", dtype=torch.float32)[0]
        * source.geometry.pixel_area_m2
    )
    expected = (result.values[..., None] * pixels).sum((0, 1))
    torch.testing.assert_close(
        image.total_flux_jy + image.normalization_correction_flux_jy,
        expected,
        rtol=3e-5,
        atol=1e-6,
    )


def test_wrong_source_grid_rejected():
    sim, source, distances, grid, region, config = setup()
    with pytest.raises(ValueError, match="pixel scale"):
        sim.magnification_map(
            region,
            replace(grid, field_of_view_uas=(6.0, 8.0)),
            method=config,
            microimages=MicroimageRequest(source, distances),
        )


def test_rendering_does_not_add_raytracing_calls():
    sim, source, distances, grid, region, config = setup(stars=4)
    original = mc.MicrolensingSimulation.raytrace_direct
    counts = []
    for request in (None, MicroimageRequest(source, distances)):
        calls = []

        def tracked(self, *args, calls=calls, **kwargs):
            calls.append(args[0].numel())
            return original(self, *args, **kwargs)

        with patch.object(mc.MicrolensingSimulation, "raytrace_direct", tracked):
            sim.magnification_map(region, grid, method=config, microimages=request)
        counts.append(calls)
    assert counts[0] == counts[1] and counts[0]


def test_identity_image_has_correct_spatial_flux_not_only_total():
    sim, source, distances, grid, _, config = setup()
    sim = replace(
        sim,
        macro_lens=mc.MacroLens(convergence=0.0, shear=0.0, smooth_matter_fraction=1.0),
    )
    config = replace(
        config,
        rays=(grid.shape[0] + 1) * (grid.shape[1] + 1),
        tiled=False,
        scout_ratio=1,
        refinement=1,
        virtual_refinement=1,
    )
    result = sim.magnification_map(
        grid.region,
        grid,
        method=config,
        microimages=MicroimageRequest(source, distances),
    )
    expected = source.brightness([0.0])[0] * source.geometry.pixel_area_m2
    torch.testing.assert_close(
        result.microimages.to_dense(), expected, rtol=2e-5, atol=1e-6
    )


def test_empty_scout_returns_zero_image():
    sim, source, distances, grid, _, config = setup()
    result = sim.magnification_map(
        mc.PlaneRegion((1.0, 1.0), center_uas=(100.0, 100.0)),
        grid,
        method=config,
        microimages=MicroimageRequest(source, distances),
    )
    assert torch.count_nonzero(result.microimages.flux_jy) == 0
    assert torch.count_nonzero(result.values) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("variable", [False, True])
def test_temporal_images_match_independent_frames(device, variable):
    from microcaustics.solvers.ipm import temporal_batch_ipm

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, config = setup(device, stars=8)
    config = replace(config, tiled=False, cell_chunk_size=700)
    calls = []

    class EpochSource:
        geometry = source.geometry

        def brightness(self, times, **kwargs):
            calls.append(torch.as_tensor(times).numel())
            values = source.brightness(times, **kwargs)
            if variable:
                times = torch.as_tensor(times, **kwargs)
                # Both a spatial delay and spectral dependence must survive batching.
                delay = torch.arange(32, **kwargs)[None, None, :, None] / 4
                amplitude = torch.tensor([0.1, 0.2], **kwargs)
                values = values * (
                    1 + amplitude * torch.sin(times[:, None, None, None] - delay)
                )
            return values

    request = MicroimageRequest(EpochSource(), distances, macro_delay_days=3)
    times = (20, 25, 30)
    results = temporal_batch_ipm(sim, region, grid, config, times, _microimages=request)
    assert calls == [3]
    for time, result in zip(times, results, strict=True):
        reference = sim.magnification_map(
            region, grid, method=config, time_days=time, microimages=request
        )
        torch.testing.assert_close(
            result.values, reference.values, rtol=3e-4, atol=3e-5
        )
        torch.testing.assert_close(
            result.microimages.flux_jy,
            reference.microimages.flux_jy,
            rtol=3e-4,
            atol=2e-6,
        )
        assert result.microimages.source_time_days == time - 3
        assert result.microimages.time_days == time


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scheduled_images_reuse_scout_and_handle_last_batch(device):
    from microcaustics.dynamic import dynamic_maps

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, method = setup(device, stars=8)
    settings = mc.DynamicConfig(
        temporal_batch_size=2,
        scout_refresh_frames=2,
        tuning=mc.AutoTuningConfig(enabled=False),
    )
    times = (0, 10, 20)
    args = (sim, region, grid, times)
    ordinary = list(dynamic_maps(*args, method=method, config=settings))
    rendered = list(
        dynamic_maps(
            *args,
            method=method,
            config=settings,
            microimages=MicroimageRequest(source, distances),
        )
    )
    assert len(rendered) == 3
    for actual, expected in zip(rendered, ordinary, strict=True):
        torch.testing.assert_close(actual.values, expected.values, rtol=3e-4, atol=3e-5)
        pixels = (
            source.brightness([actual.time_days], device=device)[0]
            * source.geometry.pixel_area_m2
        )
        torch.testing.assert_close(
            actual.microimages.total_flux_jy,
            (actual.values[..., None] * pixels).sum((0, 1)),
            rtol=3e-5,
            atol=1e-6,
        )
        assert (
            actual.metadata["dynamic_scout_anchor_frames"]
            == expected.metadata["dynamic_scout_anchor_frames"]
        )
        assert actual.microimages.time_days == actual.time_days


def test_static_lens_does_not_reuse_stale_variable_microimages():
    from microcaustics.dynamic import dynamic_maps

    sim, source, distances, grid, region, method = setup()

    class VariableSource:
        geometry = source.geometry

        def brightness(self, times, **kwargs):
            t = torch.as_tensor(times, **kwargs)
            return source.brightness(times, **kwargs) * (
                1 + t[:, None, None, None] * 0.1
            )

    settings = mc.DynamicConfig(
        temporal_batch_size=2, tuning=mc.AutoTuningConfig(enabled=False)
    )
    results = list(
        dynamic_maps(
            sim,
            region,
            grid,
            [0, 1, 2],
            method=method,
            config=settings,
            microimages=MicroimageRequest(VariableSource(), distances),
        )
    )
    for i, result in enumerate(results):
        torch.testing.assert_close(
            result.microimages.flux_jy,
            results[0].microimages.flux_jy * (1 + 0.1 * i),
            rtol=3e-5,
            atol=1e-6,
        )


def test_static_preparation_shares_buffers_without_cpu_copy():
    from microcaustics.microimages import prepare_microimage_batch

    sim, source, distances, grid, _, _ = setup()
    pixels, prefix = prepare_microimage_batch(
        MicroimageRequest(source, distances), grid, (0, 10, 20), sim.runtime
    )
    assert pixels.stride(0) == prefix.stride(0) == 0
    torch.testing.assert_close(
        prefix[:, :, 1:] - prefix[:, :, :-1], pixels, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_temporal_correction_and_empty_selection(device):
    from microcaustics.solvers.ipm import temporal_batch_ipm

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, config = setup(device)
    indices = torch.empty(0, device=device, dtype=torch.long)
    results = temporal_batch_ipm(
        sim,
        region,
        grid,
        config,
        [0, 1],
        selected_cell_indices=indices,
        selected_cell_shape=(64, 64),
        scalar_correction=0.15,
        _microimages=MicroimageRequest(source, distances),
    )
    pixels = source.brightness([0], device=device)[0] * source.geometry.pixel_area_m2
    for result in results:
        assert result.microimages.flux_jy.shape == (0, 2)
        torch.testing.assert_close(
            result.microimages.total_flux_jy, torch.zeros(2, device=device)
        )
        torch.testing.assert_close(
            result.microimages.normalization_correction_flux_jy,
            (pixels * result.values[..., None]).sum((0, 1)),
        )


def test_microimage_request_survives_temporal_memory_backoff():
    from microcaustics.dynamic import DynamicMapScheduler
    from microcaustics.solvers import ipm

    sim, source, distances, grid, region, config = setup()
    config = replace(config, tiled=False, cell_chunk_size=64)
    settings = mc.DynamicConfig(
        temporal_batch_size=2,
        minimum_cell_chunk_size=64,
        tuning=mc.AutoTuningConfig(enabled=False),
    )
    request = MicroimageRequest(source, distances)
    scheduler = DynamicMapScheduler(
        sim, region, grid, [0, 1, 2], config, settings, microimages=request
    )
    original = ipm.temporal_batch_ipm
    calls = []

    def limited(*args, **kwargs):
        assert kwargs["_microimages"] is request
        calls.append(len(args[4]))
        if len(args[4]) > 1:
            raise torch.OutOfMemoryError("deliberate allocation failure")
        return original(*args, **kwargs)

    with patch.object(ipm, "temporal_batch_ipm", limited):
        results, _, splits = scheduler._calculate_ipm_batch_with_backoff(
            config, [0, 1, 2]
        )
    assert len(results) == 3 and splits == 2
    assert calls == [3, 1, 2, 1, 1]
    assert all(result.microimages is not None for result in results)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("centres", [False, True])
def test_fused_temporal_scout_preserves_each_endpoint_selection(ratio, centres):
    from microcaustics.solvers.far_field import temporal_taylor_far_fields
    from microcaustics.solvers.ipm import (
        _source_scout_cells,
        _source_scout_union_temporal,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, _, _, grid, region, config = setup("cuda", parity="saddle", stars=32)
    config = replace(
        config,
        rays=65536,
        scout_ratio=ratio,
        scout_trace_centers=centres,
        far_field_approx=mc.FarFieldApproxConfig(enabled=True, cells_per_axis=4),
    )
    times = (0.0, 25.0, 200.0)
    fields, _ = temporal_taylor_far_fields(sim, region, config.far_field_approx, times)
    result = _source_scout_union_temporal(
        sim, fields, region, grid, config, _return_selected_masks=True
    )
    union, ny, nx, metadata, components, masks = result
    expected = []
    for time, field, actual, row in zip(
        times, fields, components, metadata, strict=True
    ):
        cells, single_ny, single_nx, single_meta = _source_scout_cells(
            sim, field, region, grid, config, time_days=time
        )
        assert (ny, nx) == (single_ny, single_nx)
        torch.testing.assert_close(actual, cells)
        assert row["scout_trace_centers"] == centres
        assert row["scout_center_traces"] == single_meta["scout_center_traces"]
        assert (
            row["scout_selected_before_dilation"]
            == single_meta["scout_selected_before_dilation"]
        )
        expected.append(cells)
    torch.testing.assert_close(union.sort().values, torch.unique(torch.cat(expected)))
    assert masks.shape[0] == len(times)


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("outside", [False, True])
def test_raster_preserves_cell_flux_with_inactive_cells(frames, outside):
    from microcaustics.solvers import triton_ipm
    from microcaustics.solvers.ipm import temporal_batch_ipm

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, config = setup("cuda")
    if outside:
        region = replace(region, center_uas=(1000.0, 1000.0))
    indices = torch.arange(4096, device="cuda")

    def run():
        return temporal_batch_ipm(
            sim,
            region,
            grid,
            config,
            list(range(frames)),
            selected_cell_indices=indices,
            selected_cell_shape=(64, 64),
            _microimages=MicroimageRequest(source, distances),
        )

    actual = run()
    original = triton_ipm._direct_cell_kernel

    class Uncompacted:
        def __getitem__(self, launch_grid):
            def launch(*args, **kwargs):
                kwargs["COMPACT_ACTIVE"] = False
                return original[launch_grid](*args, **kwargs)

            return launch

    with patch.object(triton_ipm, "_direct_cell_kernel", Uncompacted()):
        expected = run()
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.values, e.values, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            a.microimages.flux_jy, e.microimages.flux_jy, rtol=3e-5, atol=1e-7
        )
        if outside:
            assert torch.count_nonzero(a.values) == 0
            assert torch.count_nonzero(a.microimages.flux_jy) == 0
