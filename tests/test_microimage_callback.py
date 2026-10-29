"""The optional image observer must preserve ordinary photometry."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from test_microimages import setup

import microcaustics as mc


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_callback_flux_selection_and_no_extra_tracing(device, selected, batch_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, method = setup(device, stars=8)
    method = replace(method, dual_scout_scalar_correction=True)
    schedule = mc.DynamicConfig(temporal_batch_size=batch_size)
    times = [0.0, 10.0, 20.0, 30.0, 40.0]
    frames, maps = [], []
    selection = (
        mc.MicroimageConfig(times_days=[10.0, 30.0], bands=["red"])
        if selected
        else None
    )
    common = dict(method=method, schedule=schedule)
    ordinary = sim.light_curve(region, grid, times, source, distances, **common)
    curve = sim.light_curve(
        region,
        grid,
        times,
        source,
        distances,
        **common,
        microimages=selection,
        microimage_observer=lambda i, f: frames.append((i, f)),
        map_observer=lambda i, f: maps.append(f),
    )
    torch.testing.assert_close(curve.flux, ordinary.flux, rtol=3e-4, atol=1e-5)
    assert curve.metadata["microimage_output"] is True
    torch.testing.assert_close(curve.unlensed_flux, ordinary.unlensed_flux)
    assert [i for i, f in frames] == ([1, 3] if selected else list(range(5)))
    assert len(maps) == 5 and curve.maps == ()
    for i, frame in frames:
        expected = curve.flux[i, 1:] if selected else curve.flux[i]
        torch.testing.assert_close(
            frame.total_flux_jy + frame.normalization_correction_flux_jy,
            expected,
            rtol=3e-5,
            atol=1e-5,
        )
        assert frame.wavelengths_angstrom == (
            (8000.0,) if selected else (4000.0, 8000.0)
        )
        assert frame.centroid_uas.shape == (len(expected), 2)
        torch.testing.assert_close(frame.rebin(3).total_flux_jy, frame.total_flux_jy)
        assert frame.flux_jy.device.type == device


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decimal_epoch_selection_preserves_photometry(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sim, source, distances, grid, region, method = setup(device)
    times = [0.1, 0.2, 0.3]
    if dtype is not None:
        times = torch.tensor(times, dtype=dtype)
    selected = []
    ordinary = sim.light_curve(region, grid, times, source, distances, method=method)
    curve = sim.light_curve(
        region, grid, times, source, distances, method=method,
        microimages=mc.MicroimageConfig(times_days=[0.1, 0.3]),
        microimage_observer=lambda i, frame: selected.append((i, frame)),
    )
    assert [i for i, _ in selected] == [0, 2]
    expected_times = torch.as_tensor(times).tolist()
    assert [frame.time_days for _, frame in selected] == [expected_times[0], expected_times[2]]
    torch.testing.assert_close(curve.flux, ordinary.flux, rtol=3e-4, atol=1e-5)


@pytest.mark.parametrize("selection", [[0.10001], [0.1, 0.100000001]])
def test_decimal_selection_rejects_absent_or_indistinguishable_epochs(selection):
    sim, source, distances, grid, region, method = setup()
    with pytest.raises(ValueError, match="time axis precision"):
        sim.light_curve(
            region, grid, [0.1, 0.2], source, distances, method=method,
            microimages=mc.MicroimageConfig(times_days=selection),
            microimage_observer=lambda i, frame: None,
        )


@pytest.mark.parametrize(
    "config,match",
    [
        (mc.MicroimageConfig(times_days=[1]), "evaluated map epochs"),
        (mc.MicroimageConfig(bands=["missing"]), "Unknown microimage band"),
        (True, "MicroimageConfig"),
    ],
)
def test_bad_selection_fails_before_raytrace(config, match):
    sim, source, distances, grid, region, method = setup()
    with (
        patch.object(
            type(sim), "raytrace_direct", side_effect=AssertionError("traced")
        ),
        pytest.raises((ValueError, TypeError), match=match),
    ):
        sim.light_curve(
            region,
            grid,
            [0, 10],
            source,
            distances,
            method=method,
            microimages=config,
            microimage_observer=lambda i, f: None,
        )


def test_missing_observer_and_unsupported_modes():
    sim, source, distances, grid, region, method = setup()
    with pytest.raises(ValueError, match="microimage_observer"):
        sim.light_curve(
            region,
            grid,
            [0],
            source,
            distances,
            method=method,
            microimages=mc.MicroimageConfig(),
        )
    with pytest.raises(ValueError, match="exact source evolution"):
        sim.light_curve(
            region,
            grid,
            [0],
            source,
            distances,
            method=method,
            source_evolution="linear_response",
            microimage_observer=lambda i, f: None,
        )


def test_zero_flux_centroid_and_rebin_coordinates():
    grid = mc.PlaneGrid((3, 5), (3.0, 5.0), center_uas=(10.0, 20.0))
    frame = mc.MicroimageFrame(
        torch.tensor([0, 14]),
        torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
        grid,
        0.0,
        0.0,
        (5000.0, 6000.0),
        torch.zeros(2),
    )
    torch.testing.assert_close(frame.centroid_uas[0], torch.tensor([21.0, 10.5]))
    assert torch.isnan(frame.centroid_uas[1]).all()
    rebinned = frame.rebin(2)
    assert rebinned.grid.shape == (2, 3)
    assert rebinned.grid.bounds_uas[0] == grid.bounds_uas[0]
    assert rebinned.grid.bounds_uas[2] == grid.bounds_uas[2]
    torch.testing.assert_close(rebinned.to_dense().sum((0, 1)), frame.total_flux_jy)


def test_source_evaluated_once_per_batch_and_callback_failure_propagates():
    sim, source, distances, grid, region, method = setup(stars=4)

    class Variable:
        geometry = source.geometry
        calls = []

        def brightness(self, times, **kwargs):
            self.calls.extend(torch.as_tensor(times).cpu().tolist())
            return source.brightness(times, **kwargs) * (
                1 + torch.as_tensor(times, **kwargs)[:, None, None, None] / 100
            )

    variable = Variable()
    sim.light_curve(
        region,
        grid,
        [0.0, 10.0, 20.0],
        variable,
        distances,
        method=method,
        schedule=mc.DynamicConfig(temporal_batch_size=3),
        microimage_observer=lambda i, f: None,
    )
    assert variable.calls == [0.0, 10.0, 20.0]

    def fail(i, f):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        sim.light_curve(
            region,
            grid,
            [0.0],
            source,
            distances,
            method=method,
            microimage_observer=fail,
        )


def test_high_level_api_and_retained_maps():
    sim, source, distances, grid, region, method = setup(stars=8)
    system = mc.MicrolensingSystem(
        macro=sim.macro_lens,
        distances=distances,
        source=source,
        stars=sim.point_masses,
        integration_domain="full",
        lens_region=region,
        source_grid=grid,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )
    seen = []
    curve = system.light_curve(
        [0.0, 10.0],
        method=method,
        temporal_batch_size=2,
        microimage_observer=lambda i, f: seen.append((i, f)),
        keep_maps_at_days=[10.0],
    )
    assert len(seen) == 2 and len(curve.maps) == 1
    torch.testing.assert_close(
        seen[1][1].total_flux_jy, curve.flux[1], rtol=3e-5, atol=1e-5
    )
    with pytest.raises(ValueError, match="same-cadence"):
        system.light_curve(
            [0.0, 10.0],
            flux_times_days=[0.0, 5.0, 10.0],
            microimage_observer=lambda i, f: None,
        )


def test_display_does_not_modify_data():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from microcaustics.plotting import plot_microimages

    grid = mc.PlaneGrid((3, 5), (3.0, 5.0))
    frame = mc.MicroimageFrame(
        torch.tensor([0, 14]),
        torch.ones((2, 1)),
        grid,
        0.0,
        0.0,
        (5000.0,),
        torch.zeros(1),
    )
    before = frame.flux_jy.clone()
    fig, ax = plot_microimages(frame, max_pixels=3, scale_bar_uas=1)
    assert len(fig.axes) == 2 and not ax.get_xticks().size
    torch.testing.assert_close(frame.flux_jy, before)
    plt.close(fig)


def test_native_crop_preserves_positions_and_pixel_values():
    from matplotlib import pyplot as plt

    from microcaustics.plotting import plot_microimages

    grid = mc.PlaneGrid((8, 10), (4.0, 5.0), center_uas=(10.0, 20.0))
    frame = mc.MicroimageFrame(
        torch.tensor([0, 23, 46, 79]),
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        grid,
        0.0,
        0.0,
        (5000.0,),
        torch.zeros(1),
    )
    crop = frame.crop((19.0, 21.0, 9.0, 10.5))
    torch.testing.assert_close(crop.to_dense(), frame.to_dense()[2:5, 3:7])
    assert crop.grid.pixel_scale_uas == grid.pixel_scale_uas
    torch.testing.assert_close(crop.centroid_uas[0], torch.tensor([20.15, 9.85]))
    fig, ax = plot_microimages(frame, bounds_uas=(19.0, 21.0, 9.0, 10.5), log=False)
    assert ax.images[0].get_array().shape == (3, 4)
    torch.testing.assert_close(
        torch.tensor(ax.images[0].get_array().data.copy()),
        crop.to_dense()[..., 0] * 4e6,
    )
    plt.close(fig)
    tight = crop.crop()
    torch.testing.assert_close(tight.total_flux_jy, crop.total_flux_jy)
    torch.testing.assert_close(tight.centroid_uas, crop.centroid_uas)
    with pytest.raises(ValueError, match="intersect"):
        frame.crop((100.0, 101.0, 100.0, 101.0))


def test_oom_split_preserves_every_flux_and_callback():
    from microcaustics.solvers import ipm

    sim, source, distances, grid, region, method = setup(stars=4)
    method = replace(method, cell_chunk_size=4096)
    schedule = mc.DynamicConfig(temporal_batch_size=4, minimum_cell_chunk_size=4096)
    original = ipm.temporal_batch_ipm
    calls, seen = [], []

    def limited(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(len(args[4]))
        if len(args[4]) > 2:
            raise RuntimeError("simulated CUDA out of memory after source preparation")
        return result

    with patch.object(ipm, "temporal_batch_ipm", limited):
        curve = sim.light_curve(
            region,
            grid,
            [0.0, 10.0, 20.0, 30.0],
            source,
            distances,
            method=method,
            schedule=schedule,
            microimage_observer=lambda i, f: seen.append((i, f)),
        )
    assert 4 in calls and 2 in calls
    assert [i for i, f in seen] == [0, 1, 2, 3]
    for index, frame in seen:
        torch.testing.assert_close(
            frame.total_flux_jy, curve.flux[index], rtol=3e-5, atol=1e-5
        )


def test_callback_reuses_raytracing():
    sim, source, distances, grid, region, method = setup(stars=4)
    original = type(sim).raytrace_direct
    traces = []
    for render in (False, True):
        count = []

        def traced(self, *args, count=count, **kwargs):
            count.append(args[0].numel())
            return original(self, *args, **kwargs)

        options = dict(microimage_observer=lambda i, f: None) if render else {}
        with patch.object(type(sim), "raytrace_direct", traced):
            sim.light_curve(
                region,
                grid,
                [0.0, 10.0, 20.0],
                source,
                distances,
                method=method,
                schedule=mc.DynamicConfig(temporal_batch_size=3),
                **options,
            )
        traces.append(count)
    assert traces[0] == traces[1] and traces[0]
