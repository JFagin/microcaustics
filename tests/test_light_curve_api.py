"""Unified photometry, driver reproducibility, and actionable API validation."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import microcaustics as mc
from microcaustics.sources.variability import _source_at_driver_mean
from microcaustics.system import _light_curve_options, _light_curve_times


def small_system(*, signal=None, source=None, **kwargs):
    """A tiny physical source and direct lens field for fast portable checks."""

    distances = mc.LensingDistances.from_redshifts(0.5, 1.5)
    geometry = mc.SourceGeometry(
        shape=8, field_of_view_uas=2, bands_angstrom={"g": 4800}
    ).resolve(distances)
    if source is None:
        source = mc.StaticSource(torch.full((8, 8, 1), 1e-25), geometry)
    if signal is not None:
        source = mc.ModulatedSource(source, signal)
    return mc.MicrolensingSystem(
        macro=mc.MacroLens(convergence=0, shear=0),
        distances=distances,
        stars=mc.PointMassField([0.0], [0.0], [0.001]),
        source=source,
        lens_plane_uas=4,
        caustic_grid_shape=16,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        seed=12,
        **kwargs,
    )


def small_options():
    return dict(
        method=mc.IPMConfig(
            rays=64,
            tiled=False,
            refinement=1,
            virtual_refinement=1,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        ),
        temporal_batch_size=2,
        scout_refresh_frames=1,
    )


def test_unified_labels_and_multirate_match_internal_calculation():
    system = small_system()
    caustics = mc.CausticConfig(
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        minimum_determinant_sign_pixels=1,
        anchor_count=3,
        gauge_count=3,
        minimum_alignment_gauges=1,
    )
    options = small_options()
    actual = system.light_curve(
        duration_days=2,
        map_cadence_days=1,
        source_cadence_days=0.5,
        include_labels=True,
        caustics=caustics,
        keep_maps_at_days=(2, 0),
        **options,
    )
    expected = system._realize_for_times((0, 1, 2)).multirate_light_curve_with_labels(
        (0, 1, 2),
        (0, 0.5, 1, 1.5, 2),
        method=options["method"],
        caustics=caustics,
        schedule=mc.production_dynamic_config(
            temporal_batch_size=2, scout_refresh_frames=1
        ),
    )
    assert isinstance(actual, mc.LightCurve)
    assert actual.flux.shape == (5, 1)
    assert actual.labels.times_days.tolist() == [0, 1, 2]
    assert actual.map_times_days.tolist() == [0, 2]
    assert actual.maps[0].time_days == 0
    torch.testing.assert_close(actual.flux, expected.light_curve.flux)
    torch.testing.assert_close(actual.labels.crossing_labels, expected.crossing_labels)
    torch.testing.assert_close(mc.mag_to_flux(actual.magnitude), actual.flux)


def test_unavailable_retained_maps_warn_and_omit_without_extra_evaluations():
    system = small_system()
    with pytest.warns(UserWarning, match="day 0.2.*omitted"):
        result = system.light_curve(
            (0, 1), keep_maps_at_days=(1, 0.2, 0, 1 + 1e-9), **small_options()
        )
    assert result.map_times_days.tolist() == [0, 1]
    assert result.labels is None
    assert result.times_days.tolist() == [0, 1]
    empty = system.light_curve((0,), **small_options())
    assert not empty.maps and empty.map_times_days.numel() == 0
    with pytest.raises(IndexError, match="keep_maps_at_days"):
        _ = empty.maps[0]


@pytest.mark.parametrize("include_labels,batch", [(False, 49), (True, 30)])
def test_warmup_and_main_call_share_defaults(include_labels, batch):
    system = small_system()
    with patch.object(mc.MicrolensingSystem, "light_curve", return_value=None) as run:
        system.warmup_light_curve(include_labels=include_labels)
    assert len(run.call_args.args[0]) == batch
    assert run.call_args.kwargs["schedule"].temporal_batch_size == batch
    assert run.call_args.kwargs["include_labels"] == include_labels
    assert (
        _light_curve_options({}, include_labels=include_labels)[
            "schedule"
        ].temporal_batch_size
        == batch
    )


def test_explicit_controls_and_invalid_time_arguments():
    options = _light_curve_options(
        dict(
            rays=1234, temporal_batch_size=7, scout_refresh_frames=3, label_batch_size=2
        ),
        include_labels=True,
    )
    assert options["method"].rays == 1234
    assert options["schedule"].temporal_batch_size == 7
    assert options["schedule"].scout_refresh_frames == 3
    assert options["caustics"].temporal_batch_size == 2
    custom_far_field = mc.FarFieldApproxConfig(enabled=False)
    controls = _light_curve_options(
        dict(
            method=mc.production_ipm_config(far_field_approx=custom_far_field),
            label_batch_size=2,
        ),
        include_labels=True,
    )
    assert controls["caustics"].far_field_approx is custom_far_field
    with pytest.raises(ValueError, match="include_labels"):
        _light_curve_options(dict(label_batch_size=2), include_labels=False)
    for invalid in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            _light_curve_times(duration_days=2, map_cadence_days=invalid)
    with pytest.raises(ValueError, match="not both"):
        _light_curve_times((0, 1), duration_days=1)
    with pytest.raises(ValueError, match="within"):
        _light_curve_times((0, 1), flux_times_days=(0, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_horizon_driver_is_lazy_and_subset_reproducible(dtype):
    specification = mc.broken_power_law_driving_signal(
        cadence_days=0.1,
        max_duration_days=40,
        history_days=10,
        standard_deviation=0.1,
        dtype=dtype,
    )
    first = small_system(signal=specification)
    second = small_system(signal=specification)
    driver = first._bound_driving_signal
    assert driver._sampled is None
    assert driver.seed == first.seed_for("variability")
    short = torch.arange(5, dtype=dtype)
    long = torch.arange(20, dtype=dtype)
    args = dict(bands=1, dtype=dtype, device="cpu")
    expected = driver.amplitudes(short, **args)
    torch.testing.assert_close(
        driver.amplitudes(long, **args)[:5], expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        second._bound_driving_signal.amplitudes(short, **args), expected, rtol=0, atol=0
    )
    assert first._realize_for_times((0, 3)).system._bound_driving_signal is driver
    assert first._realize_for_times((0, 7)).system._bound_driving_signal is driver
    changed = replace(first, seed=13)._bound_driving_signal.amplitudes(short, **args)
    assert not torch.equal(changed, expected)
    override = mc.broken_power_law_driving_signal(
        max_duration_days=10, history_days=1, seed=99
    )
    assert small_system(signal=override)._bound_driving_signal.seed == 99
    for outside in (-11, 41):
        with pytest.raises(ValueError, match="history_days.*max_duration_days"):
            driver.amplitudes([outside], **args)


def test_fractional_day_float32_psd_grid_and_irregular_rejection():
    times = torch.arange(83001) * 0.1 - 1000
    signal = mc.broken_power_law_driving_signal(times, seed=7, dtype=torch.float32)
    assert signal.values.dtype == torch.float32
    assert bool(torch.all(torch.isfinite(signal.values)))
    with pytest.raises(ValueError, match="regular increasing"):
        mc.broken_power_law_driving_signal([0, 1, 2.5, 3])


def test_source_driver_switch_preserves_geometry_and_does_not_sample_when_off():
    spec = mc.broken_power_law_driving_signal(max_duration_days=10, history_days=2)
    system = small_system(signal=spec)
    off = system.light_curve((0, 1, 2), apply_driving_signal=False, **small_options())
    assert system._bound_driving_signal._sampled is None
    baseline = small_system().light_curve((0, 1, 2), **small_options())
    torch.testing.assert_close(off.flux, baseline.flux)
    on = system.light_curve((0, 1, 2), **small_options())
    assert not torch.equal(on.flux, off.flux)
    assert (
        system._realize_for_times((0, 1, 2)).source.geometry
        == system.realize().source.geometry
    )
    with pytest.raises(ValueError, match="requires a source"):
        small_system().light_curve((0,), apply_driving_signal=True)


def test_disabling_thermal_variability_preserves_mean_lamp_heating():
    shape = (4, 4)
    radius = torch.full(shape, 20.0)
    transfer = mc.ObserverTransfer(
        radius,
        torch.ones(shape),
        torch.ones(shape),
        torch.ones(shape, dtype=torch.bool),
    )
    geometry = mc.SourceGeometry(shape, (1e12, 1e12), (4800.0,), ("g",))
    signal = mc.broken_power_law_driving_signal(
        mean_amplitude=1.4, max_duration_days=10, history_days=2
    )
    source = mc.ThermalReprocessingSource(
        geometry,
        transfer,
        signal,
        torch.full(shape, 1e-4),
        torch.ones(shape),
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        spin=0,
        source_redshift=1,
    )
    mean = _source_at_driver_mean(source)
    expected = replace(
        source, signal=mc.CallableDrivingSignal(lambda t: torch.full_like(t, 1.4))
    )
    torch.testing.assert_close(mean.brightness([0, 1]), expected.brightness([0, 1]))
    assert mean.response_temperature4 is source.response_temperature4
    assert mean.transfer is source.transfer
    assert mean.is_time_static and signal._sampled is None


def test_disabled_custom_modulation_does_not_freeze_source_evolution():
    source = small_system().source
    evolving = mc.CallableSource(
        lambda times: source.image[None] * (1 + times[:, None, None, None]),
        source.geometry,
    )
    signal = mc.broken_power_law_driving_signal()
    mean = _source_at_driver_mean(mc.ModulatedSource(evolving, signal))
    assert not mean.is_time_static
    torch.testing.assert_close(mean.brightness([0, 1]), evolving.brightness([0, 1]))


def test_label_and_map_times_are_independent_and_flux_is_not_round_tripped():
    frame = SimpleNamespace(
        time_days=0,
        labels=SimpleNamespace(
            center_label=1,
            center_crossing=False,
            center_distance_uas=0.2,
            center_distance_censored=False,
        ),
    )
    flux = torch.tensor([[0.0], [1e-5]])
    result = mc.LightCurve(
        torch.tensor([0, 1]),
        flux,
        ("g",),
        labels=mc.LightCurveLabels.from_caustics((frame,)),
    )
    assert result.flux is flux and torch.isnan(result.magnitude[0, 0])
    assert result.labels.times_days.tolist() == [0]
    assert result.map_times_days.numel() == 0
    assert mc.mag_to_flux(float("inf")) == 0


def test_independent_batch_matches_single_calls_with_driver_and_retained_maps():
    driver = mc.broken_power_law_driving_signal(max_duration_days=10, history_days=2)
    systems = [replace(small_system(signal=driver), seed=seed) for seed in (11, 12)]
    options = dict(
        duration_days=2,
        map_cadence_days=1,
        source_cadence_days=0.5,
        keep_maps_at_days=(0, 2),
        **small_options(),
    )
    batch = mc.batched_system_light_curves(systems, curves_per_batch=2, **options)
    assert batch.wall_seconds is None and batch.seconds_per_curve is None
    for system, actual in zip(systems, batch.light_curves, strict=True):
        expected = system.light_curve(**options)
        torch.testing.assert_close(actual.flux, expected.flux)
        assert actual.map_times_days.tolist() == [0, 2]
        torch.testing.assert_close(actual.maps[0].values, expected.maps[0].values)
    profiled = mc.batched_system_light_curves(systems, profile=True, **options)
    assert profiled.wall_seconds > 0 and profiled.seconds_per_curve > 0
    off = mc.batched_system_light_curves(systems, apply_driving_signal=False, **options)
    for system, actual in zip(systems, off.light_curves, strict=True):
        torch.testing.assert_close(
            actual.flux, system.light_curve(apply_driving_signal=False, **options).flux
        )


def test_multi_image_driver_is_shared_and_delayed_with_consistent_results():
    single = small_system()
    spec = mc.broken_power_law_driving_signal(max_duration_days=10, history_days=3)
    multi = mc.MultiImageSystem(
        images={"A": single.macro, "B": single.macro},
        distances=single.distances,
        stars=single.stars,
        source=mc.ModulatedSource(single.source, spec),
        seed=71,
        lens_region=mc.PlaneRegion((4, 4)),
        caustic_grid_shape=16,
        runtime=single.runtime,
        arrival_time_delays_days={"A": 0, "B": 1},
    )
    controls = dict(
        duration_days=2,
        map_cadence_days=1,
        source_cadence_days=0.5,
        keep_maps_at_days=(0,),
        **small_options(),
    )
    off = multi.light_curves(apply_driving_signal=False, **controls)
    assert multi._shared_driving_signal._sampled is None
    on = multi.light_curves(apply_driving_signal=True, **controls)
    assert (
        multi.image("A")._bound_driving_signal is multi.image("B")._bound_driving_signal
    )
    for name, delay in (("A", 0), ("B", 1)):
        assert on[name].magnitude.shape == (5, 1)
        assert on[name].map_times_days.tolist() == [0]
        assert on[name].labels is None
        amplitude = multi._shared_driving_signal.amplitudes(
            on[name].times_days - delay,
            bands=1,
            device="cpu",
            dtype=on[name].flux.dtype,
        )
        torch.testing.assert_close(on[name].flux, off[name].flux * amplitude)


def test_invalid_retention_is_rejected_before_realization():
    system = small_system()
    with (
        patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize,
        pytest.raises(ValueError, match="finite"),
    ):
        system.light_curve((0, 1), keep_maps_at_days=(float("nan"),))
    realize.assert_not_called()


@pytest.mark.parametrize("reload_archive", [False, True])
def test_training_export_keeps_coarse_label_axis(monkeypatch, tmp_path, reload_archive):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from training_set_support import save_example

    system = small_system()

    def frame(time):
        return SimpleNamespace(
            time_days=time,
            labels=SimpleNamespace(
                center_label=1,
                center_crossing=False,
                center_distance_uas=0.2,
                center_distance_censored=False,
            ),
        )

    curve = mc.LightCurve(
        torch.tensor([0.0, 0.5, 1.0]),
        torch.ones((3, 1)),
        ("g",),
        unlensed_flux=torch.ones((3, 1)),
        labels=mc.LightCurveLabels.from_caustics((frame(0), frame(1))),
        metadata={"training_method": {"rays": 128}},
    )
    if reload_archive:
        curve = mc.load_light_curve(mc.save_light_curve(curve, tmp_path / "curve.npz"))
        assert curve.labels.caustics is None
    save_example(
        tmp_path,
        0,
        curve,
        np.ones(2),
        system.source,
        {"source_fov_uas": 2.0},
        runtime_seconds=0.1,
        worker=0,
    )
    with np.load(tmp_path / "light_curve_00000.npz") as saved:
        assert saved["map_times_days"].tolist() == [0, 1]
        assert saved["times_days"].tolist() == [0, 0.5, 1]
        np.testing.assert_allclose(saved["center_distance_uas"], [0.2, 0.2])
        assert saved["crossing_labels"].tolist() == [1, 1]
    metadata = json.loads((tmp_path / "light_curve_00000.json").read_text())
    assert metadata["map_epochs"] == 2
    assert metadata["map_cadence_days"] == 1
    assert metadata["method"]["rays"] == 128
    assert metadata["dynamic_schedule"] is None  # no invented configuration
    assert metadata["steady_seconds"] is None  # profiling was not enabled
    assert metadata["timing_components_seconds"] == {}


@pytest.mark.parametrize(
    "module_name",
    [
        "generate_q2237_training_set",
        "generate_random_training_set",
    ],
)
@pytest.mark.parametrize("seed_args, expected", [([], 0), (["--seed", "23"], 23)])
def test_training_command_seed_defaults(monkeypatch, module_name, seed_args, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    monkeypatch.setattr("sys.argv", [module_name, *seed_args])
    command = importlib.import_module(module_name)
    assert command._parse_args().seed == expected


def test_training_helpers_use_unified_api_and_record_requested_numerics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from training_set_support import (
        TrainingSystem,
        generate_labeled_example,
        generate_labeled_examples,
    )

    system = TrainingSystem(small_system(), {})
    map_times = torch.tensor([0.0, 1.0])
    flux_times = torch.tensor([0.0, 0.5, 1.0])
    single, centers, _ = generate_labeled_example(
        system,
        map_times,
        flux_times,
        rays=64,
    )
    group, report = generate_labeled_examples(
        (system, system),
        map_times,
        flux_times,
        rays=64,
        curves_per_batch=2,
    )
    assert len(report.light_curves) == 2
    for result, values, elapsed in group:
        torch.testing.assert_close(result.flux, single.flux)
        # Public epoch axes may be promoted to float64 for time registration.
        torch.testing.assert_close(result.labels.times_days, map_times, check_dtype=False)
        torch.testing.assert_close(result.times_days, flux_times, check_dtype=False)
        np.testing.assert_allclose(values, centers)
        assert result.metadata["training_method"]["rays"] == 64
        assert result.metadata["training_dynamic_schedule"]["temporal_batch_size"] == 30
        assert elapsed >= 0
    assert single.metadata["training_method"]["rays"] == 64


def test_source_driver_builds_the_same_kerr_lamppost_source_as_explicit_attachment():
    model = mc.KerrDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        bands_angstrom={"g": 4800},
        source_grid_shape=12,
        compile_solver=False,
        lamppost_nalpha=16,
        lamppost_radial_bins=16,
    )
    driver = mc.broken_power_law_driving_signal(max_duration_days=20)
    system = small_system(source=model.with_driving_signal(driver))
    realized = system.realize()
    expected = model.with_driving_signal(system._bound_driving_signal).pixelate(
        system.distances, grid=realized.source_grid, runtime=realized.simulation.runtime
    )
    assert isinstance(realized.source, mc.ThermalReprocessingSource)
    torch.testing.assert_close(realized.source.transfer.hit, expected.transfer.hit)
    assert bool(
        torch.all(
            torch.isfinite(realized.source.response_temperature4[expected.transfer.hit])
        )
    )
    torch.testing.assert_close(
        realized.source.response_temperature4,
        expected.response_temperature4,
        equal_nan=True,
    )
    torch.testing.assert_close(
        realized.source.delay_days, expected.delay_days, equal_nan=True
    )
    torch.testing.assert_close(
        realized.source.brightness([0, 1]), expected.brightness([0, 1])
    )


@pytest.mark.parametrize("mode", ["single", "multi", "batch"])
def test_unknown_solver_controls_fail_before_realizing_any_system(mode):
    single = small_system()
    multi = mc.MultiImageSystem(images={"A": single})
    calls = {
        "single": lambda: single.light_curve((0, 1), temporal_batch_sze=7),
        "multi": lambda: multi.light_curves((0, 1), temporal_batch_sze=7),
        "batch": lambda: mc.batched_system_light_curves(
            [single], (0, 1), temporal_batch_sze=7
        ),
    }
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(TypeError, match="temporal_batch_sze"):
            calls[mode]()
        realize.assert_not_called()


@pytest.mark.parametrize("apply_driver", [None, True, False])
def test_multi_image_source_override_uses_its_own_driver_like_single(
    apply_driver,
):
    base = small_system()
    driver = mc.TabulatedDrivingSignal(
        torch.tensor([-10.0, 10.0]), torch.tensor([2.0, 2.0])
    )
    previous = mc.CallableDrivingSignal(lambda t: torch.full_like(t, 3.0))
    single = base.with_source(mc.ModulatedSource(base.source, previous))
    multi = mc.MultiImageSystem(images={"A": single})
    attached = mc.ModulatedSource(base.source, driver)
    options = dict(
        source=attached, apply_driving_signal=apply_driver, **small_options()
    )
    expected = single.light_curve((0, 1), **options)
    actual = multi.light_curves((0, 1), **options)["A"]
    torch.testing.assert_close(actual.flux, expected.flux, rtol=0, atol=0)
    baseline = base.light_curve((0, 1), **small_options())
    torch.testing.assert_close(
        actual.flux, baseline.flux * (1 if apply_driver is False else 2)
    )


def test_multi_image_source_replacement_removes_or_replaces_the_old_driver():
    base = small_system()
    driver = mc.broken_power_law_driving_signal(
        max_duration_days=10, history_days=2, seed=33
    )
    source = mc.ModulatedSource(base.source, driver)
    multi = mc.MultiImageSystem(images={"A": base}, source=source)
    original = multi.light_curves((0, 1), **small_options())["A"]
    copied = multi.with_arrival_time_delays({"A": 0})
    torch.testing.assert_close(
        copied.light_curves((0, 1), **small_options())["A"].flux, original.flux
    )
    replacement = mc.ModulatedSource(
        base.source, mc.broken_power_law_driving_signal(seed=12)
    )
    changed = multi.light_curves((0, 1), source=replacement, **small_options())["A"]
    assert not torch.equal(changed.flux, original.flux)
    plain = multi.with_source(base.source)
    assert plain._shared_driving_signal is None
    assert plain.image("A")._bound_driving_signal is None
    torch.testing.assert_close(
        plain.light_curves((0, 1), **small_options())["A"].flux,
        base.light_curve((0, 1), **small_options()).flux,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"times_days": [1, 0]},
        {"times_days": [0, 1], "flux_times_days": [0, 2]},
        {"times_days": {"A": [0, 1], "missing": [0, 1]}},
        {"times_days": [0, 1], "keep_maps_at_days": [float("nan")]},
    ],
)
def test_multi_image_invalid_times_and_retention_fail_before_setup(invalid):
    multi = mc.MultiImageSystem(images={"A": small_system()})
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError):
            multi.light_curves(**invalid)
        realize.assert_not_called()


def test_multi_image_regular_and_per_image_times_are_equivalent():
    multi = mc.MultiImageSystem(images={"A": small_system(), "B": small_system()})
    options = dict(source_cadence_days=0.5, **small_options())
    regular = multi.light_curves(duration_days=2, map_cadence_days=1, **options)
    mapped = multi.light_curves({"A": [0, 1, 2], "B": [0, 1, 2]}, **options)
    for name in multi.image_names:
        torch.testing.assert_close(
            regular[name].flux, mapped[name].flux, rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_labelled_archive_round_trip_preserves_sparse_center_series(tmp_path, dtype):
    labels = mc.LightCurveLabels(
        times_days=torch.tensor([0, 2], dtype=dtype),
        crossing_labels=torch.tensor([0, 1], dtype=torch.int8),
        crossing_events=torch.tensor([False, True]),
        center_distances_uas=torch.tensor([0.3, 0.5], dtype=dtype),
        center_distance_censored=torch.tensor([False, True]),
    )
    curve = mc.LightCurve(
        torch.tensor([0, 1, 2], dtype=dtype),
        torch.ones(3, 1, dtype=dtype),
        ("g",),
        labels=labels,
    )
    restored = mc.load_light_curve(mc.save_light_curve(curve, tmp_path / "curve.npz"))
    torch.testing.assert_close(restored.flux, curve.flux, rtol=0, atol=0)
    for name in (
        "times_days",
        "crossing_labels",
        "crossing_events",
        "center_distances_uas",
        "center_distance_censored",
    ):
        torch.testing.assert_close(
            getattr(restored.labels, name), getattr(labels, name), rtol=0, atol=0
        )
    assert restored.labels.caustics is None and not restored.maps
    # Missing schema metadata is rejected instead of guessing an old format.
    with np.load(tmp_path / "curve.npz") as data:
        no_schema = {
            key: data[key]
            for key in data.files
            if key != "schema_version"
        }
    np.savez(tmp_path / "no_schema.npz", **no_schema)
    with pytest.raises(ValueError, match="unsupported.*schema_version"):
        mc.load_light_curve(tmp_path / "no_schema.npz")
    broken = {
        key: value
        for key, value in no_schema.items()
        if not key.startswith("labels_") and key != "has_labels"
    }
    np.savez(
        tmp_path / "broken.npz",
        **broken,
        schema_version=np.asarray(1),
        has_labels=True,
    )
    with pytest.raises(ValueError, match="missing label arrays"):
        mc.load_light_curve(tmp_path / "broken.npz")


def test_macroimage_archive_keeps_the_same_labels_as_its_public_view(tmp_path):
    labels = mc.LightCurveLabels.from_caustics(
        (
            SimpleNamespace(
                time_days=0.0,
                labels=SimpleNamespace(
                    center_label=1,
                    center_crossing=False,
                    center_distance_uas=0.2,
                    center_distance_censored=False,
                ),
            ),
        )
    )
    curve = mc.LightCurve(torch.tensor([0.0]), torch.ones(1, 1), ("g",))
    image = mc.MacroImageLightCurve("A", 0.0, curve, caustics=labels.caustics)
    assert image.labels is image.light_curve.labels
    assert image.crossing_labels is image.labels.crossing_labels
    loaded = mc.load_light_curve(
        mc.save_light_curve(image.light_curve, tmp_path / "image.npz")
    )
    torch.testing.assert_close(loaded.labels.crossing_labels, image.crossing_labels)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_driver_table_cache_matches_numpy_and_keeps_autograd(dtype):
    times = torch.linspace(-1, 1, 101, dtype=torch.float64)
    values = torch.exp(times)
    driver = mc.TabulatedDrivingSignal(times, values)
    query = torch.linspace(-0.9, 0.9, 31, dtype=dtype)
    first = driver.amplitudes(query, bands=1, dtype=dtype, device="cpu")
    table = driver._device_tables[(torch.device("cpu"), dtype)]
    second = driver.amplitudes(query, bands=1, dtype=dtype, device="cpu")
    assert driver._device_tables[(torch.device("cpu"), dtype)] is table
    np.testing.assert_allclose(
        first[:, 0].numpy(),
        np.interp(query.numpy(), times.numpy(), values.numpy()),
        rtol=2e-7,
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    trainable_values = values.clone().requires_grad_()
    differentiable = mc.TabulatedDrivingSignal(times, trainable_values)
    for _ in range(2):
        differentiable.amplitudes(
            query, bands=1, dtype=dtype, device="cpu"
        ).sum().backward()
    assert not differentiable._device_tables
    assert torch.isfinite(trainable_values.grad).all()


def test_partial_per_image_controls_keep_other_images_configuration():
    multi = mc.MultiImageSystem(
        images={"A": small_system(), "B": small_system()},
        methods=small_options()["method"],
    )
    simulation, _ = multi._build_simulation((0, 1), {"rays": {"A": 144}})
    assert [image.method.rays for image in simulation.images] == [144, 64]


def test_shared_unseeded_driver_is_sampled_only_once_across_threads():
    driver = mc.broken_power_law_driving_signal(max_duration_days=2, history_days=1)
    with (
        patch.object(
            type(driver),
            "_generate_samples",
            autospec=True,
            side_effect=type(driver)._generate_samples,
        ) as generate,
        ThreadPoolExecutor(max_workers=4) as workers,
    ):
        samples = list(workers.map(lambda _: driver._samples, range(8)))
    assert generate.call_count == 1
    assert all(value is samples[0] for value in samples)


@pytest.mark.parametrize(
    "option", ["cadence_days", "max_duration_days", "history_days"]
)
def test_explicit_driver_times_reject_automatic_horizon_controls(option):
    with pytest.raises(ValueError, match="not both"):
        mc.broken_power_law_driving_signal(torch.arange(10.0), **{option: 1.0})


def test_horizon_validation_precedes_float32_rounding():
    driver = mc.broken_power_law_driving_signal(max_duration_days=10, history_days=1)
    for time in (10 + 1e-8, -1 - 1e-8):
        with pytest.raises(ValueError, match="must lie between"):
            driver.amplitudes(
                torch.tensor([time], dtype=torch.float64),
                bands=1,
                dtype=torch.float32,
                device="cpu",
            )
    assert driver._sampled is None


def test_unlabeled_default_does_not_construct_caustic_settings():
    from microcaustics.system import _production_dynamic_settings

    with patch("microcaustics.system.CausticConfig") as caustics:
        schedule, labels = _production_dynamic_settings(
            mc.production_ipm_config(), None, include_labels=False
        )
    caustics.assert_not_called()
    assert labels is None and schedule.temporal_batch_size == 49


def test_invalid_batched_retention_fails_before_realization():
    systems = [small_system(), small_system()]
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError, match="finite"):
            mc.batched_system_light_curves(
                systems, (0, 1), keep_maps_at_days=[float("nan")]
            )
        with pytest.raises(ValueError, match="number of systems"):
            mc.batched_system_light_curves(systems, (0, 1), map_observers=[None])
        realize.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_driver_device_cache_is_safe_across_cuda_streams():
    driver = mc.broken_power_law_driving_signal(
        seed=51, max_duration_days=10, history_days=2
    )
    times = torch.linspace(0, 10, 41)
    expected = driver.amplitudes(times, bands=2, dtype=torch.float32, device="cpu")
    streams = [torch.cuda.Stream() for _ in range(3)]

    def evaluate(stream):
        with torch.cuda.stream(stream):
            result = driver.amplitudes(
                times, bands=2, dtype=torch.float32, device="cuda"
            )
        stream.synchronize()
        return result.cpu()

    with ThreadPoolExecutor(max_workers=3) as workers:
        actual = list(workers.map(evaluate, streams))
    for value in actual:
        torch.testing.assert_close(value, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("mode", ["single", "multi", "batch"])
@pytest.mark.parametrize("has_driver", [False, True])
@pytest.mark.parametrize("apply_driver", [None, False, True])
def test_source_driver_switch_contract(mode, has_driver, apply_driver):
    """All public schedulers use the same opt-in rules before doing work."""

    driver = mc.CallableDrivingSignal(lambda t: torch.full_like(t, 2.0))
    base = small_system()
    system = small_system(signal=driver) if has_driver else base
    multi = mc.MultiImageSystem(images={"A": system})
    options = dict(apply_driving_signal=apply_driver, **small_options())
    calls = {
        "single": lambda: system.light_curve((0, 1), **options),
        "multi": lambda: multi.light_curves((0, 1), **options)["A"].light_curve,
        "batch": lambda: mc.batched_system_light_curves(
            [system], (0, 1), **options
        ).light_curves[0],
    }
    if apply_driver is True and not has_driver:
        with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
            with pytest.raises(
                ValueError, match="source with a configured driving signal"
            ):
                calls[mode]()
            realize.assert_not_called()
        return
    result = calls[mode]()
    expected = base.light_curve((0, 1), **small_options())
    scale = 2 if has_driver and apply_driver is not False else 1
    torch.testing.assert_close(result.flux, scale * expected.flux)


@pytest.mark.parametrize("mode", ["single", "multi", "batch"])
def test_supernova_without_driver_keeps_its_intrinsic_evolution(mode):
    """Disabling a nonexistent driver must not freeze a transient source."""

    source = mc.paper_type_ia_supernova_source(
        redshift=1.5,
        wavelengths_angstrom=(4800,),
        band_names=("g",),
        maximum_observer_time_days=40,
        source_grid_shape=8,
    )
    system = small_system(source=source)
    multi = mc.MultiImageSystem(images={"A": system})

    def run(apply):
        options = dict(apply_driving_signal=apply, **small_options())
        if mode == "single":
            return system.light_curve((10, 20), **options)
        if mode == "multi":
            return multi.light_curves((10, 20), **options)["A"].light_curve
        return mc.batched_system_light_curves(
            [system], (10, 20), **options
        ).light_curves[0]

    normal, off = run(None), run(False)
    torch.testing.assert_close(normal.flux, off.flux, rtol=0, atol=0)
    assert not torch.equal(off.unlensed_flux[0], off.unlensed_flux[1])
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError, match="configured driving signal"):
            run(True)
        realize.assert_not_called()


@pytest.mark.parametrize("multi_image", [False, True])
def test_driver_free_override_never_inherits_old_source_driver(multi_image):
    base = small_system()
    driven = small_system(
        signal=mc.CallableDrivingSignal(lambda t: 2 * torch.ones_like(t))
    )
    if multi_image:
        system = mc.MultiImageSystem(images={"A": driven})
        call = system.light_curves
    else:
        system = driven
        call = system.light_curve
    options = dict(source=base.source, **small_options())
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError, match="configured driving signal"):
            call((0, 1), apply_driving_signal=True, **options)
        realize.assert_not_called()
    actual = call((0, 1), **options)
    if multi_image:
        actual = actual["A"].light_curve
    torch.testing.assert_close(
        actual.flux, base.light_curve((0, 1), **small_options()).flux
    )


def test_system_constructors_no_longer_accept_drivers():
    for constructor, kwargs in (
        (mc.MicrolensingSystem, dict(macro=mc.MacroLens(0, 0))),
        (mc.MultiImageSystem, dict(images={"A": small_system()})),
    ):
        with pytest.raises(TypeError, match="driving_signal"):
            constructor(driving_signal=mc.broken_power_law_driving_signal(), **kwargs)
    with pytest.raises(TypeError, match="driving_signal"):
        mc.ExpandingPhotosphereSource(
            driving_signal=mc.broken_power_law_driving_signal()
        )


def test_kerr_disk_driver_is_optional_and_invalid_driver_is_rejected():
    settings = dict(
        black_hole_mass_solar=1e8, eddington_ratio=0.1, bands_angstrom={"g": 4800}
    )
    disk = mc.KerrDiskModel(**settings)
    assert disk.driving_signal is None
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError, match="configured driving signal"):
            small_system(source=disk).light_curve((0,), apply_driving_signal=True)
        realize.assert_not_called()
    with pytest.raises(TypeError, match="amplitudes and metadata"):
        mc.KerrDiskModel(**settings, driving_signal=lambda t: t)


def test_shared_unseeded_multi_image_driver_survives_delay_updates():
    base = small_system()
    driver = mc.broken_power_law_driving_signal(max_duration_days=10, history_days=2)
    multi = mc.MultiImageSystem(
        images={"A": base, "B": base}, source=mc.ModulatedSource(base.source, driver)
    )
    shared = multi._shared_driving_signal
    updated = multi.with_arrival_time_delays({"B": 1.0})
    assert updated._shared_driving_signal is shared
    assert updated.image("A")._bound_driving_signal is shared
    assert updated.image("B")._bound_driving_signal is shared


def test_batched_mixed_sources_reject_explicit_driving_before_any_realization():
    systems = [
        small_system(signal=mc.broken_power_law_driving_signal()),
        small_system(),
    ]
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError, match="configured driving signal"):
            mc.batched_system_light_curves(systems, (0, 1), apply_driving_signal=True)
        realize.assert_not_called()


def test_warmup_rejects_driving_for_a_source_without_a_driver():
    with patch.object(mc.MicrolensingSystem, "_realize_for_times") as realize:
        with pytest.raises(ValueError, match="configured driving signal"):
            small_system().warmup_light_curve(apply_driving_signal=True)
        realize.assert_not_called()


def test_custom_source_owns_its_response_and_accepts_bound_or_mean_driver():
    """A user model can define a non-multiplicative response without a wrapper."""

    @dataclass(frozen=True)
    class CustomResponse:
        base: mc.StaticSource
        driving_signal: object = None
        is_time_static: bool = False

        @property
        def geometry(self):
            return self.base.geometry

        def with_driving_signal(self, signal):
            return replace(self, driving_signal=signal)

        def brightness(self, times_days, *, device=None, dtype=None):
            times = torch.as_tensor(times_days, device=device, dtype=dtype).reshape(-1)
            amplitude = torch.ones((times.numel(), 1), device=device, dtype=times.dtype)
            if self.driving_signal is not None:
                amplitude = self.driving_signal.amplitudes(
                    times, bands=1, device=times.device, dtype=times.dtype
                )
            # This toy source responds quadratically and independently evolves.
            scale = (1 + 0.01 * times[:, None]) * amplitude.square()
            return (
                self.base.brightness(times, device=device, dtype=dtype)
                * scale[:, None, None]
            )

        def metadata(self):
            return {"type": "custom_quadratic_response"}

    base = small_system()
    driver = mc.broken_power_law_driving_signal(max_duration_days=5, history_days=1)
    source = CustomResponse(base.source, driver)
    system = base.with_source(source)
    off = system.light_curve((0, 1), apply_driving_signal=False, **small_options())
    assert system._bound_driving_signal._sampled is None
    on = system.light_curve((0, 1), **small_options())
    amplitude = system._bound_driving_signal.amplitudes(
        on.times_days, bands=1, device=on.flux.device, dtype=on.flux.dtype
    )
    torch.testing.assert_close(on.flux, off.flux * amplitude.square())
    baseline = base.with_source(CustomResponse(base.source)).light_curve(
        (0, 1), **small_options()
    )
    torch.testing.assert_close(off.flux, baseline.flux)
    assert not torch.equal(off.unlensed_flux[0], off.unlensed_flux[1])


def test_prebuilt_macroimages_reject_different_intrinsic_drivers():
    first = small_system(signal=mc.broken_power_law_driving_signal(seed=1))
    second = small_system(signal=mc.broken_power_law_driving_signal(seed=2))
    with pytest.raises(ValueError, match="share one source driving signal"):
        mc.MultiImageSystem(images={"A": first, "B": second})
    # An explicit shared source intentionally replaces both image sources.
    shared = mc.MultiImageSystem(images={"A": first, "B": second}, source=first.source)
    assert (
        shared.image("A")._bound_driving_signal
        is shared.image("B")._bound_driving_signal
    )


def test_multi_image_reseed_changes_only_implicitly_seeded_driver():
    base = small_system()
    for explicit_seed in (None, 31):
        driver = mc.broken_power_law_driving_signal(
            seed=explicit_seed, max_duration_days=4
        )
        multi = mc.MultiImageSystem(
            images={"A": base, "B": base},
            source=mc.ModulatedSource(base.source, driver),
            seed=11,
        )
        changed = replace(multi, seed=12)
        if explicit_seed is None:
            assert (
                multi._shared_driving_signal.seed != changed._shared_driving_signal.seed
            )
        else:
            assert (
                multi._shared_driving_signal.seed
                == changed._shared_driving_signal.seed
                == 31
            )
