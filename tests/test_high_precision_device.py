"""Auxiliary device routing preserves precision and main runtime placement."""

from dataclasses import asdict, replace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc
from microcaustics.runtime import _resolve_high_precision_device


@pytest.mark.parametrize(
    "choice", [None, "cpu", "cuda", "cuda:1", "same", torch.device("cpu")]
)
def test_runtime_preserves_auxiliary_choice(choice):
    config = mc.RuntimeConfig(
        device="cpu", backend="torch-eager", high_precision_device=choice
    )
    runtime = mc.resolve_runtime(config)
    assert runtime.high_precision_device == (None if choice is None else str(choice))
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32


@pytest.mark.parametrize("choice", ["auto", "mps", "cpu:0", "cuda:-1", "bogus"])
def test_invalid_auxiliary_choice(choice):
    with pytest.raises(ValueError, match="high_precision_device"):
        mc.RuntimeConfig(high_precision_device=choice)


def test_defaults_overrides_and_unavailable_devices():
    config = mc.RuntimeConfig(device="cuda")
    assert _resolve_high_precision_device(config) == torch.device("cpu")
    assert _resolve_high_precision_device(default="cpu") == torch.device("cpu")
    config = replace(config, high_precision_device="cuda")
    with patch("torch.cuda.is_available", return_value=False):
        assert _resolve_high_precision_device(config, device="cpu") == torch.device(
            "cpu"
        )
        with pytest.raises(RuntimeError, match="not available"):
            _resolve_high_precision_device(config)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        pytest.raises(RuntimeError, match="not available"),
    ):
        _resolve_high_precision_device(
            replace(config, high_precision_device="cuda:3")
        )
    assert _resolve_high_precision_device(
        mc.RuntimeConfig(device="cpu", high_precision_device="same")
    ) == torch.device("cpu")
    with pytest.raises(RuntimeError, match="float64"):
        _resolve_high_precision_device(
            mc.RuntimeConfig(device="mps", high_precision_device="same")
        )


def test_macro_solver_inherits_runtime_but_not_main_dtype():
    caustics = pytest.importorskip("caustics")
    model = caustics.ExternalShear(
        caustics.FlatLambdaCDM(), z_l=0.5, z_s=1.5,
        x0=0.0, y0=0.0, gamma_1=0.0, gamma_2=0.0,
    ).to(dtype=torch.float64, device="cpu")
    runtime = mc.RuntimeConfig(device="cuda", high_precision_device="cpu")
    with patch.object(model, "forward_raytrace", wraps=model.forward_raytrace) as native:
        result = mc.solve_caustics_macroimages(
            model, 0.1, 0.05, runtime=runtime, initial_grid_size=21
        )
        beta_x = native.call_args.args[0]
        assert beta_x.dtype == torch.float64 and beta_x.device.type == "cpu"
    assert len(result) == 1
    assert result[0].x_arcsec == pytest.approx(0.1)
    result = mc.solve_caustics_macroimages(
        model,
        0.1,
        0.05,
        runtime=replace(runtime, high_precision_device="cuda"),
        device="cpu",
        initial_grid_size=21,
    )
    assert len(result) == 1


def test_epl_builder_inherits_cpu_override():
    pytest.importorskip("caustics")
    config = mc.EPLShearConfig(
        lens_redshift=0.5, source_redshift=1.0, einstein_radius_arcsec=0.7
    )
    runtime = mc.RuntimeConfig(device="cuda", high_precision_device="cpu")
    result = mc.solve_epl_shear_macroimages(
        config,
        0.1,
        0.0,
        runtime=runtime,
        initial_grid_size=31,
        field_of_view_arcsec=1.2,
    )
    assert sorted(x.x_arcsec for x in result) == pytest.approx([-0.6, 0.8], abs=1e-6)


def test_validation_preserves_float64_list_precision():
    # Converting this list to float32 before selecting the device loses its signal.
    values = [1.0 + 1e-10, 1.0 + 2e-10]
    runtime = mc.RuntimeConfig(device="cuda", high_precision_device="cpu")
    reference = torch.ones(2, dtype=torch.float64)
    expected = mc.compare_magnification_maps(
        torch.tensor(values, dtype=torch.float64), reference
    )
    actual = mc.compare_magnification_maps(values, reference, runtime=runtime)
    assert actual == expected
    assert actual.fractional_nrmse > 0
    assert mc.compare_light_curves(values, reference, runtime=runtime).rmse_mmag > 0


def small_kerr(**kwargs):
    return mc.KerrDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        bands_angstrom={"g": 4800},
        source_grid_shape=16,
        compile_solver=False,
        warn_on_compile=False,
        lamppost_nalpha=32,
        lamppost_radial_bins=32,
        driving_signal=mc.TabulatedDrivingSignal([-100.0, 100.0], [1.0, 1.0]),
        **kwargs,
    )


@pytest.mark.parametrize("override", [None, "cpu", "same"])
def test_kerr_inheritance_and_per_model_override(override):
    runtime = mc.RuntimeConfig(
        device="cpu",
        backend="torch-eager",
        high_precision_device="cpu" if override is None else "cuda",
    )
    model = small_kerr(observer_coordinate_repair_device=override)
    source = model.pixelate(source_redshift=1.5, runtime=runtime)
    assert source.transfer.metadata["coordinate_repair_device"] == "cpu"
    assert source.transfer.radius_rg.device.type == "cpu"
    assert source.transfer.radius_rg.dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_validation_gpu_cpu_agreement_and_input_preservation():
    candidate = torch.tensor([1.1, 2.01, 3.2, 4.1], device="cuda")
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    saved = candidate.clone()
    for function in (mc.compare_magnification_maps, mc.compare_light_curves):
        expected = function(candidate, reference)
        actual = function(
            candidate, reference, runtime=mc.RuntimeConfig(high_precision_device="cpu")
        )
        assert asdict(actual) == pytest.approx(asdict(expected), rel=1e-12, abs=1e-12)
        assert function(candidate.cpu(), reference.cpu(), device="cuda") == expected
    torch.testing.assert_close(candidate, saved)
    assert candidate.device.type == "cuda" and candidate.dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kerr_float64_repair_can_run_on_gpu_for_cpu_source():
    screen = mc.ObserverScreen.uniform(
        (32, 32),
        (60.0, 60.0),
        gravitational_radius_m=1.0,
        observer_distance_m=1e6,
        dtype=torch.float32,
    )
    primary = mc.trace_primary_equatorial(
        screen, spin=0.7, inclination_deg=53.0, disk_outer_rg=50.0
    )
    kwargs = dict(black_hole_mass_solar=1e9, spin=0.7, inclination_deg=53.0)
    cpu = mc.add_observer_coordinates(primary, screen, repair_device="cpu", **kwargs)
    gpu = mc.add_observer_coordinates(primary, screen, repair_device="cuda", **kwargs)
    assert gpu.repaired_pixels > 0
    assert gpu.failed_pixels == cpu.failed_pixels == 0
    assert gpu.transfer.radius_rg.device.type == "cpu"
    assert gpu.transfer.relative_delay_days.dtype == torch.float32
    torch.testing.assert_close(
        gpu.transfer.relative_delay_days,
        cpu.transfer.relative_delay_days,
        equal_nan=True,
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_kerr_repairs_follow_shared_setting():
    runtime = mc.RuntimeConfig(
        device="cuda",
        backend="torch-eager",
        high_precision_device="cuda",
        warn_on_compile=False,
    )
    distances = mc.LensingDistances.from_redshifts(0.3, 1.5)
    models = (small_kerr(spin=0.7), small_kerr(spin=0.4))
    actual = mc.batched_pixelate_sources(
        models, distances, runtime=runtime, batch_size=2
    )
    expected = tuple(
        model.pixelate(distances, runtime=replace(runtime, high_precision_device="cpu"))
        for model in models
    )
    for candidate, reference in zip(actual, expected, strict=True):
        assert candidate.transfer.metadata["coordinate_repair_device"] == "cuda"
        assert candidate.transfer.metadata["coordinate_cross_disk_pooled"]
        assert candidate.delay_days.device.type == "cuda"
        assert candidate.delay_days.dtype == torch.float32
        torch.testing.assert_close(
            candidate.delay_days,
            reference.delay_days,
            equal_nan=True,
            rtol=2e-5,
            atol=2e-3,
        )
