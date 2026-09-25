"""Release-facing validation and honest fallback contracts."""

from __future__ import annotations

import warnings
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc
from microcaustics.batching import batched_magnification_maps
from microcaustics.caustics.anchor_gauge import (
    anchor_gauge_label_map,
    label_caustic_fields,
)
from microcaustics.caustics.direct import _extract_zero_segments
from microcaustics.compile import clear_compiled_kernel_cache, run_tensor_kernel
from microcaustics.runtime import _WARNED_RUNTIME_DOWNGRADES


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_lens_inputs_reject_nonfinite_values(bad: float) -> None:
    with pytest.raises(ValueError, match="convergence"):
        mc.MacroLens(convergence=bad, shear=0.1)
    with pytest.raises(ValueError, match="shear"):
        mc.MacroLens(convergence=0.1, shear=bad)
    with pytest.raises(ValueError, match="distances"):
        mc.LensingDistances(bad, 2.0, 1.0)
    with pytest.raises(ValueError, match="lens_redshift"):
        mc.LensingDistances(1.0, 2.0, 1.0, lens_redshift=bad)
    with pytest.raises(ValueError, match="source_redshift"):
        mc.LensingDistances(1.0, 2.0, 1.0, lens_redshift=0.5, source_redshift=bad)
    with pytest.raises(ValueError, match="source_redshift"):
        mc.LensingDistances.from_redshifts(0.5, bad)

    one = torch.tensor([1.0])
    bad_tensor = torch.tensor([bad])
    with pytest.raises(ValueError, match="positions"):
        mc.PointMassField(bad_tensor, one, mass_solar=one)
    with pytest.raises(ValueError, match="mass_solar"):
        mc.PointMassField(one, one, mass_solar=bad_tensor)
    with pytest.raises(ValueError, match="Einstein radii"):
        mc.PointMassField._from_einstein_radii(
            one, one, einstein_radius_uas=bad_tensor
        )
    with pytest.raises(ValueError, match="velocities"):
        mc.PointMassField(
            one,
            one,
            mass_solar=one,
            velocity_x_uas_per_day=bad_tensor,
            velocity_y_uas_per_day=one,
        )
    distances = mc.LensingDistances(1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="masses"):
        distances.einstein_radius_uas(bad_tensor)


def test_previously_disabled_compile_still_raises_under_strict_runtime() -> None:
    clear_compiled_kernel_cache()
    with patch("microcaustics.runtime._torch_compile_supported", return_value=True):
        strict = mc.resolve_runtime(
            mc.RuntimeConfig(device="cpu", backend="torch-compile", strict_backend=True)
        )
    permissive = replace(strict, strict_backend=False)

    def kernel(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    with (
        patch("microcaustics.compile.torch.compile", side_effect=RuntimeError("compile broke")),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        value, compiled = run_tensor_kernel(
            permissive, "release regression", kernel, torch.tensor(1.0)
        )
        assert not compiled and value.item() == 2.0
        with pytest.raises(RuntimeError, match="earlier failure"):
            run_tensor_kernel(strict, "release regression", kernel, torch.tensor(1.0))
    clear_compiled_kernel_cache()


def test_explicit_backend_downgrade_warns_once_but_auto_does_not() -> None:
    capabilities = mc.RuntimeCapabilities(
        platform="test",
        torch_version="test",
        cuda_available=False,
        cuda_version=None,
        mps_available=False,
        torch_compile_available=False,
        triton_importable=False,
        gpu_name=None,
        total_device_memory_bytes=None,
    )
    _WARNED_RUNTIME_DOWNGRADES.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            mc.resolve_runtime(
                mc.RuntimeConfig(device="cpu", backend="triton"),
                capabilities=capabilities,
            )
        mc.resolve_runtime(
            mc.RuntimeConfig(device="cpu", backend="auto"),
            capabilities=capabilities,
        )
    assert len(caught) == 1
    assert "Requested backend" in str(caught[0].message)


def test_independent_map_batch_retries_text_cuda_oom_only() -> None:
    attempts: list[int] = []

    def calculate(chunk: tuple[object, ...], _method: object) -> tuple[int, ...]:
        attempts.append(len(chunk))
        if len(chunk) > 1:
            raise RuntimeError("Triton CUDA out of memory")
        return (int(chunk[0]),)

    with (
        patch("microcaustics.batching._validate_compatible_requests"),
        patch("microcaustics.batching._calculate_compatible_batch", calculate),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        output = batched_magnification_maps((0, 1, 2), method=object(), batch_size=3)
    assert output == (0, 1, 2)
    assert attempts == [3, 1, 1, 1]
    assert len(caught) == 1
    assert "batch_size=1" in str(caught[0].message)

    with (
        patch("microcaustics.batching._validate_compatible_requests"),
        patch(
            "microcaustics.batching._calculate_compatible_batch",
            side_effect=ValueError("not an OOM"),
        ),
        pytest.raises(ValueError, match="not an OOM"),
    ):
        batched_magnification_maps((0, 1), method=object())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_strict_triton_caustic_labels_reject_portable_fallback() -> None:
    grid = mc.PlaneGrid((9, 9), (2.0, 2.0))
    segments = torch.tensor(
        [[[-0.5, -0.5], [0.5, -0.5]]],
        device="cuda",
        dtype=torch.float32,
    )
    field = mc.CausticField(segments, segments, grid)
    config = mc.CausticConfig(
        anchor_count=3,
        gauge_count=3,
        minimum_alignment_gauges=1,
    )
    with (
        patch(
            "microcaustics.caustics.triton_caustics.triton_caustics_available",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="strict Triton backend"),
    ):
        label_caustic_fields((field,), grid.region, config, strict_backend=True)

    anchors = torch.zeros((3, 2), device="cuda")
    offsets = torch.zeros(3, device="cuda", dtype=torch.int64)
    with (
        patch(
            "microcaustics.caustics.triton_caustics.triton_caustics_available",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="strict Triton backend"),
    ):
        anchor_gauge_label_map(
            field,
            grid,
            anchors,
            offsets,
            frame_xor=0,
            config=config,
            strict_backend=True,
        )

    with (
        patch(
            "microcaustics.caustics.triton_caustics.triton_caustics_available",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="strict Triton backend"),
    ):
        _extract_zero_segments(
            torch.ones(grid.shape, device="cuda"),
            grid,
            strict_backend=True,
        )
