"""Small CPU regressions for portable far-field query memory budgeting."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from microcaustics.solvers import far_field

KERNELS = [
    far_field._evaluate_far_field_deflection,
    far_field._evaluate_far_field_jacobian,
]


def _query_case(count, width, dtype):
    approximation = far_field.TaylorFarFieldApproximation.__new__(
        far_field.TaylorFarFieldApproximation
    )
    approximation.simulation = SimpleNamespace(
        runtime=SimpleNamespace(device=torch.device("cpu"))
    )
    stars = torch.arange(2 * width, dtype=dtype).reshape(2, width)
    approximation.local_x = 3.0 + stars * 0.1
    approximation.local_y = 2.0 + stars * 0.2
    approximation.local_mass = 0.5 + stars * 0.03
    cell = torch.arange(count) % 2
    x = torch.linspace(-0.5, 0.5, count, dtype=dtype)
    y = 0.2 * x + 0.1
    # Nonzero, query-dependent far-field coefficients catch omission or
    # duplication of the Taylor term as well as ordering/padding mistakes.
    coefficients = torch.arange(count * 5, dtype=dtype).reshape(count, 5) * 0.01
    per_query = (coefficients, -0.3 * coefficients, x + 0.1, y - 0.2, x, y)
    minimum = torch.tensor(1.0e-30, dtype=dtype)
    return approximation, per_query, cell, minimum


@pytest.mark.parametrize("kernel", KERNELS, ids=["deflection", "jacobian"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [0, 4])
@pytest.mark.parametrize("count", [0, 1, 3, 4, 7])
def test_portable_query_chunks_are_bounded_padded_and_numerically_equal(
    monkeypatch, kernel, dtype, width, count
):
    approximation, per_query, cell, minimum = _query_case(count, width, dtype)
    bytes_per_query = 6 * approximation.local_x.element_size() * max(1, width)
    rows = 3
    budget = rows * bytes_per_query + bytes_per_query - 1
    monkeypatch.setattr(far_field, "_PORTABLE_QUERY_BYTES", budget)
    monkeypatch.setattr(far_field, "portable_accelerator", lambda device: True)
    calls = []

    def run_eager(runtime, name, function, *args):
        assert runtime.device.type == "cpu"
        assert args[0].shape[0] * bytes_per_query <= budget
        calls.append(args)
        return function(*args), False

    monkeypatch.setattr(far_field, "run_tensor_kernel", run_eager)
    expected = kernel(
        *per_query,
        approximation.local_x[cell],
        approximation.local_y[cell],
        approximation.local_mass[cell],
        minimum,
    )
    actual, compiled = approximation._local_query_kernel(
        "budget regression", kernel, per_query, cell, minimum
    )
    assert not compiled
    assert [args[0].shape[0] for args in calls] == (
        [count] if count <= rows else [rows] * ((count + rows - 1) // rows)
    )
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == (count,)
        torch.testing.assert_close(value, reference, rtol=0.0, atol=0.0)
    for chunk, args in enumerate(calls):
        if count <= rows:
            index = torch.arange(count)
        else:
            index = torch.arange(chunk * rows, (chunk + 1) * rows).clamp_max(count - 1)
        # Inspect all inputs, including repeated final queries and star packs;
        # the output must discard padding rather than duplicating its results.
        for value, original in zip(args[:6], per_query, strict=True):
            torch.testing.assert_close(value, original[index], rtol=0.0, atol=0.0)
        for value, pack in zip(
            args[6:9],
            (approximation.local_x, approximation.local_y, approximation.local_mass),
            strict=True,
        ):
            torch.testing.assert_close(value, pack[cell[index]], rtol=0.0, atol=0.0)


def test_oversized_single_query_fails_before_gathering(monkeypatch):
    approximation, per_query, cell, minimum = _query_case(1, 4, torch.float32)
    bytes_per_query = 6 * 4 * 4
    monkeypatch.setattr(far_field, "_PORTABLE_QUERY_BYTES", bytes_per_query - 1)
    monkeypatch.setattr(far_field, "portable_accelerator", lambda device: True)
    packs = []
    for name in ("local_x", "local_y", "local_mass"):
        pack = MagicMock()
        pack.shape = (2, 4)
        pack.element_size.return_value = 4
        pack.__getitem__.side_effect = AssertionError("must fail before gathering")
        setattr(approximation, name, pack)
        packs.append(pack)
    run = MagicMock(side_effect=AssertionError("must fail before kernel dispatch"))
    monkeypatch.setattr(far_field, "run_tensor_kernel", run)
    with pytest.raises(
        MemoryError,
        match=r"single far-field query.*width 4.*estimated 96.*budget of 95",
    ):
        approximation._local_query_kernel(
            "budget regression", KERNELS[0], per_query, cell, minimum
        )
    run.assert_not_called()
    for pack in packs:
        pack.__getitem__.assert_not_called()


@pytest.mark.parametrize("kernel", KERNELS, ids=["deflection", "jacobian"])
@pytest.mark.parametrize("portable,count", [(True, 0), (False, 4)])
def test_oversized_pack_allows_empty_queries_and_nonportable_runtime(
    monkeypatch, kernel, portable, count
):
    approximation, per_query, cell, minimum = _query_case(count, 4, torch.float32)
    monkeypatch.setattr(far_field, "_PORTABLE_QUERY_BYTES", 1)
    monkeypatch.setattr(far_field, "portable_accelerator", lambda device: portable)
    run = MagicMock(side_effect=lambda runtime, name, function, *args: (function(*args), False))
    monkeypatch.setattr(far_field, "run_tensor_kernel", run)
    actual, compiled = approximation._local_query_kernel(
        "budget regression", kernel, per_query, cell, minimum
    )
    expected = kernel(
        *per_query,
        approximation.local_x[cell],
        approximation.local_y[cell],
        approximation.local_mass[cell],
        minimum,
    )
    assert not compiled
    run.assert_called_once()
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0.0, atol=0.0)
