from __future__ import annotations

import warnings
from unittest.mock import patch

import torch

import microcaustics as mc
from microcaustics.compile import clear_compiled_kernel_cache, run_tensor_kernel


def _compiled_runtime(*, warn_on_compile: bool = True):
    with patch(
        "microcaustics.runtime._torch_compile_supported",
        return_value=True,
    ):
        return mc.resolve_runtime(
            mc.RuntimeConfig(
                device="cpu",
                backend="torch-compile",
                strict_backend=True,
                warn_on_compile=warn_on_compile,
            )
        )


def test_compile_warning_is_enabled_by_default_and_emitted_once() -> None:
    clear_compiled_kernel_cache()
    runtime = _compiled_runtime()

    def kernel(value):
        return value + 1

    with (
        patch("microcaustics.compile.torch.compile", side_effect=lambda fn, **_: fn),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run_tensor_kernel(runtime, "test kernel", kernel, torch.tensor(1.0))
        run_tensor_kernel(runtime, "test kernel", kernel, torch.tensor(2.0))

    compile_warnings = [
        item for item in caught if issubclass(item.category, mc.CompilationWarning)
    ]
    assert len(compile_warnings) == 1
    assert "test kernel" in str(compile_warnings[0].message)
    assert "warn_on_compile=False" in str(compile_warnings[0].message)


def test_compile_warning_can_be_disabled() -> None:
    clear_compiled_kernel_cache()
    runtime = _compiled_runtime(warn_on_compile=False)

    def kernel(value):
        return value + 1

    with (
        patch("microcaustics.compile.torch.compile", side_effect=lambda fn, **_: fn),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run_tensor_kernel(runtime, "quiet kernel", kernel, torch.tensor(1.0))

    assert not any(
        issubclass(item.category, mc.CompilationWarning) for item in caught
    )


def test_fixed_and_dynamic_shape_kernels_have_separate_compiled_wrappers() -> None:
    clear_compiled_kernel_cache()
    runtime = _compiled_runtime(warn_on_compile=False)

    def kernel(value):
        return value + 1

    with patch(
        "microcaustics.compile.torch.compile", side_effect=lambda fn, **_: fn
    ) as compile_mock:
        for dynamic in (True, False, False, True):
            result, compiled = run_tensor_kernel(
                runtime, "shape test", kernel, torch.tensor(1.0), dynamic=dynamic
            )
            assert compiled
            assert result == 2
    assert compile_mock.call_count == 2
    assert [call.kwargs["dynamic"] for call in compile_mock.call_args_list] == [
        True,
        False,
    ]


def test_fixed_shape_compilation_warns_on_new_shapes() -> None:
    clear_compiled_kernel_cache()
    runtime = _compiled_runtime()

    def kernel(value):
        return value + 1

    with (
        patch("microcaustics.compile.torch.compile", side_effect=lambda fn, **_: fn),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        for length in (2, 2, 3):
            run_tensor_kernel(
                runtime, "fixed shape test", kernel, torch.ones(length), dynamic=False
            )
    assert sum(
        issubclass(item.category, mc.CompilationWarning) for item in caught
    ) == 2


def test_compilation_warning_category_supports_standard_filters() -> None:
    clear_compiled_kernel_cache()
    runtime = _compiled_runtime()

    def kernel(value):
        return value + 1

    with (
        patch("microcaustics.compile.torch.compile", side_effect=lambda fn, **_: fn),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("ignore", mc.CompilationWarning)
        run_tensor_kernel(runtime, "filtered kernel", kernel, torch.tensor(1.0))

    assert caught == []
