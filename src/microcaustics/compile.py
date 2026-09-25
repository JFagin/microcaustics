"""Cached, honest ``torch.compile`` dispatch for portable tensor kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .config import Backend
from .runtime import ResolvedRuntime, warn_backend_fallback, warn_compilation

_COMPILED_KERNELS: dict[
    tuple[Callable[..., Any], str | None, bool], Callable[..., Any]
] = {}
_DISABLED_KERNELS: dict[
    tuple[Callable[..., Any], str, str, str | None, bool], str
] = {}
_PREPARED_SPECIALIZATIONS: set[tuple[object, ...]] = set()


def _argument_signature(args, *, include_shape: bool = False) -> tuple[object, ...]:
    signature = []
    for value in args:
        if isinstance(value, torch.Tensor):
            signature.append(
                (
                    "tensor",
                    str(value.device),
                    str(value.dtype),
                    tuple(value.shape) if include_shape else value.ndim,
                )
            )
        else:
            signature.append((type(value).__qualname__, repr(value)))
    return tuple(signature)


def run_tensor_kernel(
    runtime: ResolvedRuntime,
    component: str,
    kernel: Callable[..., Any],
    *args,
    dynamic: bool = True,
) -> tuple[Any, bool]:
    """Execute a tensor kernel eagerly or through a cached compiled callable.

    The boolean return value reports what actually executed. ``dynamic=False``
    specializes to exact tensor shapes and can recompile when shapes change.
    Compilation is attempted for a compiled-Torch runtime and for auxiliary
    Torch operations surrounding a primary Triton kernel. A failed kernel is
    disabled for that device/dtype for the rest of the process, preventing
    repeated compilation failures. Strict runtimes raise on failed or
    previously disabled compiled kernels; non-strict runtimes warn and fall
    back to eager Torch.
    """

    if runtime.backend not in {Backend.TORCH_COMPILE, Backend.TRITON}:
        return kernel(*args), False

    disabled_key = (
        kernel,
        runtime.device.type,
        str(runtime.dtype),
        runtime.torch_compile_mode,
        dynamic,
    )
    if disabled_key in _DISABLED_KERNELS:
        if runtime.strict_backend:
            raise RuntimeError(
                f"compiled {component} is unavailable after an earlier failure: "
                f"{_DISABLED_KERNELS[disabled_key]}"
            )
        return kernel(*args), False

    cache_key = (kernel, runtime.torch_compile_mode, dynamic)
    try:
        compiled = _COMPILED_KERNELS.get(cache_key)
        if compiled is None:
            # Cache the wrapper once. Torch may still compile concrete
            # specializations as shape, dtype, or device changes.
            compiled = torch.compile(
                kernel,
                fullgraph=True,
                dynamic=dynamic,
                mode=runtime.torch_compile_mode,
            )
            _COMPILED_KERNELS[cache_key] = compiled
        specialization = (
            kernel,
            runtime.torch_compile_mode,
            runtime.warn_on_compile,
            _argument_signature(args, include_shape=not dynamic),
            dynamic,
        )
        if specialization not in _PREPARED_SPECIALIZATIONS:
            # This signature mirrors the dimensions that can trigger a new
            # specialization without treating ordinary tensor values as shapes.
            warn_compilation(
                component,
                backend="torch.compile",
                device=runtime.device,
                dtype=runtime.dtype,
                enabled=runtime.warn_on_compile,
            )
            _PREPARED_SPECIALIZATIONS.add(specialization)
        return compiled(*args), True
    except Exception as error:
        if runtime.strict_backend:
            raise
        # Do not retain the exception traceback: its frames can hold GPU tensors.
        _DISABLED_KERNELS[disabled_key] = f"{type(error).__name__}: {error}"
        warn_backend_fallback(f"torch.compile {component}", error)
        return kernel(*args), False


def clear_compiled_kernel_cache() -> None:
    """Clear package wrapper caches used by tests and controlled benchmarks."""

    _COMPILED_KERNELS.clear()
    _DISABLED_KERNELS.clear()
    _PREPARED_SPECIALIZATIONS.clear()
