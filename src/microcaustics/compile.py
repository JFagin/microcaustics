"""Cached, honest ``torch.compile`` dispatch for portable tensor kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .config import Backend
from .runtime import ResolvedRuntime, warn_backend_fallback

_COMPILED_KERNELS: dict[tuple[Callable[..., Any], str | None], Callable[..., Any]] = {}
_DISABLED_KERNELS: set[
    tuple[Callable[..., Any], str, str, str | None]
] = set()


def run_tensor_kernel(
    runtime: ResolvedRuntime,
    component: str,
    kernel: Callable[..., Any],
    *args,
) -> tuple[Any, bool]:
    """Execute a tensor kernel eagerly or through a cached compiled callable.

    The boolean return value reports what actually executed. Compilation is
    attempted for a compiled-Torch runtime and for auxiliary Torch operations
    surrounding a primary Triton kernel. A failed kernel is disabled for that
    device/dtype for the rest of the process, preventing repeated compilation
    failures. A strict, explicitly compiled runtime raises the original error;
    an auxiliary Triton-side failure falls back with a warning.
    """

    if runtime.backend not in {Backend.TORCH_COMPILE, Backend.TRITON}:
        return kernel(*args), False

    disabled_key = (
        kernel,
        runtime.device.type,
        str(runtime.dtype),
        runtime.torch_compile_mode,
    )
    if disabled_key in _DISABLED_KERNELS:
        return kernel(*args), False

    cache_key = (kernel, runtime.torch_compile_mode)
    try:
        compiled = _COMPILED_KERNELS.get(cache_key)
        if compiled is None:
            compiled = torch.compile(
                kernel,
                fullgraph=True,
                dynamic=True,
                mode=runtime.torch_compile_mode,
            )
            _COMPILED_KERNELS[cache_key] = compiled
        return compiled(*args), True
    except Exception as error:
        if runtime.strict_backend and runtime.backend is Backend.TORCH_COMPILE:
            raise
        _DISABLED_KERNELS.add(disabled_key)
        warn_backend_fallback(f"torch.compile {component}", error)
        return kernel(*args), False


def clear_compiled_kernel_cache() -> None:
    """Clear package wrapper caches used by tests and controlled benchmarks."""

    _COMPILED_KERNELS.clear()
    _DISABLED_KERNELS.clear()
