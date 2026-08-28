"""Plots and text summaries for honest accelerated-runtime reporting."""

from __future__ import annotations

import platform

import torch

from ..benchmarking import CallableBenchmark
from ..results import TimingBreakdown
from ._common import axes_or_new, finish_axis, require_matplotlib


def runtime_description() -> str:
    """Return a printable description of Python, PyTorch, and accelerator."""

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        device = torch.cuda.get_device_name(index)
        capability = ".".join(str(value) for value in torch.cuda.get_device_capability(index))
        accelerator = f"CUDA {torch.version.cuda}, {device}, compute capability {capability}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "Apple Metal Performance Shaders (MPS)"
    else:
        accelerator = f"CPU, {platform.processor() or platform.machine()}"
    return (
        f"Python {platform.python_version()} | PyTorch {torch.__version__} | "
        f"{platform.system()} {platform.machine()} | {accelerator}"
    )


def print_benchmark(benchmark: CallableBenchmark, *, label: str = "calculation") -> None:
    """Print first-call and warmed timing without calling overhead 'compile'."""

    print(runtime_description())
    print(f"{label} first call in this process: {benchmark.first_call_seconds:.6g} s")
    print(
        f"{label} warmed steady state: {benchmark.median_steady_seconds:.6g} s "
        f"(median of {len(benchmark.steady_seconds)}, "
        f"sample std={benchmark.steady_standard_deviation_seconds:.3g} s)"
    )
    print(
        "First-call excess includes compilation when needed, plus allocator and "
        "cache initialization. An on-disk compiler cache may already be warm."
    )


def plot_timing_breakdown(
    timing: TimingBreakdown,
    *,
    ax=None,
    title: str = "Steady-state timing breakdown",
):
    """Plot named steady-state components from a result's timing metadata."""

    require_matplotlib()
    if not timing.collected:
        raise ValueError(
            "timing was not collected; use RuntimeConfig(profiling='detailed')"
        )
    figure, ax = axes_or_new(ax, figsize=(6.4, 3.8))
    names = tuple(timing.component_seconds)
    values = [float(timing.component_seconds[name]) for name in names]
    if not names:
        names = ("steady state",)
        values = [float(timing.steady_seconds)]
    ax.barh(names, values, color="C0")
    ax.set_xlabel("Runtime [s]")
    ax.set_title(title)
    ax.invert_yaxis()
    finish_axis(ax, grid=True)
    return figure, ax
