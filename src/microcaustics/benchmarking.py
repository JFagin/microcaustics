"""Small, explicit utilities for measuring accelerated calculations."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, TypeVar

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class CallableBenchmark(Generic[ResultT]):
    """First-call and steady-state timings for a zero-argument callable.

    The first call may include compilation, allocator initialization, cache
    population, and ordinary numerical work. It is intentionally not labeled
    pure compile time because those costs cannot be separated portably.
    """

    first_result: ResultT
    first_call_seconds: float
    warmup_seconds: tuple[float, ...]
    steady_seconds: tuple[float, ...]

    @property
    def median_steady_seconds(self) -> float:
        """Median repeat time after first-call and warmup work."""

        return float(statistics.median(self.steady_seconds))

    @property
    def mean_steady_seconds(self) -> float:
        """Arithmetic mean repeat time after warmup."""

        return float(statistics.fmean(self.steady_seconds))

    @property
    def steady_standard_deviation_seconds(self) -> float:
        """Sample standard deviation, or zero when only one repeat was run."""

        if len(self.steady_seconds) < 2:
            return 0.0
        return float(statistics.stdev(self.steady_seconds))

    @property
    def estimated_first_call_overhead_seconds(self) -> float:
        """First-call excess over median steady time.

        This includes compilation but may also include allocator and cache
        initialization, so it must not be reported as pure compile time.
        """

        return max(0.0, self.first_call_seconds - self.median_steady_seconds)


def benchmark_callable(
    function: Callable[[], ResultT],
    *,
    warmup: int = 1,
    repeats: int = 5,
    synchronize: Callable[[], None] | None = None,
) -> CallableBenchmark[ResultT]:
    """Measure first-call and post-warmup runtime without hiding compilation.

    ``synchronize`` should be the selected runtime's ``synchronize`` method for
    asynchronous CUDA or MPS work. The callable is executed once for the
    first-call measurement, ``warmup`` additional times, and then ``repeats``
    times for the steady-state distribution.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    sync = (lambda: None) if synchronize is None else synchronize

    def measured_call():
        sync()
        started = perf_counter()
        result = function()
        sync()
        return result, perf_counter() - started

    first_result, first_seconds = measured_call()
    warmup_values = tuple(measured_call()[1] for _ in range(int(warmup)))
    steady_values = tuple(measured_call()[1] for _ in range(int(repeats)))
    return CallableBenchmark(
        first_result=first_result,
        first_call_seconds=first_seconds,
        warmup_seconds=warmup_values,
        steady_seconds=steady_values,
    )
