"""Extensible stellar and compact-object mass distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class MassFunction(Protocol):
    """Probability distribution that samples masses in solar-mass units."""

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw ``count`` independent masses in solar-mass units."""

        ...

    def mean_mass(self) -> float:
        """Return the analytic mean mass in solar-mass units."""

        ...

    def second_moment(self) -> float:
        """Return the analytic mean squared mass in solar-mass-squared units."""

        ...


def _power_integral(lower: float, upper: float, exponent: float) -> float:
    """Integrate ``m**exponent`` between two positive bounds."""

    power = exponent + 1.0
    if abs(power) < 1.0e-12:
        return float(torch.log(torch.tensor(upper / lower, dtype=torch.float64)))
    return (upper**power - lower**power) / power


def _sample_power_law_interval(
    uniform: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    slope: torch.Tensor,
) -> torch.Tensor:
    """Apply the inverse CDF of ``p(m) proportional to m**(-slope)``."""

    power = 1.0 - slope
    logarithmic = power.abs() < 1.0e-7
    safe_power = torch.where(logarithmic, torch.ones_like(power), power)
    ordinary = (
        uniform * (upper.pow(safe_power) - lower.pow(safe_power))
        + lower.pow(safe_power)
    ).pow(1.0 / safe_power)
    log_uniform = lower * torch.exp(uniform * torch.log(upper / lower))
    return torch.where(logarithmic, log_uniform, ordinary)


@dataclass(frozen=True)
class PowerLawMassFunction:
    """A truncated power-law distribution ``dN/dm proportional to m^-slope``."""

    minimum_mass: float
    maximum_mass: float
    slope: float

    def __post_init__(self) -> None:
        if self.minimum_mass <= 0:
            raise ValueError("minimum_mass must be positive")
        if self.maximum_mass < self.minimum_mass:
            raise ValueError("maximum_mass must be at least minimum_mass")

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw masses by an analytic inverse CDF without discretization."""

        if int(count) < 0:
            raise ValueError("count must be non-negative")
        if self.maximum_mass == self.minimum_mass:
            return torch.full(
                (int(count),),
                self.minimum_mass,
                device=device,
                dtype=dtype,
            )
        uniform = torch.rand(
            int(count),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        lower = torch.as_tensor(self.minimum_mass, device=device, dtype=dtype)
        upper = torch.as_tensor(self.maximum_mass, device=device, dtype=dtype)
        slope = torch.as_tensor(self.slope, device=device, dtype=dtype)
        return _sample_power_law_interval(uniform, lower, upper, slope)

    def mean_mass(self) -> float:
        """Return the exact first moment of the truncated distribution."""

        if self.maximum_mass == self.minimum_mass:
            return float(self.minimum_mass)
        normalization = _power_integral(
            self.minimum_mass,
            self.maximum_mass,
            -self.slope,
        )
        first_moment = _power_integral(
            self.minimum_mass,
            self.maximum_mass,
            1.0 - self.slope,
        )
        return first_moment / normalization

    def second_moment(self) -> float:
        """Return the exact second moment of the truncated distribution."""

        if self.maximum_mass == self.minimum_mass:
            return float(self.minimum_mass**2)
        normalization = _power_integral(
            self.minimum_mass,
            self.maximum_mass,
            -self.slope,
        )
        second_moment = _power_integral(
            self.minimum_mass,
            self.maximum_mass,
            2.0 - self.slope,
        )
        return second_moment / normalization


@dataclass(frozen=True)
class BrokenPowerLawMassFunction:
    """A continuous piecewise power-law mass function.

    ``edges`` contains every interval boundary and ``slopes`` contains one
    ``dN/dm`` exponent per interval. Relative interval normalizations are
    chosen so the probability density is continuous at every break.
    """

    edges: tuple[float, ...]
    slopes: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.slopes) + 1:
            raise ValueError("edges must contain exactly one more value than slopes")
        if len(self.slopes) < 1:
            raise ValueError("at least one power-law interval is required")
        if any(value <= 0 for value in self.edges):
            raise ValueError("mass edges must be positive")
        if any(
            right <= left
            for left, right in zip(self.edges, self.edges[1:], strict=False)
        ):
            raise ValueError("mass edges must be strictly increasing")

    def _coefficients_and_weights(self) -> tuple[list[float], list[float]]:
        coefficients = [1.0]
        for index in range(1, len(self.slopes)):
            boundary = self.edges[index]
            coefficients.append(
                coefficients[-1]
                * boundary ** (self.slopes[index] - self.slopes[index - 1])
            )
        weights = [
            coefficient * _power_integral(lower, upper, -slope)
            for coefficient, lower, upper, slope in zip(
                coefficients,
                self.edges[:-1],
                self.edges[1:],
                self.slopes,
                strict=True,
            )
        ]
        return coefficients, weights

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw masses using exact categorical and interval inverse CDFs."""

        if int(count) < 0:
            raise ValueError("count must be non-negative")
        _, weights = self._coefficients_and_weights()
        probabilities = torch.as_tensor(weights, device=device, dtype=torch.float64)
        intervals = torch.multinomial(
            probabilities,
            int(count),
            replacement=True,
            generator=generator,
        )
        uniform = torch.rand(
            int(count),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        edges = torch.as_tensor(self.edges, device=device, dtype=dtype)
        slopes = torch.as_tensor(self.slopes, device=device, dtype=dtype)
        lower = edges[intervals]
        upper = edges[intervals + 1]
        return _sample_power_law_interval(uniform, lower, upper, slopes[intervals])

    def mean_mass(self) -> float:
        """Return the exact first moment of the piecewise distribution."""

        coefficients, weights = self._coefficients_and_weights()
        first_moment = sum(
            coefficient * _power_integral(lower, upper, 1.0 - slope)
            for coefficient, lower, upper, slope in zip(
                coefficients,
                self.edges[:-1],
                self.edges[1:],
                self.slopes,
                strict=True,
            )
        )
        return first_moment / sum(weights)

    def second_moment(self) -> float:
        """Return the exact second moment of the piecewise distribution."""

        coefficients, weights = self._coefficients_and_weights()
        second_moment = sum(
            coefficient * _power_integral(lower, upper, 2.0 - slope)
            for coefficient, lower, upper, slope in zip(
                coefficients,
                self.edges[:-1],
                self.edges[1:],
                self.slopes,
                strict=True,
            )
        )
        return second_moment / sum(weights)


def salpeter_mass_function(
    minimum_mass: float = 0.1,
    maximum_mass: float = 100.0,
) -> PowerLawMassFunction:
    """Return the standard Salpeter ``dN/dm proportional to m^-2.35`` form."""

    return PowerLawMassFunction(minimum_mass, maximum_mass, 2.35)


def kroupa_mass_function(
    minimum_mass: float = 0.01,
    maximum_mass: float = 100.0,
) -> BrokenPowerLawMassFunction:
    """Return a canonical three-segment Kroupa-like stellar mass function.

    The default break masses are 0.08 and 0.5 solar masses with slopes 0.3,
    1.3, and 2.3. Truncating the requested range removes intervals outside it
    while preserving the appropriate local slope.
    """

    if minimum_mass <= 0 or maximum_mass <= minimum_mass:
        raise ValueError("mass limits must be positive and ordered")
    base_edges = (0.01, 0.08, 0.5, 100.0)
    base_slopes = (0.3, 1.3, 2.3)
    edges = [minimum_mass]
    edges.extend(edge for edge in base_edges[1:-1] if minimum_mass < edge < maximum_mass)
    edges.append(maximum_mass)
    slopes = []
    for left, right in zip(edges, edges[1:], strict=False):
        midpoint = (left * right) ** 0.5
        if midpoint < 0.08:
            slopes.append(base_slopes[0])
        elif midpoint < 0.5:
            slopes.append(base_slopes[1])
        else:
            slopes.append(base_slopes[2])
    return BrokenPowerLawMassFunction(tuple(edges), tuple(slopes))
