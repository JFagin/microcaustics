"""Frozen pre-retirement macro solver for internal regression comparisons only.

Copied on 2026-10-16. This is not a runtime API or ground truth. Analytic
solutions and independent lensing codes remain the primary accuracy checks.
Do not import this module from package source or user examples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from microcaustics.runtime import (
    ResolvedRuntime,
    RuntimeConfig,
    _resolve_high_precision_device,
)
from microcaustics.strong_lensing import CausticsMacroModel, MacroImageSolution


@runtime_checkable
class MacroModel(Protocol):
    """Minimal global lens-model contract used by the image solver.

    Coordinates are angular arcseconds and delays are observer-frame days.
    Implementations may use ``caustics``, analytic expressions, another
    lensing package, or arbitrary user callables.
    """

    def raytrace(
        self,
        x_arcsec: torch.Tensor,
        y_arcsec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map image-plane coordinates into the source plane."""

        ...

    def jacobian_lens_equation(
        self,
        x_arcsec: torch.Tensor,
        y_arcsec: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``d beta / d theta`` with trailing shape ``[2, 2]``."""

        ...

    def time_delay_days(
        self,
        x_arcsec: torch.Tensor,
        y_arcsec: torch.Tensor,
    ) -> torch.Tensor:
        """Return consistently normalized arrival times in days."""

        ...


@dataclass(frozen=True)
class CallableMacroModel:
    """Adapt three user callables to the general :class:`MacroModel` API."""

    raytrace_function: Callable
    jacobian_function: Callable
    time_delay_function: Callable
    name: str = "callable_macro_model"
    user_metadata: Mapping[str, object] | None = None

    def raytrace(self, x_arcsec, y_arcsec):
        """Evaluate and validate the user ray-tracing callable."""

        output = self.raytrace_function(x_arcsec, y_arcsec)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("raytrace_function must return an (x, y) tuple")
        beta_x = torch.as_tensor(
            output[0],
            device=x_arcsec.device,
            dtype=x_arcsec.dtype,
        )
        beta_y = torch.as_tensor(
            output[1],
            device=x_arcsec.device,
            dtype=x_arcsec.dtype,
        )
        if beta_x.shape != x_arcsec.shape or beta_y.shape != y_arcsec.shape:
            raise ValueError("raytrace_function outputs must match the inputs")
        return beta_x, beta_y

    def jacobian_lens_equation(self, x_arcsec, y_arcsec):
        """Evaluate and validate the user Jacobian callable."""

        value = torch.as_tensor(
            self.jacobian_function(x_arcsec, y_arcsec),
            device=x_arcsec.device,
            dtype=x_arcsec.dtype,
        )
        if value.shape != (*x_arcsec.shape, 2, 2):
            raise ValueError("jacobian_function must return shape [*input, 2, 2]")
        return value

    def time_delay_days(self, x_arcsec, y_arcsec):
        """Evaluate and validate the user time-delay callable."""

        value = torch.as_tensor(
            self.time_delay_function(x_arcsec, y_arcsec),
            device=x_arcsec.device,
            dtype=x_arcsec.dtype,
        )
        if value.shape != x_arcsec.shape:
            raise ValueError("time_delay_function output must match the input")
        return value

    def metadata(self) -> Mapping[str, object]:
        """Return user-supplied model provenance."""

        return {
            "type": "callable",
            "name": self.name,
            **dict(self.user_metadata or {}),
        }


@torch.no_grad()
def solve_macroimages(
    model: MacroModel,
    source_x_arcsec: float,
    source_y_arcsec: float,
    *,
    image_names: Sequence[str] | None = None,
    include_central_images: bool = False,
    initial_grid_size: int = 200,
    field_of_view_arcsec: float = 3.0,
    source_tolerance_arcsec: float | None = None,
    cluster_tolerance_arcsec: float | None = None,
    deduplication_tolerance_arcsec: float | None = None,
    maximum_function_evaluations: int = 200,
    refinement_tolerance_arcsec: float = 1.0e-8,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    runtime: RuntimeConfig | ResolvedRuntime | None = None,
) -> tuple[MacroImageSolution, ...]:
    """Solve macroimages for any model implementing :class:`MacroModel`.

    A regular grid supplies robust seeds and SciPy refines roots of the lens
    equation. The model's Jacobian supplies magnification and local
    convergence/shear. Its arrival-time function supplies relative delays.
    A candidate is retained only when SciPy reports convergence and its
    source-plane residual is no larger than ``refinement_tolerance_arcsec``.
    Results are sorted by arrival time before names are assigned.
    Central arrival-time maxima are excluded unless ``include_central_images``
    is true. This classification requires a symmetric lens Jacobian.
    ``device`` overrides ``runtime.high_precision_device``. Without either,
    solving remains on CPU. The default dtype remains float64 independently
    of the main runtime dtype. Supplied models must match the selected device.
    """

    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise ImportError("macroimage root solving requires scipy>=1.10") from error
    if not isinstance(model, MacroModel):
        raise TypeError("model must implement the MacroModel protocol")
    if initial_grid_size < 8 or field_of_view_arcsec <= 0:
        raise ValueError("initial_grid_size and field_of_view_arcsec are too small")
    if not math.isfinite(source_x_arcsec) or not math.isfinite(source_y_arcsec):
        raise ValueError("source position must be finite")
    if maximum_function_evaluations < 1:
        raise ValueError("maximum_function_evaluations must be positive")
    resolved_device = _resolve_high_precision_device(runtime, device=device)

    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=resolved_device)

    beta_x, beta_y = tensor(source_x_arcsec), tensor(source_y_arcsec)
    pixel = 2.0 * field_of_view_arcsec / (initial_grid_size - 1)
    source_tolerance = (
        0.75 * pixel
        if source_tolerance_arcsec is None
        else float(source_tolerance_arcsec)
    )
    cluster_tolerance = (
        2.5 * pixel
        if cluster_tolerance_arcsec is None
        else float(cluster_tolerance_arcsec)
    )
    deduplication_tolerance = (
        3.0 * pixel
        if deduplication_tolerance_arcsec is None
        else float(deduplication_tolerance_arcsec)
    )
    if (
        min(
            source_tolerance,
            cluster_tolerance,
            deduplication_tolerance,
            refinement_tolerance_arcsec,
        )
        <= 0
    ):
        raise ValueError("image-finding tolerances must be positive")

    axis = torch.linspace(
        -field_of_view_arcsec,
        field_of_view_arcsec,
        initial_grid_size,
        dtype=dtype,
        device=resolved_device,
    )
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    flat_x, flat_y = grid_x.reshape(-1), grid_y.reshape(-1)
    traced_x, traced_y = model.raytrace(flat_x, flat_y)
    residual = torch.hypot(traced_x - beta_x, traced_y - beta_y)
    # The grid is only a robust seed finder. Nearby selected pixels are merged
    # before continuous root refinement so one image does not launch many fits.
    selected = residual < source_tolerance
    if not bool(torch.any(selected)):
        raise RuntimeError("no macroimage seeds found. Enlarge the field or tolerance")
    candidates = torch.stack((flat_x[selected], flat_y[selected]), dim=1)
    candidates = candidates.detach().cpu().numpy()
    clusters: list[list[np.ndarray]] = []
    for point in candidates:
        for cluster in clusters:
            if np.linalg.norm(point - np.mean(cluster, axis=0)) < cluster_tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    guesses = [np.mean(cluster, axis=0) for cluster in clusters]

    def residual_function(position):
        x, y = tensor([position[0]]), tensor([position[1]])
        traced = model.raytrace(x, y)
        return np.array(
            [float(traced[0][0] - beta_x), float(traced[1][0] - beta_y)],
            dtype=np.float64,
        )

    def jacobian_function(position):
        x, y = tensor([position[0]]), tensor([position[1]])
        jacobian = model.jacobian_lens_equation(x, y)[0]
        return jacobian.detach().cpu().numpy().astype(np.float64)

    refined = []
    for guess in guesses:
        fit = least_squares(
            residual_function,
            np.asarray(guess, dtype=np.float64),
            jac=jacobian_function,
            method="trf",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=maximum_function_evaluations,
        )
        fit_residual = float(np.linalg.norm(fit.fun))
        if (
            fit.success
            and np.isfinite(fit_residual)
            and fit_residual <= refinement_tolerance_arcsec
        ):
            refined.append((fit_residual, float(fit.x[0]), float(fit.x[1])))
    refined.sort(key=lambda value: value[0])
    # Keep the lowest-residual representative of each converged image. This
    # second deduplication is required because separate seed clusters can flow
    # to the same nonlinear root.
    unique = []
    for candidate in refined:
        if all(
            math.hypot(candidate[1] - other[1], candidate[2] - other[2])
            >= deduplication_tolerance
            for other in unique
        ):
            unique.append(candidate)
    if not unique:
        raise RuntimeError("macroimage root refinement did not produce a valid image")

    x = tensor([value[1] for value in unique])
    y = tensor([value[2] for value in unique])
    return _macroimage_solutions(
        model,
        x,
        y,
        tensor([value[0] for value in unique]),
        image_names,
        include_central_images=include_central_images,
    )


def _macroimage_solutions(
    model, x, y, residual, image_names, *, include_central_images=False
):
    """Convert lens-supplied derivatives and delays to the public result."""

    jacobian = model.jacobian_lens_equation(x, y)
    if jacobian.shape != (x.numel(), 2, 2):
        raise ValueError("macro-model Jacobian must have shape [image, 2, 2]")
    determinant = torch.linalg.det(jacobian)
    singular = ~torch.isfinite(determinant) | (determinant.abs() <= 1.0e-10)
    if bool(torch.any(singular)):
        raise RuntimeError("macroimage solution contains a singular Jacobian")
    classify_maxima = not include_central_images
    if classify_maxima and isinstance(model, CausticsMacroModel):
        try:
            from caustics.lenses.base import ThickLens
        except ImportError:
            pass
        else:
            # A multi-plane arrival-time Hessian is not just the 2x2 source
            # Jacobian, even when that Jacobian happens to be symmetric.
            classify_maxima = not isinstance(model.lens, ThickLens)
    if classify_maxima:
        # For a single-plane lens J is the arrival-time Hessian up to a
        # positive factor. Two negative eigenvalues identify a maximum, not
        # a minimum or saddle. Do not apply this test to rotating/asymmetric
        # multi-plane Jacobians, or force an arbitrary two/four-image count.
        symmetric_part = 0.5 * (jacobian + jacobian.transpose(-1, -2))
        tolerance = (
            64
            * torch.finfo(jacobian.dtype).eps
            * jacobian.abs().amax(dim=(-1, -2)).clamp_min(1.0)
        )
        symmetric = (jacobian - jacobian.transpose(-1, -2)).abs().amax(
            dim=(-1, -2)
        ) <= tolerance
        maximum = symmetric & (
            torch.linalg.eigvalsh(symmetric_part)[:, -1] < -tolerance
        )
        keep = ~maximum
        if not bool(keep.any()):
            raise RuntimeError(
                "all solved macroimages are central arrival-time maxima. "
                "Use include_central_images=True to retain them"
            )
        x, y, residual = x[keep], y[keep], residual[keep]
        jacobian, determinant = jacobian[keep], determinant[keep]
    a_xx, a_xy = jacobian[:, 0, 0], jacobian[:, 0, 1]
    a_yx, a_yy = jacobian[:, 1, 0], jacobian[:, 1, 1]
    convergence = 1.0 - 0.5 * (a_xx + a_yy)
    gamma1 = 0.5 * (a_yy - a_xx)
    gamma2 = -0.5 * (a_xy + a_yx)
    gamma = torch.hypot(gamma1, gamma2)
    angle = 0.5 * torch.atan2(gamma2, gamma1)
    absolute_delay = model.time_delay_days(x, y)
    valid_delay = absolute_delay.shape == x.shape and bool(
        torch.all(torch.isfinite(absolute_delay))
    )
    if not valid_delay:
        raise ValueError("macro-model time delays must be finite with shape [image]")
    relative_delay = absolute_delay - absolute_delay.min()
    # Arrival-time ordering is the public deterministic order; user-provided
    # names are assigned only after this physical ordering is established.
    ordering = torch.argsort(relative_delay).detach().cpu().tolist()
    if image_names is not None and len(image_names) != len(ordering):
        raise ValueError("image_names must match the number of solved macroimages")
    names = (
        tuple(str(value) for value in image_names)
        if image_names is not None
        else tuple(
            chr(65 + index) if index < 26 else f"image_{index + 1}"
            for index in range(len(ordering))
        )
    )
    solutions = []
    for output_index, solution_index in enumerate(ordering):
        solutions.append(
            MacroImageSolution(
                name=names[output_index],
                x_arcsec=float(x[solution_index]),
                y_arcsec=float(y[solution_index]),
                arrival_time_delay_days=float(relative_delay[solution_index]),
                absolute_time_delay_days=float(absolute_delay[solution_index]),
                macro_magnification=float(1.0 / determinant[solution_index]),
                parity=int(torch.sign(determinant[solution_index])),
                convergence=float(convergence[solution_index]),
                shear=float(gamma[solution_index]),
                shear_gamma1=float(gamma1[solution_index]),
                shear_gamma2=float(gamma2[solution_index]),
                shear_angle_deg=math.degrees(float(angle[solution_index])),
                source_residual_arcsec=float(residual[solution_index]),
            )
        )
    return tuple(solutions)
