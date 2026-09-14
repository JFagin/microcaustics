"""Global strong-lens models and resolved macroimage geometry."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .lens import MacroLens


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


@dataclass(frozen=True)
class CausticsMacroModel:
    """Adapt any compatible model from the ``caustics`` package.

    ``parameters`` is passed to ``raytrace`` and
    ``jacobian_lens_equation``. ``time_delay_parameters`` is passed to
    ``time_delay`` and commonly contains dynamic redshift parameters. Both may
    be tensors, dictionaries, or other parameter structures accepted by the
    wrapped model. No assumption is made about its mass profile or lens planes.
    """

    lens: object
    parameters: object
    time_delay_parameters: object | None = None
    name: str = "caustics_macro_model"

    @staticmethod
    def _has_parameters(parameters) -> bool:
        """Return whether an explicit caskade parameter vector was supplied."""

        if parameters is None:
            return False
        try:
            return int(torch.as_tensor(parameters).numel()) > 0
        except (TypeError, ValueError):
            return True

    def raytrace(self, x_arcsec, y_arcsec):
        """Delegate ray tracing to the wrapped ``caustics`` model."""

        if not self._has_parameters(self.parameters):
            return self.lens.raytrace(x_arcsec, y_arcsec)
        return self.lens.raytrace(x_arcsec, y_arcsec, params=self.parameters)

    def jacobian_lens_equation(self, x_arcsec, y_arcsec):
        """Delegate lens-equation Jacobian evaluation."""

        if not self._has_parameters(self.parameters):
            return self.lens.jacobian_lens_equation(x_arcsec, y_arcsec)
        return self.lens.jacobian_lens_equation(
            x_arcsec, y_arcsec, params=self.parameters
        )

    def time_delay_days(self, x_arcsec, y_arcsec):
        """Delegate arrival-time evaluation to the wrapped model."""

        parameters = (
            self.parameters
            if self.time_delay_parameters is None
            else self.time_delay_parameters
        )
        if not self._has_parameters(parameters):
            return self.lens.time_delay(x_arcsec, y_arcsec)
        return self.lens.time_delay(x_arcsec, y_arcsec, params=parameters)

    def metadata(self) -> Mapping[str, object]:
        """Return lightweight wrapped-model provenance."""

        return {
            "type": "caustics",
            "name": self.name,
            "lens_class": type(self.lens).__name__,
        }


@dataclass(frozen=True)
class MacroImageSolution:
    """One solved macroimage and its local microlensing approximation."""

    name: str
    x_arcsec: float
    y_arcsec: float
    arrival_time_delay_days: float
    absolute_time_delay_days: float
    macro_magnification: float
    parity: int
    convergence: float
    shear: float
    shear_gamma1: float
    shear_gamma2: float
    shear_angle_deg: float
    source_residual_arcsec: float

    @property
    def shear_angle_rad(self) -> float:
        """Return the local shear position angle in radians."""

        return math.radians(self.shear_angle_deg)

    def local_macro_lens(
        self,
        *,
        smooth_matter_fraction: float = 0.0,
    ) -> MacroLens:
        """Return the local convergence/shear model for microlensing."""

        return MacroLens(
            convergence=self.convergence,
            shear=self.shear,
            shear_angle_deg=self.shear_angle_deg,
            smooth_matter_fraction=smooth_matter_fraction,
        )


@torch.no_grad()
def solve_macroimages(
    model: MacroModel,
    source_x_arcsec: float,
    source_y_arcsec: float,
    *,
    image_names: Sequence[str] | None = None,
    initial_grid_size: int = 200,
    field_of_view_arcsec: float = 3.0,
    source_tolerance_arcsec: float | None = None,
    cluster_tolerance_arcsec: float | None = None,
    deduplication_tolerance_arcsec: float | None = None,
    maximum_function_evaluations: int = 200,
    refinement_tolerance_arcsec: float = 1.0e-8,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[MacroImageSolution, ...]:
    """Solve macroimages for any model implementing :class:`MacroModel`.

    A regular grid supplies robust seeds and SciPy refines roots of the lens
    equation. The model's Jacobian supplies magnification and local
    convergence/shear. Its arrival-time function supplies relative delays.
    A candidate is retained only when SciPy reports convergence and its
    source-plane residual is no larger than ``refinement_tolerance_arcsec``.
    Results are sorted by arrival time before names are assigned.
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
    resolved_device = torch.device(device)

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
    jacobian = model.jacobian_lens_equation(x, y)
    if jacobian.shape != (len(unique), 2, 2):
        raise ValueError("macro-model Jacobian must have shape [image, 2, 2]")
    determinant = torch.linalg.det(jacobian)
    singular = ~torch.isfinite(determinant) | (determinant.abs() <= 1.0e-10)
    if bool(torch.any(singular)):
        raise RuntimeError("macroimage solution contains a singular Jacobian")
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
                source_residual_arcsec=float(unique[solution_index][0]),
            )
        )
    return tuple(solutions)


def solve_caustics_macroimages(
    lens,
    source_x_arcsec: float,
    source_y_arcsec: float,
    *,
    parameters=None,
    time_delay_parameters=None,
    **solver_options,
) -> tuple[MacroImageSolution, ...]:
    """Solve images for any compatible lens built with ``caustics``.

    The lens may be a single analytic profile, a ``SinglePlane`` composite, or
    a multiplane system. Static models commonly use an empty parameter tensor;
    dynamic models can pass the parameter structure required by ``caustics``.
    """

    model = CausticsMacroModel(
        lens=lens,
        parameters=parameters,
        time_delay_parameters=time_delay_parameters,
    )
    return solve_macroimages(
        model,
        source_x_arcsec,
        source_y_arcsec,
        **solver_options,
    )


@dataclass(frozen=True)
class EPLShearConfig:
    """Parameters for the EPL plus external-shear convenience model."""

    lens_redshift: float
    source_redshift: float
    einstein_radius_arcsec: float
    axis_ratio: float = 1.0
    position_angle_rad: float = 0.0
    density_slope: float = 1.0
    shear_gamma1: float = 0.0
    shear_gamma2: float = 0.0
    center_x_arcsec: float = 0.0
    center_y_arcsec: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("EPL+shear parameters must be finite")
        if not 0 < self.lens_redshift < self.source_redshift:
            raise ValueError("redshifts must satisfy 0 < z_lens < z_source")
        if self.einstein_radius_arcsec <= 0 or self.density_slope <= 0:
            raise ValueError("Einstein radius and density slope must be positive")
        if not 0 < self.axis_ratio <= 1:
            raise ValueError("axis_ratio must lie in (0, 1]")


@torch.no_grad()
def solve_epl_shear_macroimages(
    config: EPLShearConfig,
    source_x_arcsec: float,
    source_y_arcsec: float,
    **solver_options,
) -> tuple[MacroImageSolution, ...]:
    """Build an EPL plus external-shear lens, then use the general solver."""

    try:
        import caustics
    except ImportError as error:
        raise ImportError(
            "EPL+shear requires the 'macro' optional dependencies"
        ) from error
    dtype = solver_options.get("dtype", torch.float64)
    device = torch.device(solver_options.get("device", "cpu"))

    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device)

    cosmology = caustics.FlatLambdaCDM(name="cosmology")
    cosmology.to(dtype=dtype, device=device)
    common = {
        "cosmology": cosmology,
        "x0": tensor(config.center_x_arcsec),
        "y0": tensor(config.center_y_arcsec),
    }
    epl = caustics.EPL(
        **common,
        q=tensor(config.axis_ratio),
        phi=tensor(config.position_angle_rad),
        Rein=tensor(config.einstein_radius_arcsec),
        t=tensor(config.density_slope),
        s=1.0e-4,
        name="epl",
    )
    shear = caustics.ExternalShear(
        **common,
        gamma_1=tensor(config.shear_gamma1),
        gamma_2=tensor(config.shear_gamma2),
        name="external_shear",
    )
    lens = caustics.SinglePlane(
        cosmology=cosmology,
        lenses=[epl, shear],
        z_l=tensor(config.lens_redshift),
        z_s=tensor(config.source_redshift),
        name="macro_model",
    )
    lens.to(dtype=dtype, device=device)
    return solve_caustics_macroimages(
        lens,
        source_x_arcsec,
        source_y_arcsec,
        **solver_options,
    )


def arrival_time_delay_mapping(
    solutions: Sequence[MacroImageSolution],
) -> dict[str, float]:
    """Return a mapping accepted by ``with_arrival_time_delays``."""

    return {item.name: float(item.arrival_time_delay_days) for item in solutions}
