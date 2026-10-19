"""Global strong-lens models and resolved macroimage geometry."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.overrides import TorchFunctionMode

from .lens import MacroLens
from .runtime import _resolve_high_precision_device


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
        if isinstance(parameters, Mapping):
            return True
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
    """One solved macroimage and its local microlensing approximation.

    ``local_microlensing_supported`` is false for multi-plane lenses or
    rotating Jacobians that cannot be represented by convergence and shear.
    """

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
    local_microlensing_supported: bool = True

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

        if not self.local_microlensing_supported:
            warnings.warn(
                "Conversion of multi-plane lenses or rotating Jacobians to a "
                "local microlensing model is not currently supported. The "
                "returned convergence/shear approximation omits lens-plane "
                "coupling and any image rotation.",
                RuntimeWarning,
                stacklevel=2,
            )
        return MacroLens(
            convergence=self.convergence,
            shear=self.shear,
            shear_angle_deg=self.shear_angle_deg,
            smooth_matter_fraction=smooth_matter_fraction,
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
    is_multiplane = False
    if isinstance(model, CausticsMacroModel):
        try:
            from caustics.lenses.base import ThickLens
        except ImportError:
            pass
        else:
            # A multi-plane arrival-time Hessian is not just the 2x2 source
            # Jacobian, even when that Jacobian happens to be symmetric.
            is_multiplane = isinstance(model.lens, ThickLens)
    tolerance = (
        64 * torch.finfo(jacobian.dtype).eps
        * jacobian.abs().amax(dim=(-1, -2)).clamp_min(1.0)
    )
    symmetric = (
        (jacobian - jacobian.transpose(-1, -2)).abs().amax(dim=(-1, -2))
        <= tolerance
    )
    local_supported = symmetric & (not is_multiplane)
    classify_maxima = not include_central_images and not is_multiplane
    if classify_maxima:
        # For a single-plane lens J is the arrival-time Hessian up to a
        # positive factor. Two negative eigenvalues identify a maximum, not
        # a minimum or saddle. Do not apply this test to rotating/asymmetric
        # multi-plane Jacobians, or force an arbitrary two/four-image count.
        symmetric_part = 0.5 * (jacobian + jacobian.transpose(-1, -2))
        maximum = symmetric & (torch.linalg.eigvalsh(symmetric_part)[:, -1] < -tolerance)
        keep = ~maximum
        if not bool(keep.any()):
            raise RuntimeError(
                "all solved macroimages are central arrival-time maxima. "
                "Use include_central_images=True to retain them"
            )
        x, y, residual = x[keep], y[keep], residual[keep]
        jacobian, determinant = jacobian[keep], determinant[keep]
        local_supported = local_supported[keep]
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
                local_microlensing_supported=bool(local_supported[solution_index]),
            )
        )
    return tuple(solutions)


class _CausticsGridPrecision(TorchFunctionMode):
    """Keep untyped caustics search grids in the requested precision.

    Caustics 1.7.1 creates its initial linspace without a dtype. This scoped,
    thread-local context avoids changing either caustics or torch defaults.
    Explicitly typed factories are left alone.
    """

    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if func is torch.linspace and kwargs.get("dtype") is None:
            kwargs["dtype"] = self.dtype
            kwargs.setdefault("device", self.device)
        return func(*args, **kwargs)


def _solve_native_caustics_macroimages(
    model,
    source_x_arcsec,
    source_y_arcsec,
    *,
    image_names=None,
    include_central_images=False,
    initial_grid_size=100,
    field_of_view_arcsec=3.0,
    refinement_tolerance_arcsec=1.0e-8,
    max_depth=25,
    warn_on_nonconvergence=True,
    dtype=torch.float64,
    device=None,
    runtime=None,
):
    """Delegate image finding to caustics, then check the returned roots."""

    if not callable(getattr(model.lens, "forward_raytrace", None)):
        raise TypeError("lens must provide caustics forward_raytrace")
    if not isinstance(initial_grid_size, int) or initial_grid_size < 8:
        raise ValueError("initial_grid_size must be an integer of at least 8")
    if not math.isfinite(field_of_view_arcsec) or field_of_view_arcsec <= 0:
        raise ValueError("field_of_view_arcsec must be finite and positive")
    if not math.isfinite(source_x_arcsec) or not math.isfinite(source_y_arcsec):
        raise ValueError("source position must be finite")
    if (
        not math.isfinite(refinement_tolerance_arcsec)
        or refinement_tolerance_arcsec <= 0
    ):
        raise ValueError("refinement_tolerance_arcsec must be finite and positive")
    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("macroimage dtype must be torch.float32 or torch.float64")
    resolved_device = _resolve_high_precision_device(runtime, device=device)
    beta_x = torch.as_tensor(source_x_arcsec, dtype=dtype, device=resolved_device)
    beta_y = torch.as_tensor(source_y_arcsec, dtype=dtype, device=resolved_device)
    parameters = (
        {"params": model.parameters}
        if model._has_parameters(model.parameters)
        else {}
    )
    try:
        warning_context = (
            nullcontext() if warn_on_nonconvergence else warnings.catch_warnings()
        )
        with warning_context, _CausticsGridPrecision(dtype, resolved_device):
            if not warn_on_nonconvergence:
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"^Forward raytrace unable to converge reliably, reached "
                        r"maximum depth of \d+ iterations\."
                    ),
                    category=UserWarning,
                    module=r"^caustics\.lenses\.func\.base$",
                )
            x, y = model.lens.forward_raytrace(
                beta_x,
                beta_y,
                fov=2.0 * field_of_view_arcsec,
                divisions=initial_grid_size,
                epsilon=refinement_tolerance_arcsec,
                max_depth=max_depth,
                **parameters,
            )
    except (RuntimeError, IndexError, UnboundLocalError) as error:
        raise RuntimeError(
            "caustics macroimage search failed. Check the search field and "
            "lens/parameter device and dtype (float64 is recommended). "
            "The original caustics error is chained below."
        ) from error
    valid_coordinates = (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and x.ndim == 1
        and y.shape == x.shape
        and x.numel() > 0
        and bool(torch.all(torch.isfinite(x) & torch.isfinite(y)))
    )
    if not valid_coordinates:
        raise RuntimeError("caustics returned no valid macroimage coordinates")
    traced_x, traced_y = model.raytrace(x, y)
    residual = torch.hypot(traced_x - beta_x, traced_y - beta_y)
    if residual.shape != x.shape or not bool(
        torch.all(torch.isfinite(residual) & (residual <= refinement_tolerance_arcsec))
    ):
        raise RuntimeError("caustics macroimage roots exceed the source residual tolerance")
    return _macroimage_solutions(
        model, x, y, residual, image_names,
        include_central_images=include_central_images,
    )


def solve_caustics_macroimages(
    lens,
    source_x_arcsec: float,
    source_y_arcsec: float,
    *,
    parameters=None,
    time_delay_parameters=None,
    include_central_images: bool = False,
    **solver_options,
) -> tuple[MacroImageSolution, ...]:
    """Solve images using the native ``caustics.forward_raytrace`` method.

    The lens may be a single analytic profile, a ``SinglePlane`` composite, or
    a multiplane system. Static models commonly use an empty parameter tensor;
    dynamic models can pass the parameter structure required by ``caustics``.
    The default search uses float64, 100 initial grid divisions and a maximum
    depth of 25. ``field_of_view_arcsec`` is the search half-width.
    ``refinement_tolerance_arcsec`` is passed as the native ``epsilon``, which
    also controls image deduplication. Native search warnings remain visible
    unless ``warn_on_nonconvergence=False`` silences the specific maximum-depth
    warning. This opt-out does not establish completeness or bypass checks.
    A small residual verifies a root, not completeness of the image set.

    Central arrival-time maxima are excluded by default. Set
    ``include_central_images=True`` to retain them. The filter uses negative
    eigenvalues of a symmetric Jacobian, not image brightness, proximity to
    the origin, or a fixed image count. Multi-plane models and asymmetric
    Jacobians are left alone.
    Names and relative delays are assigned after filtering.

    ``device`` overrides ``runtime.high_precision_device`` and otherwise
    defaults to CPU. Supplied lenses and parameters must already match the
    selected device and dtype. The main simulation precision is unchanged.
    """

    model = CausticsMacroModel(
        lens=lens,
        parameters=parameters,
        time_delay_parameters=time_delay_parameters,
    )
    return _solve_native_caustics_macroimages(
        model,
        source_x_arcsec,
        source_y_arcsec,
        include_central_images=include_central_images,
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
    """Build an EPL plus external-shear lens and use the native caustics solver."""

    try:
        import caustics
    except ImportError as error:
        raise ImportError(
            "EPL+shear requires the 'macro' optional dependencies"
        ) from error
    dtype = solver_options.get("dtype", torch.float64)
    device = _resolve_high_precision_device(
        solver_options.get("runtime"), device=solver_options.get("device")
    )

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
