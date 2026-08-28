"""Opt-in interoperability with the external :mod:`caustics` package.

The core numerical package does not depend on or import ``caustics``. Calling
:func:`as_caustics_thin_lens` lazily constructs a real ``caustics.ThinLens``
whose lens equation delegates to an already-built microcaustics tracer. The
adapter therefore preserves the selected eager, compiled, or Triton kernels;
it changes only the outer orchestration interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class _LocalRayTracer(Protocol):
    """Structural contract required by the caustics adapter."""

    def raytrace(self, x_uas, y_uas) -> tuple[torch.Tensor, torch.Tensor]:
        """Map local lens-plane coordinates to local source coordinates."""


_ADAPTER_TYPE: type | None = None


def _caustics_module():
    """Import the optional dependency with an actionable error message."""

    try:
        import caustics
    except ImportError as error:
        raise ImportError(
            "as_caustics_thin_lens requires the optional 'macro' extra. "
            "install it with `pip install \"microcaustics[macro]\"`"
        ) from error
    return caustics


def _adapter_type(caustics) -> type:
    """Create the adapter class only after the optional import succeeds."""

    global _ADAPTER_TYPE
    if _ADAPTER_TYPE is not None:
        return _ADAPTER_TYPE

    class CausticsThinLensAdapter(caustics.ThinLens):
        """A caustics lens backed by one fixed microcaustics lens state."""

        def __init__(
            self,
            tracer: _LocalRayTracer,
            *,
            cosmology,
            z_l,
            z_s,
            name: str,
            origin_arcsec: tuple[float, float],
            microarcseconds_per_unit: float,
        ) -> None:
            super().__init__(cosmology=cosmology, z_l=z_l, z_s=z_s, name=name)
            self.tracer = tracer
            self.origin_x_arcsec = float(origin_arcsec[0])
            self.origin_y_arcsec = float(origin_arcsec[1])
            self.microarcseconds_per_unit = float(microarcseconds_per_unit)

        @caustics.forward
        def reduced_deflection_angle(self, x, y):
            """Return the complete local reduced deflection in arcseconds."""

            scale = self.microarcseconds_per_unit
            local_x = (x - self.origin_x_arcsec) * scale
            local_y = (y - self.origin_y_arcsec) * scale
            source_x_uas, source_y_uas = self.tracer.raytrace(local_x, local_y)
            source_x = self.origin_x_arcsec + source_x_uas / scale
            source_y = self.origin_y_arcsec + source_y_uas / scale
            return x - source_x, y - source_y

    CausticsThinLensAdapter.__name__ = "CausticsThinLensAdapter"
    CausticsThinLensAdapter.__qualname__ = "CausticsThinLensAdapter"
    CausticsThinLensAdapter.__module__ = __name__
    _ADAPTER_TYPE = CausticsThinLensAdapter
    return CausticsThinLensAdapter


def as_caustics_thin_lens(
    tracer: _LocalRayTracer,
    *,
    cosmology=None,
    z_l: float | None = None,
    z_s: float | None = None,
    name: str = "microcaustics_local_lens",
    origin_arcsec: tuple[float, float] = (0.0, 0.0),
    microarcseconds_per_unit: float = 1.0e6,
):
    """Expose a fixed microcaustics tracer as a ``caustics.ThinLens``.

    Parameters
    ----------
    tracer:
        A fixed-state local tracer with ``raytrace(x_uas, y_uas)``. A
        :class:`microcaustics.solvers.TaylorFarFieldApproximation` is the normal choice.
        Its macro sheet, shear, local exact stars, and Taylor far field are all
        retained. A temporal batch is not one physical thin lens. Adapt its
        individual ``far_fields[frame]`` instead.
    cosmology:
        A caustics cosmology. If omitted, ``caustics.FlatLambdaCDM()`` is used.
        Cosmological distances do not rescale the already reduced local lens
        equation, but caustics requires a cosmology for lens composition.
    z_l, z_s:
        Optional lens and source redshifts stored on the returned thin lens.
    name:
        Name registered with caustics. Supply unique names when composing
        several adapters in one caustics model.
    origin_arcsec:
        Global caustics coordinate corresponding to local ``(0, 0)`` in the
        microcaustics tracer. This lets a local field be placed at a resolved
        macroimage without translating its internal star coordinates.
    microarcseconds_per_unit:
        Conversion from caustics coordinates to microarcseconds. The default
        is ``1e6`` because caustics uses arcseconds.

    Returns
    -------
    caustics.ThinLens
        A real caustics lens that may be placed in ``caustics.SinglePlane`` or
        passed to other caustics simulators.

    Notes
    -----
    This adapter is entirely opt-in and is not a numerical backend. Ordinary
    microcaustics calls continue to use the direct package interface. The
    returned lens delegates to the tracer without copying tensors to the CPU,
    so its eager, ``torch.compile``, or Triton query path remains active.
    """

    if not isinstance(tracer, _LocalRayTracer):
        raise TypeError("tracer must provide raytrace(x_uas, y_uas)")
    from ..solvers.far_field import BatchedTaylorFarFieldApproximation

    if isinstance(tracer, BatchedTaylorFarFieldApproximation):
        raise TypeError(
            "a temporal far-field approximation batch is not one ThinLens. Adapt "
            "far_fields[frame] instead"
        )
    if len(origin_arcsec) != 2:
        raise ValueError("origin_arcsec must contain exactly two coordinates")
    if microarcseconds_per_unit <= 0:
        raise ValueError("microarcseconds_per_unit must be positive")
    if z_l is not None and z_l < 0:
        raise ValueError("z_l must be non-negative")
    if z_s is not None and z_s <= 0:
        raise ValueError("z_s must be positive")
    if z_l is not None and z_s is not None and z_s <= z_l:
        raise ValueError("z_s must exceed z_l")

    caustics = _caustics_module()
    if cosmology is None:
        cosmology = caustics.FlatLambdaCDM()
    adapter = _adapter_type(caustics)
    return adapter(
        tracer,
        cosmology=cosmology,
        z_l=z_l,
        z_s=z_s,
        name=name,
        origin_arcsec=origin_arcsec,
        microarcseconds_per_unit=microarcseconds_per_unit,
    )


__all__ = ["as_caustics_thin_lens"]
