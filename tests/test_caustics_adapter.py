"""Tests for the optional caustics ThinLens interoperability boundary."""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

import torch

import microcaustics as mc
from microcaustics.integrations import caustics as adapter_module
from microcaustics.solvers import (
    BatchedTaylorFarFieldApproximation,
    TaylorFarFieldApproximation,
)

CAUSTICS_AVAILABLE = importlib.util.find_spec("caustics") is not None


def _far_field() -> TaylorFarFieldApproximation:
    """Build a compact deterministic float64 lens for adapter tests."""

    generator = torch.Generator().manual_seed(31415)
    star_x = torch.rand(24, generator=generator, dtype=torch.float64) * 7.0 - 3.5
    star_y = torch.rand(24, generator=generator, dtype=torch.float64) * 7.0 - 3.5
    simulation = mc.MicrolensingSimulation.create(
        mc.MacroLens(
            convergence=0.27,
            shear=0.16,
            shear_angle_deg=17.76169164905552,
            smooth_matter_fraction=0.4,
        ),
        mc.PointMassField._from_einstein_radii(
            star_x,
            star_y,
            einstein_radius_uas=torch.full((24,), 0.075, dtype=torch.float64),
        ),
        runtime=mc.RuntimeConfig(
            device="cpu",
            backend=mc.Backend.TORCH_EAGER,
            dtype="float64",
        ),
    )
    return TaylorFarFieldApproximation(
        simulation,
        mc.PlaneRegion((4.0, 4.0)),
        mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
            taylor_order=4,
            center_translation_order=10,
        ),
    )


class CausticsAdapterDependencyTests(unittest.TestCase):
    def test_missing_optional_dependency_has_actionable_message(self) -> None:
        tracer = mock.Mock()
        tracer.raytrace = mock.Mock(return_value=(torch.zeros(1), torch.zeros(1)))
        with (
            mock.patch.object(
                adapter_module,
                "_caustics_module",
                side_effect=ImportError(
                    "as_caustics_thin_lens requires the optional 'macro' extra"
                ),
            ),
            self.assertRaisesRegex(ImportError, "optional 'macro' extra"),
        ):
            mc.as_caustics_thin_lens(tracer)

    def test_invalid_geometry_is_rejected_before_optional_import(self) -> None:
        tracer = mock.Mock()
        tracer.raytrace = mock.Mock()
        with self.assertRaisesRegex(ValueError, "positive"):
            mc.as_caustics_thin_lens(tracer, microarcseconds_per_unit=0.0)


@unittest.skipUnless(CAUSTICS_AVAILABLE, "requires caustics")
class CausticsThinLensEquivalenceTests(unittest.TestCase):
    def test_adapter_is_a_thin_lens_and_preserves_local_raytrace(self) -> None:
        import caustics

        tracer = _far_field()
        origin = (0.42, -0.17)
        lens = mc.as_caustics_thin_lens(
            tracer,
            cosmology=caustics.FlatLambdaCDM(),
            z_l=0.5,
            z_s=2.0,
            origin_arcsec=origin,
        )
        self.assertIsInstance(lens, caustics.ThinLens)

        x_uas = torch.tensor([-1.6, -0.7, 0.1, 0.8, 1.5], dtype=torch.float64)
        y_uas = torch.tensor([0.4, -1.2, 1.1, -0.3, 0.9], dtype=torch.float64)
        expected_x, expected_y = tracer.raytrace(x_uas, y_uas)
        x_arcsec = origin[0] + x_uas / 1.0e6
        y_arcsec = origin[1] + y_uas / 1.0e6
        actual_x, actual_y = lens.raytrace(x=x_arcsec, y=y_arcsec)
        torch.testing.assert_close(
            (actual_x - origin[0]) * 1.0e6,
            expected_x,
            rtol=0.0,
            # Subtracting a nonzero global origin and converting back to uas
            # magnifies float64 roundoff at the 1e-10 uas level.
            atol=5.0e-10,
        )
        torch.testing.assert_close(
            (actual_y - origin[1]) * 1.0e6,
            expected_y,
            rtol=0.0,
            atol=5.0e-10,
        )

    def test_single_plane_orchestration_is_equivalent(self) -> None:
        import caustics

        tracer = _far_field()
        cosmology = caustics.FlatLambdaCDM()
        lens = mc.as_caustics_thin_lens(
            tracer,
            cosmology=cosmology,
        )
        plane = caustics.SinglePlane(
            cosmology=cosmology,
            lenses=[lens],
            z_l=0.5,
            z_s=2.0,
            name="adapter_equivalence_plane",
        )
        x_uas = torch.tensor([-1.4, -0.4, 0.6, 1.4], dtype=torch.float64)
        y_uas = torch.tensor([1.1, -0.9, -0.2, 0.7], dtype=torch.float64)
        expected_x, expected_y = tracer.raytrace(x_uas, y_uas)
        actual_x, actual_y = plane.raytrace(
            x=x_uas / 1.0e6,
            y=y_uas / 1.0e6,
        )
        torch.testing.assert_close(
            actual_x * 1.0e6,
            expected_x,
            rtol=0.0,
            atol=4.0e-11,
        )
        torch.testing.assert_close(
            actual_y * 1.0e6,
            expected_y,
            rtol=0.0,
            atol=4.0e-11,
        )

    def test_temporal_batch_requires_selecting_one_physical_frame(self) -> None:
        frame = _far_field()
        batch = BatchedTaylorFarFieldApproximation((frame, frame))
        with self.assertRaisesRegex(TypeError, r"far_fields\[frame\]"):
            mc.as_caustics_thin_lens(batch)


if __name__ == "__main__":
    unittest.main()
