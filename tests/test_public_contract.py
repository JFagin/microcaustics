"""Tests for the documented, portable public package contract."""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

import torch

import microcaustics as mc

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "microcaustics"


class PublicAPIContractTests(unittest.TestCase):
    def test_every_export_resolves_and_is_documented(self) -> None:
        self.assertEqual(len(mc.__all__), len(set(mc.__all__)))
        for name in mc.__all__:
            with self.subTest(name=name):
                value = getattr(mc, name)
                self.assertIsNotNone(value)
                self.assertTrue(inspect.getdoc(value), f"{name} has no docstring")

    def test_external_research_comparators_are_not_runtime_imports(self) -> None:
        """Luke/Weisenbach and lenstronomy adapters belong only in tests."""

        forbidden_roots = {"lenstronomy", "weisluke", "microlensing"}
        violations: list[str] = []
        for path in SOURCE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                for module in imported:
                    if module.split(".", 1)[0].lower() in forbidden_roots:
                        violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
        self.assertEqual(violations, [])

    def test_general_ipm_parameters_are_validated_not_hard_coded(self) -> None:
        candidates = (
            (1_003, 1, 1, 1),
            (2_501, 2, 2, 4),
            (10_007, 3, 3, 6),
            (20_011, 5, 4, 8),
        )
        for rays, scout, refinement, virtual in candidates:
            with self.subTest(
                rays=rays,
                scout=scout,
                refinement=refinement,
                virtual=virtual,
            ):
                config = mc.IPMConfig(
                    rays=rays,
                    scout_ratio=scout,
                    refinement=refinement,
                    virtual_refinement=virtual,
                    tiled=scout > 1,
                )
                self.assertEqual(config.rays, rays)
                self.assertEqual(config.scout_ratio, scout)
                self.assertEqual(config.refinement, refinement)
                self.assertEqual(config.virtual_refinement, virtual)

    def test_float32_and_float64_are_public_runtime_choices(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                resolved = mc.resolve_runtime(
                    mc.RuntimeConfig(
                        device="cpu",
                        backend="torch-eager",
                        dtype=dtype,
                    )
                )
                self.assertEqual(resolved.dtype, dtype)
                self.assertEqual(resolved.device.type, "cpu")


class DTypeParityTests(unittest.TestCase):
    def test_direct_raytrace_and_jacobian_float32_match_float64(self) -> None:
        positions_x = [-1.3, -0.25, 0.8, 1.6]
        positions_y = [0.7, -1.1, 0.35, -0.55]
        radii = [0.12, 0.08, 0.1, 0.07]
        query_x = torch.linspace(-1.8, 1.8, 127, dtype=torch.float64)
        query_y = 1.4 * torch.sin(query_x * 1.7)

        outputs = {}
        for dtype in (torch.float32, torch.float64):
            simulation = mc.MicrolensingSimulation.create(
                mc.MacroLens(
                    0.31,
                    0.16,
                    shear_angle_rad=0.23,
                    smooth_matter_fraction=0.25,
                ),
                mc.PointMassField._from_einstein_radii(
                    torch.tensor(positions_x, dtype=dtype),
                    torch.tensor(positions_y, dtype=dtype),
                    einstein_radius_uas=torch.tensor(radii, dtype=dtype),
                ),
                runtime=mc.RuntimeConfig(
                    device="cpu",
                    backend="torch-eager",
                    dtype=dtype,
                ),
            )
            source_x, source_y, _ = simulation.raytrace_direct(query_x, query_y)
            determinant, _ = simulation.jacobian_determinant_direct(query_x, query_y)
            outputs[dtype] = (
                source_x.double(),
                source_y.double(),
                determinant.double(),
            )

        for actual, reference in zip(
            outputs[torch.float32], outputs[torch.float64], strict=True
        ):
            torch.testing.assert_close(actual, reference, rtol=2.0e-5, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
