"""Numerical and provenance contracts for the compiled Torch backend."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import microcaustics as mc
from microcaustics.runtime import RuntimeCapabilities, _torch_compile_supported
from microcaustics.solvers import TaylorFarFieldApproximation


class CompiledRuntimeResolutionTests(unittest.TestCase):
    def test_missing_device_toolchain_falls_back_or_raises_explicitly(self) -> None:
        with patch(
            "microcaustics.runtime._torch_compile_supported",
            return_value=False,
        ):
            automatic = mc.resolve_runtime(
                mc.RuntimeConfig(device="cpu", backend="auto")
            )
            self.assertEqual(automatic.backend, mc.Backend.TORCH_EAGER)
            fallback = mc.resolve_runtime(
                mc.RuntimeConfig(
                    device="cpu",
                    backend="torch-compile",
                    strict_backend=False,
                )
            )
            self.assertEqual(fallback.backend, mc.Backend.TORCH_EAGER)
            self.assertIn("toolchain", fallback.fallback_reason)
            with self.assertRaisesRegex(RuntimeError, "toolchain"):
                mc.resolve_runtime(
                    mc.RuntimeConfig(
                        device="cpu",
                        backend="torch-compile",
                        strict_backend=True,
                    )
                )


def _simulation(backend: str, *, strict: bool) -> mc.MicrolensingSimulation:
    device = torch.device("cuda")
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(
            0.23,
            0.11,
            shear_angle_rad=0.17,
            smooth_matter_fraction=0.3,
        ),
        mc.PointMassField(
            torch.tensor([-0.8, -0.1, 0.65, 1.2], device=device),
            torch.tensor([0.5, -0.7, 0.25, -0.4], device=device),
            torch.tensor([0.12, 0.09, 0.11, 0.08], device=device),
        ),
        runtime=mc.RuntimeConfig(
            device=device,
            backend=backend,
            strict_backend=strict,
        ),
    )


@unittest.skipUnless(
    torch.cuda.is_available()
    and _torch_compile_supported(
        torch.device("cuda"), RuntimeCapabilities.detect()
    ),
    "CUDA torch.compile or its native compiler toolchain is unavailable",
)
class CompiledBackendCudaTests(unittest.TestCase):
    def test_direct_far_field_and_irs_match_eager(self) -> None:
        eager = _simulation("torch-eager", strict=True)
        compiled = _simulation("torch-compile", strict=True)
        x = torch.linspace(-1.45, 1.45, 257, device="cuda")
        y = 0.9 * torch.sin(1.3 * x)

        eager_x, eager_y, eager_trace = eager.raytrace_direct(
            x, y, ray_chunk_size=257, star_chunk_size=4
        )
        actual_x, actual_y, compiled_trace = compiled.raytrace_direct(
            x, y, ray_chunk_size=257, star_chunk_size=4
        )
        torch.testing.assert_close(actual_x, eager_x, rtol=2e-6, atol=3e-7)
        torch.testing.assert_close(actual_y, eager_y, rtol=2e-6, atol=3e-7)
        self.assertEqual(eager_trace.effective_backend, "torch-eager")
        self.assertEqual(compiled_trace.effective_backend, "torch-compile")

        eager_det, _ = eager.jacobian_determinant_direct(
            x, y, ray_chunk_size=257, star_chunk_size=4
        )
        actual_det, compiled_det = compiled.jacobian_determinant_direct(
            x, y, ray_chunk_size=257, star_chunk_size=4
        )
        torch.testing.assert_close(actual_det, eager_det, rtol=3e-6, atol=2e-5)
        self.assertEqual(compiled_det.effective_backend, "torch-compile")

        region = mc.PlaneRegion((4.0, 4.0))
        tree_config = mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
            taylor_order=4,
            center_translation_order=10,
        )
        eager_tree = TaylorFarFieldApproximation(eager, region, tree_config)
        compiled_tree = TaylorFarFieldApproximation(compiled, region, tree_config)
        eager_tree_x, eager_tree_y = eager_tree.raytrace(x, y)
        tree_x, tree_y = compiled_tree.raytrace(x, y)
        torch.testing.assert_close(tree_x, eager_tree_x, rtol=3e-6, atol=7e-7)
        torch.testing.assert_close(tree_y, eager_tree_y, rtol=3e-6, atol=7e-7)
        self.assertEqual(compiled_tree.last_query_backend, "torch-compile")
        torch.testing.assert_close(
            compiled_tree.jacobian_determinant(x, y),
            eager_tree.jacobian_determinant(x, y),
            rtol=3e-6,
            atol=3e-5,
        )
        self.assertEqual(compiled_tree.last_query_backend, "torch-compile")

        grid = mc.PlaneGrid((24, 25), (1.8, 1.9))
        method = mc.IRSConfig(
            rays=4_096,
            ray_chunk_size=4_096,
            star_chunk_size=4,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        eager_map = eager.magnification_map(region, grid, method=method)
        compiled_map = compiled.magnification_map(region, grid, method=method)
        torch.testing.assert_close(
            compiled_map.values,
            eager_map.values,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(compiled_map.metadata["requested_backend"], "torch-compile")
        self.assertEqual(compiled_map.metadata["effective_backend"], "torch-compile")
        self.assertEqual(
            compiled_map.metadata["backend_components"],
            {"raytrace": "torch-compile", "deposition": "torch-compile"},
        )

    def test_ipm_reports_partial_compile_for_portable_exact_rasterizer(self) -> None:
        simulation = _simulation("torch-compile", strict=True)
        result = simulation.magnification_map(
            mc.PlaneRegion((3.0, 3.0)),
            mc.PlaneGrid((8, 8), (1.2, 1.2)),
            method=mc.IPMConfig(
                rays=64,
                scout_ratio=1,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                cell_chunk_size=64,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
        )
        self.assertEqual(result.metadata["requested_backend"], "torch-compile")
        self.assertEqual(result.metadata["effective_backend"], "torch-compile-partial")
        self.assertEqual(
            result.metadata["backend_components"]["rasterization"],
            "python-exact-reference",
        )


if __name__ == "__main__":
    unittest.main()
