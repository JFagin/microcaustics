"""Release-level portability, precision, and notebook execution checks."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import microcaustics as mc

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PACKAGE_ROOT / "examples" / "notebooks"


class MinimalInstallationTests(unittest.TestCase):
    def test_core_import_does_not_require_optional_packages(self) -> None:
        blocked = (
            "astropy",
            "caustics",
            "jupyter",
            "lenstronomy",
            "matplotlib",
            "scipy",
        )
        script = f"""
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {blocked!r}:
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, BlockOptional())
import microcaustics
assert microcaustics.MicrolensingSimulation is not None
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tempfile.gettempdir(),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class SolverDTypeAndReproducibilityTests(unittest.TestCase):
    @staticmethod
    def _simulation(dtype: torch.dtype, *, moving: bool = False):
        velocity_x = [1.0e-4, -0.8e-4, 0.5e-4] if moving else None
        velocity_y = [-0.4e-4, 0.6e-4, -0.3e-4] if moving else None
        return mc.MicrolensingSimulation.create(
            mc.MacroLens(convergence=0.12, shear=0.05),
            mc.PointMassField._from_einstein_radii(
                x_uas=torch.tensor([-0.7, 0.1, 0.75], dtype=dtype),
                y_uas=torch.tensor([0.35, -0.5, 0.15], dtype=dtype),
                einstein_radius_uas=torch.tensor([0.16, 0.13, 0.15], dtype=dtype),
                velocity_x_uas_per_day=(
                    None
                    if velocity_x is None
                    else torch.tensor(velocity_x, dtype=dtype)
                ),
                velocity_y_uas_per_day=(
                    None
                    if velocity_y is None
                    else torch.tensor(velocity_y, dtype=dtype)
                ),
            ),
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=dtype),
        )

    def test_portable_irs_and_ipm_float32_match_float64(self) -> None:
        region = mc.PlaneRegion((4.0, 4.0))
        grid = mc.PlaneGrid((32, 32), (1.8, 1.8))
        methods = (
            mc.IRSConfig(
                rays=16_384, far_field_approx=mc.FarFieldApproxConfig(enabled=False)
            ),
            mc.IPMConfig(
                rays=16_384,
                scout_ratio=1,
                refinement=2,
                virtual_refinement=4,
                tiled=False,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
        )
        for method in methods:
            with self.subTest(method=type(method).__name__):
                maps = [
                    self._simulation(dtype).magnification_map(
                        region, grid, method=method
                    )
                    for dtype in (torch.float32, torch.float64)
                ]
                torch.testing.assert_close(
                    maps[0].values.double(),
                    maps[1].values.double(),
                    rtol=2.0e-4,
                    atol=2.0e-4,
                )

    def test_repeated_dynamic_sequence_is_deterministic(self) -> None:
        simulation = self._simulation(torch.float32, moving=True)
        region = mc.PlaneRegion((4.0, 4.0))
        grid = mc.PlaneGrid((24, 24), (1.8, 1.8))
        method = mc.IPMConfig(
            rays=4_096,
            scout_ratio=1,
            refinement=1,
            virtual_refinement=1,
            tiled=False,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        schedule = mc.DynamicConfig(
            temporal_batch_size=2,
            scout_refresh_frames=1,
        )
        first = tuple(
            frame.values.clone()
            for frame in simulation.dynamic_maps(
                region,
                grid,
                (0.0, 5.0, 10.0, 15.0),
                method=method,
                schedule=schedule,
            )
        )
        second = tuple(
            frame.values.clone()
            for frame in simulation.dynamic_maps(
                region,
                grid,
                (0.0, 5.0, 10.0, 15.0),
                method=method,
                schedule=schedule,
            )
        )
        self.assertEqual(len(first), len(second))
        for actual, expected in zip(first, second, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@unittest.skipUnless(
    os.environ.get("MICROCAUSTICS_RUN_LONG_CUDA") == "1" and torch.cuda.is_available(),
    "set MICROCAUSTICS_RUN_LONG_CUDA=1 on CUDA for the repeated-memory audit",
)
class LongCudaStabilityTests(unittest.TestCase):
    def test_repeated_dynamic_runs_do_not_grow_live_tensor_memory(self) -> None:
        simulation = SolverDTypeAndReproducibilityTests._simulation(
            torch.float32, moving=True
        )
        if simulation.runtime.device.type != "cuda":
            simulation = mc.MicrolensingSimulation.create(
                simulation.macro_lens,
                simulation.point_masses,
                runtime=mc.RuntimeConfig(
                    device="cuda", backend="auto", dtype=torch.float32
                ),
            )
        region = mc.PlaneRegion((4.0, 4.0))
        grid = mc.PlaneGrid((48, 48), (1.8, 1.8))
        method = mc.IPMConfig(
            rays=65_536,
            scout_ratio=2,
            refinement=2,
            virtual_refinement=4,
            far_field_approx=mc.FarFieldApproxConfig(
                cells_per_axis=4, nodes_per_cell_axis=4
            ),
        )
        schedule = mc.DynamicConfig(temporal_batch_size=4, scout_refresh_frames=2)

        def run_once() -> None:
            for frame in simulation.dynamic_maps(
                region,
                grid,
                tuple(float(value) for value in range(12)),
                method=method,
                schedule=schedule,
            ):
                _ = float(frame.values.mean().detach().cpu())

        run_once()
        simulation.runtime.synchronize()
        baseline = int(torch.cuda.memory_allocated(simulation.runtime.device))
        for _ in range(8):
            run_once()
        simulation.runtime.synchronize()
        final = int(torch.cuda.memory_allocated(simulation.runtime.device))
        self.assertLessEqual(final, baseline + 8 * 1024**2)


@unittest.skipUnless(
    os.environ.get("MICROCAUSTICS_RUN_NOTEBOOKS") == "1",
    "set MICROCAUSTICS_RUN_NOTEBOOKS=1 to execute the notebook suite",
)
class NotebookExecutionTests(unittest.TestCase):
    _executed_names: tuple[str, ...] = ()
    _cuda_notebook_outputs: dict[str, str] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if (
            importlib.util.find_spec("nbclient") is None
            or importlib.util.find_spec("nbformat") is None
        ):
            raise unittest.SkipTest("install microcaustics[notebooks]")
        import nbformat
        from nbclient import NotebookClient

        cuda_notebooks = {
            "getting_started/01_q2237_production_light_curve_and_gif.ipynb",
            "workflows/02_end_to_end_lensed_quasar.ipynb",
            "validation/02_accuracy_and_performance.ipynb",
            "workflows/03_streaming_and_exporting_results.ipynb",
        }
        executed_names: list[str] = []
        cuda_outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as temporary:
            for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
                notebook = nbformat.read(path, as_version=4)
                executed = NotebookClient(
                    notebook,
                    timeout=1200,
                    kernel_name="python3",
                    resources={"metadata": {"path": temporary}},
                ).execute()
                name = path.relative_to(NOTEBOOK_ROOT).as_posix()
                executed_names.append(name)
                if name in cuda_notebooks:
                    cuda_outputs[name] = json.dumps(executed)
        cls._executed_names = tuple(executed_names)
        cls._cuda_notebook_outputs = cuda_outputs

    def test_notebooks_execute_headlessly(self) -> None:
        expected = tuple(
            path.relative_to(NOTEBOOK_ROOT).as_posix()
            for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
        )
        self.assertTupleEqual(self._executed_names, expected)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for final-notebook execution"
    )
    def test_final_notebooks_report_cuda(self) -> None:
        for name, output_text in self._cuda_notebook_outputs.items():
            with self.subTest(name=name):
                self.assertIn("cuda", output_text.lower())
        self.assertEqual(len(self._cuda_notebook_outputs), 4)


if __name__ == "__main__":
    unittest.main()
