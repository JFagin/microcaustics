"""Distribution metadata, command-line, and opt-in wheel smoke tests."""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import venv
from contextlib import redirect_stdout
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUN_BUILD_SMOKE = os.environ.get("MICROCAUSTICS_RUN_BUILD_SMOKE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class PackagingContractTests(unittest.TestCase):
    def test_package_source_uses_python_310_grammar(self) -> None:
        source_root = PACKAGE_ROOT / "src" / "microcaustics"
        for path in source_root.rglob("*.py"):
            with self.subTest(path=path.relative_to(PACKAGE_ROOT)):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 10),
                )

    def test_core_and_validation_dependencies_are_separated(self) -> None:
        metadata = tomllib.loads(
            (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        core = tuple(metadata["dependencies"])
        self.assertTrue(any(value.startswith("numpy") for value in core))
        self.assertTrue(any(value.startswith("torch") for value in core))
        for external in ("astropy", "lenstronomy", "scipy", "pytest", "hypothesis"):
            self.assertFalse(
                any(value.lower().startswith(external) for value in core),
                f"{external} must not be a mandatory runtime dependency",
            )
        validation = tuple(metadata["optional-dependencies"]["validation"])
        self.assertTrue(any(value.startswith("astropy") for value in validation))
        self.assertTrue(any(value.startswith("lenstronomy") for value in validation))
        self.assertTrue(any(value.startswith("scipy") for value in validation))

    def test_release_version_and_license_metadata_are_consistent(self) -> None:
        metadata = tomllib.loads(
            (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        import microcaustics

        citation = (PACKAGE_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        license_text = (PACKAGE_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(microcaustics.__version__, metadata["version"])
        self.assertIn(f"version: {metadata['version']}", citation)
        self.assertEqual(metadata["license"], "MIT")
        self.assertEqual(metadata["license-files"], ["LICENSE"])
        self.assertIn("MIT License", license_text)

    def test_doctor_cli_returns_machine_readable_runtime_information(self) -> None:
        from microcaustics.cli import main

        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["doctor"])
        payload = json.loads(output.getvalue())
        self.assertEqual(return_code, 0)
        self.assertIn("cuda_available", payload)
        self.assertIn("torch_compile_available", payload)
        self.assertIn("automatic_selection", payload)
        self.assertIn(
            payload["automatic_selection"]["device"].split(":")[0],
            {"cpu", "cuda", "mps"},
        )
        self.assertIn(
            payload["automatic_selection"]["backend"],
            {"torch-eager", "torch-compile", "triton"},
        )


@unittest.skipUnless(
    RUN_BUILD_SMOKE,
    "set MICROCAUSTICS_RUN_BUILD_SMOKE=1 to build and install a wheel",
)
class WheelInstallationTests(unittest.TestCase):
    def test_wheel_installs_and_imports_away_from_the_source_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="microcaustics-wheel-") as temporary:
            root = Path(temporary)
            wheel_dir = root / "wheel"
            wheel_dir.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                ],
                cwd=PACKAGE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            wheels = tuple(wheel_dir.glob("microcaustics-*.whl"))
            self.assertEqual(len(wheels), 1)

            environment = root / "environment"
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(
                environment
            )
            executable = (
                environment / "Scripts" / "python.exe"
                if os.name == "nt"
                else environment / "bin" / "python"
            )
            subprocess.run(
                [
                    str(executable),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    str(wheels[0]),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    str(executable),
                    "-c",
                    (
                        "import json, microcaustics as mc; "
                        "from microcaustics.cli import main; "
                        "assert mc.__version__; "
                        "assert main(['doctor']) == 0"
                    ),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertIn("automatic_selection", payload)


if __name__ == "__main__":
    unittest.main()
