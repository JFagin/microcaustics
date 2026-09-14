"""Keep documentation and example programs synchronized with the public API."""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import re
import unittest
from pathlib import Path

import microcaustics as mc

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "microcaustics"


def _python_blocks(text: str):
    pattern = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    yield from pattern.findall(text)


class DocumentationIntegrityTests(unittest.TestCase):
    def test_retired_far_field_terminology_is_absent(self) -> None:
        """Prevent the removed historical name from returning to public surfaces."""

        retired = "tree" + "point"
        roots = (
            SOURCE_ROOT,
            PACKAGE_ROOT / "docs",
            PACKAGE_ROOT / "examples",
        )
        violations: list[str] = []
        text_suffixes = {".py", ".md", ".ipynb", ".json", ".txt"}
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(PACKAGE_ROOT)
                if retired in path.name.lower():
                    violations.append(f"retired filename: {relative}")
                if (
                    path.suffix.lower() in text_suffixes
                    and retired in path.read_text(encoding="utf-8").lower()
                ):
                    violations.append(f"retired text: {relative}")
        self.assertEqual(violations, [])

    def test_q2237_method_figure_fixtures_are_distributed(self) -> None:
        fixture_root = PACKAGE_ROOT / "examples" / "data" / "q2237b_method_figures"
        expected = {
            "paper_anchor_gauge_method.png",
            "paper_anchor_gauge_method_data.npz",
            "paper_tile_upsampling_schematic.png",
            "paper_tile_upsampling_schematic_data.npz",
            "paper_far_field_schematic.png",
            "paper_far_field_schematic_data.npz",
        }
        self.assertEqual(
            {path.name for path in fixture_root.iterdir() if path.is_file()},
            expected,
        )

    def test_mkdocs_navigation_and_api_targets_are_resolvable(self) -> None:
        """Keep the declared documentation site synchronized with the package."""

        configuration = (PACKAGE_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        referenced = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*\.md", configuration))
        self.assertGreaterEqual(len(referenced), 10)
        missing = sorted(
            name for name in referenced if not (PACKAGE_ROOT / "docs" / name).is_file()
        )
        self.assertEqual(missing, [])

        api = (PACKAGE_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
        modules = re.findall(r"^:::\s+([A-Za-z_][A-Za-z0-9_.]+)\s*$", api, re.MULTILINE)
        self.assertGreaterEqual(len(modules), 5)
        for name in modules:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_public_source_definitions_have_docstrings(self) -> None:
        """Require API documentation beyond the top-level re-export list."""

        missing: list[str] = []
        for path in SOURCE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(PACKAGE_ROOT)
            if not ast.get_docstring(tree):
                missing.append(f"{relative}:1: module")
            for node in tree.body:
                if not isinstance(
                    node,
                    ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
                ) or node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    missing.append(f"{relative}:{node.lineno}: {node.name}")
                if not isinstance(node, ast.ClassDef):
                    continue
                for child in node.body:
                    if (
                        isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                        and not child.name.startswith("_")
                        and not ast.get_docstring(child)
                    ):
                        missing.append(
                            f"{relative}:{child.lineno}: {node.name}.{child.name}"
                        )
        self.assertEqual(missing, [])

    def test_every_python_example_is_valid_syntax(self) -> None:
        examples = tuple((PACKAGE_ROOT / "examples").glob("*.py"))
        self.assertGreaterEqual(len(examples), 10)
        for path in examples:
            with self.subTest(path=path.name):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_documented_python_blocks_compile(self) -> None:
        markdown = (PACKAGE_ROOT / "README.md", *(PACKAGE_ROOT / "docs").glob("*.md"))
        block_count = 0
        for path in markdown:
            for index, block in enumerate(
                _python_blocks(path.read_text(encoding="utf-8"))
            ):
                block_count += 1
                with self.subTest(path=path.name, block=index):
                    compile(block, f"{path}::python-block-{index}", "exec")
        self.assertGreaterEqual(block_count, 10)

    def test_readme_python_blocks_do_not_use_undefined_names(self) -> None:
        """Treat the README tutorial blocks as one executable Python session."""

        defined = set(dir(builtins))
        unresolved: list[str] = []
        readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        for index, block in enumerate(_python_blocks(readme)):
            tree = ast.parse(block, filename=f"README.md::python-block-{index}")
            assigned = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.update(alias.asname or alias.name for alias in node.names)
            available = defined | assigned | imported
            unresolved.extend(
                f"block {index}: {node.id}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in available
            )
            defined.update(assigned)
            defined.update(imported)
        self.assertEqual(sorted(set(unresolved)), [])

    def test_examples_use_public_package_imports(self) -> None:
        violations: list[str] = []
        referenced_public_names: set[str] = set()
        for path in (PACKAGE_ROOT / "examples").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            aliases: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "microcaustics":
                            aliases.add(alias.asname or "microcaustics")
                        elif alias.name.startswith("microcaustics."):
                            violations.append(f"{path.name}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("microcaustics."):
                        violations.append(f"{path.name}: imports from {node.module}")
                elif (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in aliases
                ):
                    referenced_public_names.add(node.attr)
        self.assertEqual(violations, [])
        missing = sorted(referenced_public_names - set(mc.__all__))
        self.assertEqual(missing, [], f"examples use unexported names: {missing}")

    def test_notebooks_are_valid_json_with_compilable_code_cells(self) -> None:
        notebooks = tuple((PACKAGE_ROOT / "examples" / "notebooks").rglob("*.ipynb"))
        self.assertGreaterEqual(len(notebooks), 16)
        code_cells = 0
        for path in notebooks:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["nbformat"], 4)
                self.assertIn("kernelspec", document["metadata"])
                for index, cell in enumerate(document["cells"]):
                    if cell.get("cell_type") != "code":
                        continue
                    code_cells += 1
                    source = "".join(cell.get("source", []))
                    compile(source, f"{path}::cell-{index}", "exec")
        self.assertGreaterEqual(code_cells, len(notebooks) * 2)

    def test_primary_ipm_tutorials_generate_the_approved_method_figure(self) -> None:
        """Keep tutorials 01/02 on the live publication-style renderer."""

        for name in (
            "getting_started/01_q2237_production_light_curve_and_gif.ipynb",
            "getting_started/00_static_maps_and_numerical_methods.ipynb",
        ):
            text = (PACKAGE_ROOT / "examples" / "notebooks" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn("render_live_paper_ipm_schematic(", text)
            self.assertNotIn("paper_tile_upsampling_schematic_data.npz", text)
            self.assertNotIn("paper_far_field_schematic_data.npz", text)

    def test_internal_method_tutorials_do_not_load_paper_run_arrays(self) -> None:
        """Keep package-owned demonstrations reproducible from public APIs."""

        forbidden = (
            "paper_method_map_caustic_comparison_data.npz",
            "final_paper_run_production_v4",
        )
        notebook_root = PACKAGE_ROOT / "examples" / "notebooks"
        external_or_export_notebooks = {
            "validation/00_weisenbach_ipm_visual_validation.ipynb",
            "validation/01_sim5_gr_visual_validation.ipynb",
            "workflows/03_streaming_and_exporting_results.ipynb",
            "workflows/04_simulation_datasets.ipynb",
        }
        for path in notebook_root.rglob("*.ipynb"):
            name = path.relative_to(notebook_root).as_posix()
            if name in external_or_export_notebooks:
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(name=name):
                for value in forbidden:
                    self.assertNotIn(value, text)
                self.assertNotIn("paper_far_field_schematic_data.npz", text)
                self.assertNotIn("paper_anchor_gauge_method_data.npz", text)
                self.assertNotIn("paper_map_rmse_vs_nrays.json", text)

    def test_notebooks_use_current_ipm_selector(self) -> None:
        """The static/dynamic context now selects the scout automatically."""

        for path in (PACKAGE_ROOT / "examples" / "notebooks").rglob("*.ipynb"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("dynamic=True", text)
                self.assertNotIn("dynamic=False", text)

    def test_distributed_notebooks_contain_successful_saved_outputs(self) -> None:
        """Prevent a release from silently shipping cleared tutorial notebooks."""

        notebooks = tuple((PACKAGE_ROOT / "examples" / "notebooks").rglob("*.ipynb"))
        self.assertGreaterEqual(len(notebooks), 16)
        for path in notebooks:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                code_cells = [
                    cell
                    for cell in document["cells"]
                    if cell.get("cell_type") == "code"
                ]
                self.assertTrue(code_cells)
                self.assertTrue(
                    all(cell.get("execution_count") is not None for cell in code_cells),
                    f"{path.name} has unexecuted code cells",
                )
                outputs = [
                    output for cell in code_cells for output in cell.get("outputs", ())
                ]
                self.assertTrue(outputs, f"{path.name} has no saved outputs")
                self.assertFalse(
                    any(output.get("output_type") == "error" for output in outputs),
                    f"{path.name} contains a saved execution error",
                )

    def test_release_notebook_sequence_is_complete(self) -> None:
        root = PACKAGE_ROOT / "examples" / "notebooks"
        names = {path.relative_to(root).as_posix() for path in root.rglob("*.ipynb")}
        required = {
            "getting_started/00_static_maps_and_numerical_methods.ipynb",
            "getting_started/01_q2237_production_light_curve_and_gif.ipynb",
            "getting_started/02_dynamic_maps_and_light_curves.ipynb",
            "methods/00_stellar_populations_and_mass_functions.ipynb",
            "methods/01_far_field_approximation.ipynb",
            "methods/02_caustics_and_labels.ipynb",
            "source_models/00_relativistic_disks_and_reverberation.ipynb",
            "workflows/02_end_to_end_lensed_quasar.ipynb",
            "workflows/03_streaming_and_exporting_results.ipynb",
            "workflows/04_simulation_datasets.ipynb",
            "validation/02_accuracy_and_performance.ipynb",
        }
        self.assertEqual(required - names, set())

    def test_training_notebook_is_self_contained(self) -> None:
        """Generate datasets directly and reserve worker scripts for optional use."""

        name = "workflows/04_simulation_datasets.ipynb"
        notebook = json.loads(
            (PACKAGE_ROOT / "examples" / "notebooks" / name).read_text(
                encoding="utf-8"
            )
        )
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for call in (
            "mc.MicrolensingSystem(",
            ".light_curve(",
            "mc.batched_system_light_curves(",
        ):
            self.assertIn(call, code)
        for dependency in ("subprocess", "training_set_support", "__file__"):
            self.assertNotIn(dependency, code)
        for script in (
            "generate_q2237_training_set.py",
            "generate_random_training_set.py",
        ):
            self.assertNotIn(script, code)
            self.assertIn(script, markdown)


if __name__ == "__main__":
    unittest.main()
