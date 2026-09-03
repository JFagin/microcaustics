"""Fast checks of notebook API calls and the self-contained dataset workflow."""

import ast
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import microcaustics as mc

NOTEBOOKS = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


def cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize(
    "path", sorted(NOTEBOOKS.rglob("*.ipynb")), ids=lambda p: p.stem
)
def test_notebook_public_calls_match_current_signatures(path):
    """Check public constructors and class methods, not just top-level calls."""
    for cell in cells(path):
        if cell["cell_type"] != "code":
            continue
        code = "\n".join(
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            attributes = []
            root = node.func
            while isinstance(root, ast.Attribute):
                attributes.append(root.attr)
                root = root.value
            if not isinstance(root, ast.Name) or root.id != "mc":
                continue
            function = mc
            for attribute in reversed(attributes):
                function = getattr(function, attribute)
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                kw.arg is None for kw in node.keywords
            ):
                continue
            # Bind placeholders to check positional arguments too. Keyword-only
            # scans missed the former PointMassField velocity/mass ordering.
            inspect.signature(function).bind(
                *([None] * len(node.args)), **{kw.arg: None for kw in node.keywords}
            )


def test_custom_source_notebook_generates_physical_brightness():
    """Execute the actual source definition and preserve its Jy normalization."""
    pytest.importorskip("matplotlib")
    notebook = cells(NOTEBOOKS / "source_models/02_custom_sources_and_variability.ipynb")
    namespace = {}
    exec(compile("".join(notebook[2]["source"]), "custom source setup", "exec"), namespace)
    source = namespace["system"].realize().source
    geometry = source.geometry
    times = torch.tensor([0.0, 150.0, 300.0])
    images = source.brightness(times)
    assert images.shape == (3, 128, 128, 2)
    assert torch.isfinite(images).all() and (images >= 0).all()
    assert geometry.band_names == ("blue", "red")
    assert geometry.wavelengths_angstrom == (4800.0, 7500.0)
    pixel_area = geometry.pixel_scale_m[0] * geometry.pixel_scale_m[1]
    expected = torch.tensor([1.0, 0.85, 0.70])[:, None] * torch.tensor([20e-6, 15e-6])
    torch.testing.assert_close(images.sum(dim=(1, 2)) * pixel_area, expected)
    assert not torch.allclose(images[0], images[1], atol=0.0)


def test_dataset_tutorial_runs_without_scripts_or_saved_simulations(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from time import perf_counter

    from matplotlib import pyplot as plt

    import microcaustics.plotting as mcp

    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = cells(NOTEBOOKS / "workflows/04_simulation_datasets.ipynb")
    code = "\n".join("".join(c["source"]) for c in notebook if c["cell_type"] == "code")
    for forbidden in (
        "subprocess",
        "training_set_support",
        "np.load",
        "sys.path",
        "__file__",
    ):
        assert forbidden not in code
    namespace = dict(
        mc=mc,
        mcp=mcp,
        torch=torch,
        np=np,
        plt=plt,
        perf_counter=perf_counter,
        source_pixels=16,
        label_pixels=32,
        duration_days=50.0,
        map_cadence_days=25.0,
        source_cadence_days=10.0,
        temporal_batch_size=2,
        scout_refresh_frames=10,
        count=2,
        rays=64,
        bands={"g": 4800.0, "i": 7500.0},
        OUTPUT=tmp_path,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )
    # Execute the actual tutorial definitions, not a second implementation.
    for index in (2, 3, 5, 10):
        tree = ast.parse("".join(notebook[index]["source"]))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOKS), "exec"
            ),
            namespace,
        )
    system = namespace["make_system"](0)
    assert system.seed == 0
    assert system.source.signal.seed is None  # inherited by the system
    assert system.stellar_population is not None
    kinematics = system.stellar_population.kinematics
    assert isinstance(kinematics, mc.SkyProjectedKinematics)
    assert kinematics.include_cmb_dipole
    assert kinematics.stellar_dispersion_km_s > 0
    assert kinematics.peculiar_velocity_dispersion_km_s > 0
    assert system.source.source.grid.shape == (16, 16)
    prior = namespace["sample_parameters"](np.random.default_rng(0))
    assert 0 <= prior["inclination_deg"] <= 60
    assert prior["source_redshift"] > prior["lens_redshift"] + 0.4
    random_system = namespace["make_system"](1, **prior)
    assert random_system.macro.convergence == prior["convergence"]
    random_kinematics = random_system.stellar_population.kinematics
    assert isinstance(random_kinematics, mc.SkyProjectedKinematics)
    assert random_kinematics.include_cmb_dipole
    assert random_kinematics.stellar_dispersion_km_s > 0
    assert random_kinematics.peculiar_velocity_dispersion_km_s > 0
    assert random_kinematics.ra_deg == prior["ra_deg"]
    assert random_kinematics.dec_deg == prior["dec_deg"]

    # A tiny direct field keeps this functional test fast. The notebook itself
    # continues to use its automatic circular Salpeter field at full resolution.
    system = replace(
        system,
        macro=mc.MacroLens(0.0, 0.0),
        stellar_population=None,
        stars=mc.PointMassField([0.0], [0.0], [1e-8]),
    )
    namespace["systems"] = [system, system.with_seed(1)]
    try:
        for index in (3, 5, 17):
            exec(
                compile(
                    "".join(notebook[index]["source"]), f"dataset cell {index}", "exec"
                ),
                namespace,
            )
            if index == 3:
                assert namespace["axes"].shape == (2, 4)
                for axes, curve, centers in zip(
                    namespace["axes"],
                    namespace["curves"],
                    namespace["centers"],
                    strict=True,
                ):
                    assert axes[0].yaxis_inverted()
                    np.testing.assert_allclose(
                        axes[1].lines[0].get_ydata(), np.log10(centers.numpy())
                    )
                    np.testing.assert_allclose(
                        axes[3].lines[0].get_xdata(), curve.labels.times_days
                    )
                    np.testing.assert_allclose(
                        axes[3].lines[0].get_ydata(), curve.labels.center_distances_uas
                    )
            if index == 5:
                axes = namespace["axes"]
                assert axes.shape == (2,)
                assert len(namespace["figure"].axes) == 3  # one shared colorbar
                assert axes[0].images[0].get_clim() == axes[1].images[0].get_clim()
        assert (tmp_path / "five_q2237_b_training_curves.pdf").is_file()
        assert (tmp_path / "five_q2237_b_maps.pdf").is_file()
        assert len(namespace["curves"]) == len(namespace["batch"].light_curves) == 2
        curve = namespace["curves"][0]
        torch.testing.assert_close(
            curve.labels.times_days, torch.tensor([0.0, 25.0, 50.0]), check_dtype=False
        )
        only_micro = namespace["microlensing_fluxes"]([system], [curve])[0]
        direct = system.light_curve(
            duration_days=50,
            map_cadence_days=25,
            source_cadence_days=10,
            apply_driving_signal=False,
            rays=64,
            temporal_batch_size=2,
        )
        torch.testing.assert_close(only_micro, direct.flux)
        with pytest.raises(ValueError, match="label epoch axis"):
            mcp.plot_light_curve_dataset([curve], center_magnifications=[np.ones(6)])
        with pytest.raises(KeyError, match="unknown band"):
            mcp.plot_light_curve_dataset(
                [curve], center_magnifications=[np.ones(3)], band="missing"
            )
        with pytest.raises(ValueError, match="at least one"):
            mcp.plot_labeled_map_gallery([])
    finally:
        plt.close("all")
