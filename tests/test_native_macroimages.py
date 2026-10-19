"""Native caustics inversion, public conventions, and legacy compatibility."""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch
from _legacy_macro_solver import solve_macroimages as solve_legacy_macroimages

import microcaustics as mc
from microcaustics.strong_lensing import _CausticsGridPrecision, _macroimage_solutions


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_lens(kind, device="cpu"):
    caustics = pytest.importorskip("caustics")
    cosmology = caustics.FlatLambdaCDM().to(dtype=torch.float64, device=device)
    common = dict(cosmology=cosmology, x0=0.0, y0=0.0)
    redshifts = dict(z_l=0.5, z_s=1.5)
    parameters = None
    source = (0.04, -0.02)
    if kind.startswith("point"):
        dynamic = kind == "point_dynamic"
        lens = caustics.Point(**common, **redshifts, Rein=None if dynamic else 0.9)
        if dynamic:
            parameters = torch.tensor([0.9], dtype=torch.float64, device=device)
        source = (0.18, -0.07)
    elif kind == "epl_shear":
        epl = caustics.EPL(
            **common, q=0.7, phi=1.16, Rein=0.87, t=1.0, s=1e-4, name="epl"
        )
        shear = caustics.ExternalShear(
            **common, gamma_1=-0.048, gamma_2=0.0, name="shear"
        )
        lens = caustics.SinglePlane(
            cosmology=cosmology, lenses=[epl, shear], z_l=0.0394, z_s=1.695
        )
    else:
        lens = caustics.SIE(**common, **redshifts, q=0.72, phi=0.4, Rein=0.9)
        if kind == "sie_double":
            source = (0.4, 0.12)
    lens.to(dtype=torch.float64, device=device)
    return lens, parameters, source


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "kind,count",
    [
        ("point", 2),
        ("point_dynamic", 2),
        ("sie_quad", 4),
        ("sie_double", 2),
        ("epl_shear", 4),
    ],
)
def test_native_images_and_caustics_properties(kind, count, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    lens, parameters, source = make_lens(kind, device)
    # The shared high-precision device, not the main runtime dtype, controls
    # this solve. Image finding must be delegated to the supplied lens.
    runtime = mc.RuntimeConfig(
        device=device, dtype="float32", high_precision_device=device
    )
    with patch.object(lens, "forward_raytrace", wraps=lens.forward_raytrace) as native:
        solutions = mc.solve_caustics_macroimages(
            lens,
            *source,
            parameters=parameters,
            runtime=runtime,
            field_of_view_arcsec=2.0,
            image_names=tuple(f"image{i}" for i in range(count)),
        )
    assert 1 <= native.call_count <= 3
    assert [call.kwargs["divisions"] for call in native.call_args_list] == list(
        range(100, 100 + native.call_count)
    )
    assert len(solutions) == count
    assert [s.name for s in solutions] == [f"image{i}" for i in range(count)]
    assert max(s.source_residual_arcsec for s in solutions) <= 1e-8
    assert solutions[0].arrival_time_delay_days == 0
    assert all(
        a.arrival_time_delay_days <= b.arrival_time_delay_days
        for a, b in zip(solutions, solutions[1:], strict=False)
    )
    positions = torch.tensor(
        [[s.x_arcsec, s.y_arcsec] for s in solutions],
        dtype=torch.float64,
        device=device,
    )
    kwargs = {} if parameters is None else {"params": parameters}
    x, y = positions.unbind(-1)
    delays = lens.time_delay(x, y, **kwargs)
    magnifications = lens.magnification(x, y, **kwargs)
    jacobian = lens.jacobian_lens_equation(x, y, **kwargs)
    for index, solution in enumerate(solutions):
        assert solution.absolute_time_delay_days == pytest.approx(
            float(delays[index]), abs=1e-7
        )
        assert abs(solution.macro_magnification) == pytest.approx(
            float(magnifications[index]), rel=1e-7
        )
        assert solution.parity == int(torch.sign(torch.linalg.det(jacobian[index])))
    if kind.startswith("point"):
        # Analytic point-lens image positions, independent of either solver.
        radius = math.hypot(*source)
        # Static caustics scalars were initially stored in float32.
        rein = 0.9 if parameters is not None else float(torch.tensor(0.9))
        for sign in (1, -1):
            r = 0.5 * (radius + sign * math.sqrt(radius**2 + 4 * rein**2))
            expected = positions.new_tensor(source) * (r / radius)
            assert (
                float(torch.linalg.vector_norm(positions - expected, dim=-1).min())
                < 1e-7
            )


@pytest.mark.parametrize("kind", ["point", "sie_quad", "epl_shear"])
def test_native_matches_frozen_internal_solver(kind):
    lens, parameters, source = make_lens(kind)
    kwargs = dict(parameters=parameters, field_of_view_arcsec=2.0)
    native = mc.solve_caustics_macroimages(lens, *source, **kwargs)
    legacy = solve_legacy_macroimages(
        mc.CausticsMacroModel(lens, parameters),
        *source,
        deduplication_tolerance_arcsec=1e-5,
        field_of_view_arcsec=2.0,
    )
    assert len(native) == len(legacy)
    for first, second in zip(native, legacy, strict=True):
        for field in (
            "x_arcsec",
            "y_arcsec",
            "arrival_time_delay_days",
            "macro_magnification",
            "convergence",
            "shear_gamma1",
            "shear_gamma2",
        ):
            assert getattr(first, field) == pytest.approx(
                getattr(second, field), rel=1e-6, abs=1e-6
            )


@pytest.mark.parametrize("inside,count", [(True, 4), (False, 2)])
def test_fold_pair_is_not_merged(inside, count):
    scipy = pytest.importorskip("scipy.optimize")
    lens, _, _ = make_lens("sie_quad")
    direction = torch.tensor([math.cos(0.9), math.sin(0.9)], dtype=torch.float64)

    def determinant(radius):
        x, y = radius * direction
        return float(torch.linalg.det(lens.jacobian_lens_equation(x, y)))

    critical = scipy.brentq(determinant, 0.2, 2.0) * direction
    beta = torch.stack(lens.raytrace(*critical))
    beta *= 1 - 1e-4 if inside else 1 + 1e-4
    solutions = mc.solve_caustics_macroimages(
        lens, *beta.tolist(), field_of_view_arcsec=2.0
    )
    assert len(solutions) == count
    assert max(s.source_residual_arcsec for s in solutions) < 1e-8
    if inside:
        positions = torch.tensor([[s.x_arcsec, s.y_arcsec] for s in solutions])
        assert 0.001 < float(torch.pdist(positions).min()) < 0.02


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_softened_sis_central_image_has_analytic_root(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    caustics = pytest.importorskip("caustics")
    lens = caustics.SIS(
        caustics.FlatLambdaCDM(), z_l=0.5, z_s=1.0, x0=0.0, y0=0.0, Rein=0.7, s=0.01
    ).to(dtype=torch.float64, device=device)
    solutions = mc.solve_caustics_macroimages(
        lens,
        0.1,
        0.0,
        initial_grid_size=31,
        field_of_view_arcsec=1.2,
        include_central_images=True,
        device=device,
    )
    # On the negative axis the softened SIS equation is
    # x^2 + (Rein - s - beta) x + beta*s = 0.
    coefficient = float(torch.tensor(0.7)) - 0.01 - 0.1
    roots = [
        (-coefficient + sign * math.sqrt(coefficient**2 - 4 * 0.1 * 0.01)) / 2
        for sign in (-1, 1)
    ]
    assert len(solutions) == 3
    for root in roots:
        assert min(abs(s.x_arcsec - root) for s in solutions) < 1e-8
    central = min(solutions, key=lambda s: abs(s.x_arcsec))
    assert 0 < central.macro_magnification < 0.001
    observable = mc.solve_caustics_macroimages(
        lens,
        0.1,
        0.0,
        initial_grid_size=31,
        field_of_view_arcsec=1.2,
        image_names=("first", "second"),
        device=device,
    )
    assert len(observable) == 2
    for filtered, original in zip(observable, solutions[:2], strict=True):
        assert filtered.x_arcsec == original.x_arcsec
        assert filtered.macro_magnification == original.macro_magnification
        assert filtered.arrival_time_delay_days == original.arrival_time_delay_days
    assert [s.name for s in observable] == ["first", "second"]


def test_native_caustics_dictionary_parameters():
    lens, parameters, source = make_lens("point_dynamic")
    vector_result = mc.solve_caustics_macroimages(lens, *source, parameters=parameters)
    dictionary_result = mc.solve_caustics_macroimages(
        lens, *source, parameters={"Rein": parameters[0]}
    )
    assert dictionary_result == vector_result


def test_point_lens_delay_matches_analytic_potential_and_astropy():
    caustics = pytest.importorskip("caustics")
    cosmology_module = pytest.importorskip("astropy.cosmology")
    units = pytest.importorskip("astropy.units")
    constants = pytest.importorskip("astropy.constants")
    cosmology = caustics.FlatLambdaCDM(
        h0=torch.tensor(0.7, dtype=torch.float64),
        Om0=torch.tensor(0.3, dtype=torch.float64),
    ).to(dtype=torch.float64)
    lens = caustics.Point(
        cosmology,
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=torch.tensor(0.9, dtype=torch.float64),
    ).to(dtype=torch.float64)
    solutions = mc.solve_caustics_macroimages(lens, 0.18, -0.07)
    beta = torch.tensor([0.18, -0.07], dtype=torch.float64)
    positions = torch.tensor(
        [[s.x_arcsec, s.y_arcsec] for s in solutions], dtype=beta.dtype
    )
    potential = 0.5 * (positions - beta).square().sum(-1) - 0.9**2 * torch.log(
        torch.linalg.vector_norm(positions, dim=-1)
    )
    oracle = cosmology_module.FlatLambdaCDM(H0=70.0, Om0=0.3)
    distance = (
        1.5
        * oracle.angular_diameter_distance(0.5)
        * oracle.angular_diameter_distance(1.5)
        / oracle.angular_diameter_distance_z1z2(0.5, 1.5)
    )
    days_per_arcsec_squared = (distance / constants.c).to_value(
        units.day
    ) * units.arcsec.to(units.rad) ** 2
    expected = (potential - potential.min()) * days_per_arcsec_squared
    actual = expected.new_tensor([s.arrival_time_delay_days for s in solutions])
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)


def test_retired_solver_is_not_part_of_the_package():
    import microcaustics.strong_lensing as strong_lensing

    for name in ("MacroModel", "CallableMacroModel", "solve_macroimages"):
        assert not hasattr(mc, name)
        assert not hasattr(strong_lensing, name)
        assert name not in mc.__all__


@pytest.mark.parametrize("solver", ["legacy", "caustics"])
def test_retired_solver_selection_is_rejected(solver):
    with pytest.raises(TypeError, match="solver"):
        mc.solve_caustics_macroimages(IdentityLens(), 0.1, 0.2, solver=solver)


class IdentityLens:
    """Record forwarding without needing the optional caustics dependency."""

    def forward_raytrace(self, bx, by, **kwargs):
        self.search = (bx, by, kwargs)
        assert torch.linspace(0, 1, 3).dtype == torch.float64
        return bx[None], by[None]

    def raytrace(self, x, y, **kwargs):
        self.ray_parameters = kwargs
        return x, y

    def jacobian_lens_equation(self, x, y, **kwargs):
        self.jacobian_parameters = kwargs
        return torch.eye(2, dtype=x.dtype, device=x.device).expand(len(x), 2, 2)

    def time_delay(self, x, y, **kwargs):
        self.delay_parameters = kwargs
        return x + y


@pytest.mark.parametrize("parameters", [None, torch.empty(0), {"lens": [1.0]}])
def test_native_options_and_separate_delay_parameters(parameters):
    lens = IdentityLens()
    delay_parameters = {"redshift": 0.5}
    solutions = mc.solve_caustics_macroimages(
        lens,
        0.1,
        0.2,
        parameters=parameters,
        time_delay_parameters=delay_parameters,
        field_of_view_arcsec=2.0,
        initial_grid_size=50,
        max_depth=30,
        runtime=mc.RuntimeConfig(device="cuda", high_precision_device="cuda"),
        device="cpu",  # explicit override also works on a CPU-only machine
    )
    bx, _, options = lens.search
    assert bx.dtype == torch.float64 and bx.device.type == "cpu"
    assert options["fov"] == 4.0 and options["divisions"] == 50
    assert options["max_depth"] == 30 and options["epsilon"] == 1e-8
    expected = {"params": parameters} if isinstance(parameters, dict) else {}
    assert lens.ray_parameters == lens.jacobian_parameters == expected
    assert {key: value for key, value in options.items() if key == "params"} == expected
    assert lens.delay_parameters == {"params": delay_parameters}
    assert solutions[0].macro_magnification == 1


@pytest.mark.parametrize("failed_attempts", [1, 2])
def test_singular_native_refinement_retries_only_the_initial_grid(failed_attempts):
    lens = IdentityLens()
    native = lens.forward_raytrace
    attempts = []
    filters = warnings.filters[:]
    default_dtype = torch.get_default_dtype()

    def search(bx, by, **kwargs):
        attempts.append(kwargs.copy())
        if len(attempts) <= failed_attempts:
            # Produce the same public exception as the native LM solve.
            torch.linalg.solve(torch.zeros(2, 2), torch.ones(2))
        return native(bx, by, **kwargs)

    with (
        patch.object(lens, "forward_raytrace", side_effect=search),
        pytest.warns(RuntimeWarning, match="Retrying its native search"),
    ):
        result = mc.solve_caustics_macroimages(
            lens, 0.1, 0.2, initial_grid_size=50, field_of_view_arcsec=2.0,
            max_depth=30, warn_on_nonconvergence=False,
        )
    assert [a["divisions"] for a in attempts] == list(range(50, 51 + failed_attempts))
    for attempt in attempts:
        assert attempt["fov"] == 4.0
        assert attempt["epsilon"] == 1e-8
        assert attempt["max_depth"] == 30
    assert result[0].source_residual_arcsec == 0
    assert warnings.filters == filters
    assert torch.get_default_dtype() == default_dtype


@pytest.mark.parametrize("failure,count", [("singular", 3), ("other", 1)])
def test_native_refinement_retries_are_bounded_and_do_not_hide_other_errors(failure, count):
    lens = IdentityLens()
    error = torch.linalg.LinAlgError(
        "input matrix is singular" if failure == "singular" else "invalid solve"
    )
    with (
        patch.object(lens, "forward_raytrace", side_effect=error) as native,
        warnings.catch_warnings(record=True),
        pytest.raises(RuntimeError, match="caustics macroimage search failed") as caught,
    ):
        mc.solve_caustics_macroimages(lens, 0.1, 0.2)
    assert native.call_count == count
    assert caught.value.__cause__ is error


def test_successful_retry_still_must_meet_original_residual_tolerance():
    lens = IdentityLens()
    with (
        patch.object(lens, "forward_raytrace", side_effect=[
            torch.linalg.LinAlgError("input matrix is singular"),
            (torch.tensor([0.2], dtype=torch.float64), torch.tensor([0.2], dtype=torch.float64)),
        ]) as native,
        pytest.warns(RuntimeWarning, match="Retrying its native search"),
        pytest.raises(RuntimeError, match="residual tolerance"),
    ):
        mc.solve_caustics_macroimages(lens, 0.1, 0.2)
    assert native.call_count == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_restarted_epl_search_preserves_images_and_native_implementation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    lens, _, source = make_lens("epl_shear", device)
    import caustics.lenses.func.base as native_base
    import caustics.utils as native_utils

    native = lens.forward_raytrace
    original_search = native_base.forward_raytrace
    original_refine = native_base.forward_raytrace_rootfind
    original_lm = native_utils.batch_lm
    reference = mc.solve_caustics_macroimages(lens, *source, device=device)
    attempts = 0

    def search(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.linalg.LinAlgError("input matrix is singular")
        return native(*args, **kwargs)

    with (
        patch.object(lens, "forward_raytrace", side_effect=search),
        pytest.warns(RuntimeWarning, match="Retrying its native search"),
    ):
        result = mc.solve_caustics_macroimages(lens, *source, device=device)
    assert len(result) == len(reference) == 4
    for actual, expected in zip(result, reference, strict=True):
        assert actual.parity == expected.parity
        assert actual.source_residual_arcsec <= 1e-8
        for field in ("x_arcsec", "y_arcsec", "macro_magnification", "arrival_time_delay_days"):
            assert getattr(actual, field) == pytest.approx(getattr(expected, field), abs=1e-6)
    assert native_base.forward_raytrace is original_search
    assert native_base.forward_raytrace_rootfind is original_refine
    assert native_utils.batch_lm is original_lm


def test_scoped_precision_is_thread_local_and_restored_after_error():
    default = torch.get_default_dtype()
    linspace = torch.linspace
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="test failure"),
        _CausticsGridPrecision(torch.float64, torch.device("cpu")),
    ):
        assert torch.linspace(0, 1, 3).dtype == torch.float64
        assert torch.linspace(0, 1, 3, dtype=torch.float32).dtype == torch.float32
        assert pool.submit(lambda: torch.linspace(0, 1, 3).dtype).result() == default
        raise RuntimeError("test failure")
    assert torch.linspace is linspace
    assert torch.get_default_dtype() == default
    assert torch.linspace(0, 1, 3).dtype == default


def test_native_failures_and_warnings_are_not_hidden():
    lens = IdentityLens()
    error = IndexError("empty native search")
    with patch.object(lens, "forward_raytrace", side_effect=error):
        with pytest.raises(
            RuntimeError, match="caustics macroimage search failed"
        ) as caught:
            mc.solve_caustics_macroimages(lens, 0.1, 0.2)
        assert caught.value.__cause__ is error
    for coordinates in [
        (torch.empty(0), torch.empty(0)),
        (torch.tensor([float("nan")]), torch.tensor([0.2])),
        (torch.tensor([0.3]), torch.tensor([0.2])),
    ]:
        with (
            patch.object(lens, "forward_raytrace", return_value=coordinates),
            pytest.raises(RuntimeError, match="coordinates|residual tolerance"),
        ):
            mc.solve_caustics_macroimages(lens, 0.1, 0.2)

    def warned(bx, by, **kwargs):
        warnings.warn("native search incomplete", UserWarning, stacklevel=2)
        return bx[None], by[None]

    with (
        patch.object(lens, "forward_raytrace", side_effect=warned),
        pytest.warns(UserWarning, match="incomplete"),
    ):
        assert len(mc.solve_caustics_macroimages(lens, 0.1, 0.2)) == 1
    with pytest.raises(ValueError, match="image_names"):
        mc.solve_caustics_macroimages(lens, 0.1, 0.2, image_names=("A", "B"))


@pytest.mark.parametrize(
    "options",
    [
        {"initial_grid_size": 1},
        {"field_of_view_arcsec": float("nan")},
        {"refinement_tolerance_arcsec": 0},
        {"max_depth": 0},
        {"dtype": torch.float16},
    ],
)
def test_invalid_native_options(options):
    with pytest.raises(ValueError):
        mc.solve_caustics_macroimages(IdentityLens(), 0.1, 0.2, **options)


def test_legacy_only_options_are_not_silently_ignored():
    with pytest.raises(TypeError, match="cluster_tolerance_arcsec"):
        mc.solve_caustics_macroimages(
            IdentityLens(), 0.1, 0.2, cluster_tolerance_arcsec=0.01
        )


def test_empty_real_search_has_context():
    lens, _, _ = make_lens("sie_quad")
    with pytest.raises(RuntimeError, match="caustics macroimage search failed"):
        mc.solve_caustics_macroimages(lens, 100.0, 100.0, field_of_view_arcsec=2.0)


def test_image_filter_keeps_minima_saddles_and_asymmetric_jacobians():
    model = IdentityLens()
    x = torch.arange(5, dtype=torch.float64)
    y = torch.zeros_like(x)
    jacobians = torch.tensor(
        [
            [[100.0, 0.0], [0.0, 100.0]],  # faint minimum at the origin
            [[-100.0, 0.0], [0.0, 100.0]],  # faint saddle
            [[-2.0, 0.0], [0.0, -3.0]],  # maximum away from the origin
            [[-2.0, 1.0], [0.0, -3.0]],  # asymmetric, not classifiable here
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=x.dtype,
    )
    # Caustics adapter supplies the protocol's time_delay_days method.
    adapted = mc.CausticsMacroModel(model, None)
    with patch.object(model, "jacobian_lens_equation", return_value=jacobians):
        result = _macroimage_solutions(adapted, x, y, y, None)
        complete = _macroimage_solutions(
            adapted, x, y, y, None, include_central_images=True
        )
    assert [s.x_arcsec for s in result] == [0, 1, 3, 4]
    assert len(complete) == 5
    assert [s.name for s in result] == ["A", "B", "C", "D"]
    assert [s.local_microlensing_supported for s in result] == [True, True, False, True]
    with pytest.warns(RuntimeWarning, match="not currently supported"):
        result[2].local_macro_lens()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result[3].local_macro_lens()
    assert not caught


def test_all_maxima_selection_and_delay_zero_after_filtering():
    lens = IdentityLens()
    model = mc.CausticsMacroModel(lens, None)
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y = torch.zeros_like(x)
    jacobians = torch.tensor(
        [[[-1.0, 0.0], [0.0, -1.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=x.dtype
    )
    with patch.object(lens, "jacobian_lens_equation", return_value=jacobians):
        result = _macroimage_solutions(model, x, y, y, ("remaining",))
    assert len(result) == 1
    assert result[0].absolute_time_delay_days == 1
    assert result[0].arrival_time_delay_days == 0
    with patch.object(lens, "jacobian_lens_equation", return_value=jacobians[:1]):
        with pytest.raises(RuntimeError, match="include_central_images=True"):
            _macroimage_solutions(model, x[:1], y[:1], y[:1], None)
        assert (
            len(
                _macroimage_solutions(
                    model, x[:1], y[:1], y[:1], None, include_central_images=True
                )
            )
            == 1
        )


@pytest.mark.parametrize("include_central_images", [False, True])
def test_multiplane_jacobian_is_not_used_to_remove_images(include_central_images):
    caustics = pytest.importorskip("caustics")
    lens = caustics.Multiplane(caustics.FlatLambdaCDM(), (), z_s=1.5)
    model = mc.CausticsMacroModel(lens, None)
    x = torch.ones(1, dtype=torch.float64)
    jacobian = -torch.eye(2, dtype=x.dtype)[None]
    with (
        patch.object(lens, "jacobian_lens_equation", return_value=jacobian),
        patch.object(lens, "time_delay", return_value=x),
    ):
        result = _macroimage_solutions(
            model, x, x, torch.zeros_like(x), None,
            include_central_images=include_central_images,
        )
    assert len(result) == 1
    assert not result[0].local_microlensing_supported
    with pytest.warns(RuntimeWarning, match="not currently supported"):
        local = result[0].local_macro_lens(smooth_matter_fraction=0.2)
    assert local.convergence == result[0].convergence
    assert local.shear == result[0].shear
    assert local.smooth_matter_fraction == 0.2
    with pytest.warns(RuntimeWarning, match="not currently supported"):
        system = mc.MultiImageSystem.from_macroimage_solutions(
            result,
            lens_redshift=0.5,
            source_redshift=1.5,
            source_grid=mc.PlaneGrid((8, 8), (1.0, 1.0)),
            stellar_population=mc.StellarPopulation.salpeter(),
        )
    assert system.image("A").macro.convergence == result[0].convergence


def test_real_epl_warning_opt_out_preserves_solutions_and_filters():
    lens, _, source = make_lens("epl_shear")
    with pytest.warns(UserWarning, match="Forward raytrace unable"):
        reference = mc.solve_caustics_macroimages(lens, *source)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        original_filters = warnings.filters[:]
        quiet = mc.solve_caustics_macroimages(
            lens, *source, warn_on_nonconvergence=False
        )
        assert warnings.filters == original_filters
    assert quiet == reference
    assert not any("Forward raytrace unable" in str(w.message) for w in caught)


@pytest.mark.parametrize("failure", [None, "residual", "exception"])
def test_warning_opt_out_is_narrow_and_does_not_hide_failure(failure):
    lens = IdentityLens()
    message = (
        "Forward raytrace unable to converge reliably, reached maximum depth "
        "of 25 iterations. There may be singularities in the lens, or other numerical challenges."
    )

    def search(bx, by, **kwargs):
        for text, module in (
            (message, "caustics.lenses.func.base"),
            ("different warning", "caustics.lenses.func.base"),
            (message, "unrelated_module"),
        ):
            warnings.warn_explicit(
                text, UserWarning, filename="test.py", lineno=1, module=module
            )
        if failure == "exception":
            raise RuntimeError("native failure")
        return (bx + (1 if failure == "residual" else 0))[None], by[None]

    with (
        patch.object(lens, "forward_raytrace", side_effect=search),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        original_filters = warnings.filters[:]
        if failure is None:
            mc.solve_caustics_macroimages(lens, 0.1, 0.2, warn_on_nonconvergence=False)
        else:
            with pytest.raises(RuntimeError):
                mc.solve_caustics_macroimages(
                    lens, 0.1, 0.2, warn_on_nonconvergence=False
                )
        assert warnings.filters == original_filters
    assert [str(w.message) for w in caught] == ["different warning", message]
