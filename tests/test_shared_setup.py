"""Shared setup preserves source settings and light-curve request semantics."""

from dataclasses import replace

import pytest

import microcaustics as mc
from microcaustics.photometry import LightCurveRequest
from microcaustics.simulation import _light_curve_requests


@pytest.mark.parametrize(
    "evolution", ["exact", "linear_response", "quadratic_response", "cubic_response"]
)
@pytest.mark.parametrize("comparison", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("delay_bin", [None, 0.25])
def test_shared_requests_preserve_settings(evolution, comparison, strict, delay_bin):
    source, distances, trajectory = object(), object(), object()
    mean_source = object() if comparison else None
    actual = _light_curve_requests(
        source,
        distances,
        trajectory,
        strict,
        microlensing_only_source=mean_source,
        source_evolution=evolution,
        response_delay_bin_days=delay_bin,
    )
    expected = [
        LightCurveRequest(
            source,
            distances,
            trajectory,
            strict,
            source_evolution=evolution,
            response_delay_bin_days=delay_bin,
        )
    ]
    if comparison:
        expected.append(LightCurveRequest(mean_source, distances, trajectory, strict))
    assert actual == expected
    if comparison:
        assert actual[1].source_evolution == "exact"
        assert actual[1].response_delay_bin_days is None


def make_model(kind, **kwargs):
    if kind == "gaussian":
        return mc.GaussianModel(sigma_uas=0.3, **kwargs)
    cls = mc.ThinDiskModel if kind == "thin" else mc.KerrDiskModel
    return cls(black_hole_mass_solar=1e8, eddington_ratio=0.1, **kwargs)


@pytest.mark.parametrize("kind", ["thin", "kerr", "gaussian"])
@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_source_sampling_preserves_band_order_and_grid(kind, mapping, override):
    policy = mc.SourceGridConfig(shape=(16, 24), radial_samples=256)
    bands = (
        dict(bands_angstrom={"r": 6200, "g": 4800})
        if mapping
        else dict(wavelengths_angstrom=(6200, 4800), band_names=("r", "g"))
    )
    overrides = (
        dict(source_grid_shape=(24, 32), enclosed_flux_fraction=0.99, source_margin=1.2)
        if override
        else {}
    )
    model = make_model(kind, grid=policy, **bands, **overrides)
    assert model.wavelengths_angstrom == (6200.0, 4800.0)
    assert model.band_names == ("r", "g")
    if override:
        assert model.grid == replace(
            policy, shape=(24, 32), enclosed_flux_fraction=0.99, margin=1.2
        )
    else:
        assert model.grid is policy
    assert policy.shape == (16, 24)
    assert policy.margin == 1.05


@pytest.mark.parametrize("kind", ["thin", "kerr", "gaussian"])
def test_band_errors_still_precede_sampling_errors(kind):
    with pytest.raises(ValueError, match="bands must contain matching"):
        make_model(
            kind, wavelengths_angstrom=(5000,), band_names=(), source_grid_shape=0
        )
    with pytest.raises(ValueError, match="two positive integers"):
        make_model(kind, bands_angstrom={"g": 4800}, source_grid_shape=0)


@pytest.mark.parametrize("kind", ["thin", "kerr", "gaussian"])
def test_band_conflicts_remain_errors(kind):
    with pytest.raises(ValueError, match="bands_angstrom conflicts"):
        make_model(
            kind,
            wavelengths_angstrom=(6200,),
            band_names=("r",),
            bands_angstrom={"g": 4800},
        )
