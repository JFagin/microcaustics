from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

import microcaustics as mc
from microcaustics.relativity.lamppost import _flat_lamppost_illumination


def _custom_flux_factor(radius_rg, spin, isco_rg):
    del spin
    safe_radius = radius_rg.clamp_min(torch.finfo(radius_rg.dtype).tiny)
    return torch.where(
        radius_rg > isco_rg,
        safe_radius.pow(-3.0)
        * (1.0 - torch.sqrt(isco_rg / safe_radius)).square(),
        torch.zeros_like(radius_rg),
    )


@dataclass(frozen=True)
class WindSuppressedFlux:
    wind_index: float
    transition_rg: float
    name: str = "wind-suppressed"

    def __call__(self, radius_rg, spin, isco_rg):
        baseline = mc.shakura_sunyaev_flux_factor(radius_rg, spin, isco_rg)
        suppression = (1.0 + radius_rg / self.transition_rg).pow(-self.wind_index)
        return baseline * suppression

    def metadata(self):
        return {
            "wind_index": self.wind_index,
            "transition_rg": self.transition_rg,
        }


def test_shakura_sunyaev_flux_factor_matches_analytic_profile() -> None:
    radius = torch.tensor([5.0, 6.0, 12.0, 24.0], dtype=torch.float64)
    factor = mc.shakura_sunyaev_flux_factor(radius, spin=0.0)
    expected = radius.pow(-3.0) * (
        1.0 - torch.sqrt(torch.tensor(6.0, dtype=radius.dtype) / radius)
    )
    expected[:2] = 0.0
    torch.testing.assert_close(factor, expected, rtol=1.0e-14, atol=0.0)


def test_builtin_radiative_efficiencies_are_analytic() -> None:
    spin = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    nt = mc.novikov_thorne_radiative_efficiency(spin)
    ss = mc.shakura_sunyaev_radiative_efficiency(spin)
    torch.testing.assert_close(
        nt,
        torch.tensor(1.0 - math.sqrt(8.0 / 9.0), dtype=torch.float64),
    )
    torch.testing.assert_close(ss, torch.tensor(1.0 / 12.0, dtype=torch.float64))
    (nt + ss).backward()
    assert torch.isfinite(spin.grad)


def test_novikov_thorne_remains_the_default_temperature_profile() -> None:
    radius = torch.logspace(0.8, 3.0, 256, dtype=torch.float64)
    kwargs = {
        "black_hole_mass_solar": 1.0e8,
        "eddington_ratio": 0.1,
        "spin": 0.4,
        "normalization_samples": 1024,
    }
    default, default_isco = mc.thin_disk_temperature4(radius, **kwargs)
    explicit, explicit_isco = mc.thin_disk_temperature4(
        radius,
        viscous_flux_profile="novikov-thorne",
        radiative_efficiency=None,
        **kwargs,
    )
    torch.testing.assert_close(default, explicit, rtol=0.0, atol=0.0)
    torch.testing.assert_close(default_isco, explicit_isco, rtol=0.0, atol=0.0)


def test_beta_normalization_preserves_shakura_sunyaev_power() -> None:
    radius = torch.logspace(-4.0, 6.0, 20_000, dtype=torch.float64)
    widths = torch.empty_like(radius)
    widths[1:-1] = 0.5 * (radius[2:] - radius[:-2])
    widths[0] = radius[1] - radius[0]
    widths[-1] = radius[-1] - radius[-2]
    area_weight = radius * widths
    common = {
        "black_hole_mass_solar": 1.0e8,
        "eddington_ratio": 0.1,
        "spin": 0.6,
        "viscous_flux_profile": "shakura-sunyaev",
    }
    baseline, _ = mc.thin_disk_temperature4(
        radius,
        temperature_slope_beta=0.75,
        **common,
    )
    tilted, _ = mc.thin_disk_temperature4(
        radius,
        temperature_slope_beta=0.6,
        **common,
    )
    torch.testing.assert_close(
        (tilted * area_weight).sum(),
        (baseline * area_weight).sum(),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_custom_profile_requires_and_uses_explicit_efficiency() -> None:
    radius = torch.logspace(0.8, 2.0, 128, dtype=torch.float64)
    common = {
        "black_hole_mass_solar": 1.0e8,
        "eddington_ratio": 0.1,
        "spin": 0.0,
        "viscous_flux_profile": _custom_flux_factor,
        "normalization_samples": 512,
    }
    with pytest.raises(ValueError, match="requires radiative_efficiency"):
        mc.thin_disk_temperature4(radius, **common)
    fixed, _ = mc.thin_disk_temperature4(
        radius,
        radiative_efficiency=0.1,
        **common,
    )
    named, _ = mc.thin_disk_temperature4(
        radius,
        radiative_efficiency="shakura-sunyaev",
        **common,
    )
    assert torch.all(torch.isfinite(fixed))
    assert torch.all(fixed >= 0.0)
    assert not torch.allclose(fixed, named)


def test_custom_profile_can_carry_additional_parameters() -> None:
    profile = WindSuppressedFlux(wind_index=0.35, transition_rg=50.0)
    model = mc.ThinDiskModel(
        black_hole_mass_solar=1.0e8,
        eddington_ratio=0.1,
        bands_angstrom={"r": 6200.0},
        source_redshift=1.0,
        viscous_flux_profile=profile,
        radiative_efficiency=0.1,
        source_grid_shape=32,
    )
    source = model.pixelate(H0=70.0, Om0=0.3)
    metadata = source.metadata()
    assert metadata["viscous_flux_profile"] == "wind-suppressed"
    assert metadata["viscous_flux_profile_metadata"] == {
        "wind_index": 0.35,
        "transition_rg": 50.0,
    }


def test_custom_efficiency_can_be_parameterized_callable() -> None:
    radius = torch.logspace(0.8, 2.0, 128, dtype=torch.float64)

    def scaled_efficiency(spin, isco_rg):
        return 0.8 * mc.shakura_sunyaev_radiative_efficiency(spin, isco_rg)

    temperature4, _ = mc.thin_disk_temperature4(
        radius,
        black_hole_mass_solar=1.0e8,
        eddington_ratio=0.1,
        spin=0.0,
        viscous_flux_profile=WindSuppressedFlux(0.35, 50.0),
        radiative_efficiency=scaled_efficiency,
    )
    assert torch.all(torch.isfinite(temperature4))
    assert torch.any(temperature4 > 0)


def test_model_metadata_records_resolved_profile_and_efficiency() -> None:
    model = mc.ThinDiskModel(
        black_hole_mass_solar=1.0e8,
        eddington_ratio=0.1,
        bands_angstrom={"r": 6200.0},
        source_redshift=1.0,
        viscous_flux_profile="shakura-sunyaev",
        source_grid_shape=32,
    )
    source = model.pixelate(
        H0=70.0,
        Om0=0.3,
    )
    metadata = source.metadata()
    assert metadata["viscous_flux_profile"] == "shakura-sunyaev"
    assert metadata["radiative_efficiency_prescription"] == "profile-default"
    assert metadata["radiative_efficiency"] == pytest.approx(
        1.0 / (2.0 * float(mc.kerr_isco_radius(0.0)))
    )


def test_lamppost_normalization_uses_selected_efficiency() -> None:
    nt = mc.lamppost_irradiation_efficiency(
        0.1,
        0.2,
        0.0,
        viscous_flux_profile="novikov-thorne",
    )
    ss = mc.lamppost_irradiation_efficiency(
        0.1,
        0.2,
        0.0,
        viscous_flux_profile="shakura-sunyaev",
    )
    reflected = mc.lamppost_irradiation_efficiency(0.1, 0.2, 0.0, albedo=0.25)
    torch.testing.assert_close(
        nt,
        0.1 * mc.novikov_thorne_radiative_efficiency(0.0),
    )
    torch.testing.assert_close(
        ss,
        0.1 * mc.shakura_sunyaev_radiative_efficiency(0.0),
    )
    torch.testing.assert_close(reflected, 0.75 * nt)
    with pytest.raises(ValueError, match="albedo"):
        mc.lamppost_irradiation_efficiency(0.1, 0.2, 0.0, albedo=1.0)


def test_flat_lamppost_illumination_puts_half_the_photons_on_one_face() -> None:
    height = 26.0
    radius = torch.logspace(-6.0, 9.0, 200_001, dtype=torch.float64)
    illumination = _flat_lamppost_illumination(radius, height)
    total = torch.trapezoid(illumination * 2.0 * math.pi * radius, radius)
    assert float(total) == pytest.approx(0.5, rel=1e-4)


@pytest.mark.parametrize(
    ("eddington_ratio", "albedo"), [(0.03, 0.0), (0.1, 0.0), (1.0, 0.0), (0.1, 0.3)]
)
def test_lamppost_heating_absorbs_half_the_lamp_luminosity(
    eddington_ratio: float, albedo: float
) -> None:
    from microcaustics.sources.thin_disk import (
        _C,
        _G,
        _M_PROTON,
        _M_SUN,
        _SIGMA_SB,
        _SIGMA_T,
        _accretion_power_temperature4_scale,
    )

    mass, lamp_fraction, height = 1.0e8, 0.1, 26.0
    scale, _ = _accretion_power_temperature4_scale(
        mass, eddington_ratio, 0.0, device="cpu", dtype=torch.float64
    )
    efficiency = mc.lamppost_irradiation_efficiency(
        lamp_fraction, eddington_ratio, 0.0, albedo=albedo
    )
    radius = torch.logspace(-6.0, 9.0, 200_001, dtype=torch.float64)
    heating = efficiency * scale * _flat_lamppost_illumination(radius, height)
    r_g = _G * _M_SUN * mass / _C**2
    absorbed = _SIGMA_SB * torch.trapezoid(heating * 2.0 * math.pi * radius, radius) * r_g**2
    eddington_luminosity = 4.0 * math.pi * _G * _M_SUN * mass * _M_PROTON * _C / _SIGMA_T
    lamp_luminosity = lamp_fraction * eddington_ratio * eddington_luminosity
    assert float(absorbed) == pytest.approx(0.5 * (1.0 - albedo) * lamp_luminosity, rel=2e-3)


def test_newtonian_lamppost_heating_follows_lamp_luminosity_and_albedo() -> None:
    signal = mc.broken_power_law_driving_signal(
        cadence_days=1.0, standard_deviation=0.3, seed=1, history_days=200.0
    )

    def response(eddington_ratio: float, albedo: float) -> torch.Tensor:
        model = mc.ThinDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=eddington_ratio,
            bands_angstrom={"i": 7550.0},
            source_redshift=1.0,
            driving_signal=signal,
            lamp_fraction=0.1,
            lamp_albedo=albedo,
            source_grid_shape=64,
        )
        source = model.pixelate(
            source_redshift=1.0, H0=70.0, Om0=0.3, grid=mc.PlaneGrid((64, 64), (1.0, 1.0))
        )
        return source.response_temperature4.double()

    base = response(0.1, 0.0)
    torch.testing.assert_close(response(0.1, 0.4), 0.6 * base, rtol=1e-5, atol=0.0)
    # The lamp carries a fixed fraction of the accretion power, so its heating
    # grows in proportion to the Eddington ratio.
    torch.testing.assert_close(response(0.3, 0.0), 3.0 * base, rtol=1e-5, atol=0.0)


@pytest.mark.cuda
def test_disk_profiles_run_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    radius = torch.logspace(0.8, 3.0, 1024, device="cuda")
    for profile in ("novikov-thorne", "shakura-sunyaev"):
        temperature4, isco = mc.thin_disk_temperature4(
            radius,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.4,
            viscous_flux_profile=profile,
            normalization_samples=1024,
        )
        assert temperature4.is_cuda
        assert isco.is_cuda
        assert torch.all(torch.isfinite(temperature4))
