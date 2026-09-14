from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

import microcaustics as mc


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
    torch.testing.assert_close(
        nt,
        0.5 * mc.novikov_thorne_radiative_efficiency(0.0),
    )
    torch.testing.assert_close(
        ss,
        0.5 * mc.shakura_sunyaev_radiative_efficiency(0.0),
    )


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
