"""Evaluate a delayed thermal disk and its linear transfer functions."""

from __future__ import annotations

import torch

import microcaustics as mc


def main() -> None:
    """Build and evaluate a delayed thermal-reprocessing source."""

    dtype = torch.float64
    mass_solar = 1.0e8
    spin = 0.7
    inclination = 45.0
    screen = mc.ObserverScreen.uniform(
        (96, 96),
        60.0,
        gravitational_radius_m=float(mc.gravitational_radius_m(mass_solar)),
        observer_distance_m=5.0e25,
        dtype=dtype,
    )
    primary = mc.trace_primary_equatorial(
        screen,
        spin=spin,
        inclination_deg=inclination,
        disk_outer_rg=50.0,
    )
    coordinates = mc.add_observer_coordinates(
        primary,
        screen,
        black_hole_mass_solar=mass_solar,
        spin=spin,
        inclination_deg=inclination,
        source_redshift=1.0,
        coordinate_dtype=dtype,
    )
    geometry = mc.SourceGeometry(
        screen.shape,
        (1.0e12, 1.0e12),
        (4000.0, 5500.0, 8000.0),
        ("blue", "green", "red"),
    )
    lamppost = mc.axis_lamppost_profile(
        spin=spin,
        height_above_isco_rg=10.0,
        disk_outer_rg=50.0,
        nalpha=512,
        radial_bins=128,
        dtype=dtype,
    )
    signal = mc.CallableDrivingSignal(
        lambda time: 1.0 + 0.2 * torch.sin(2.0 * torch.pi * time / 30.0),
        name="sinusoid",
    )
    source = mc.ThermalReprocessingSource.from_axis_lamppost(
        geometry,
        coordinates.transfer,
        signal,
        lamppost,
        black_hole_mass_solar=mass_solar,
        eddington_ratio=0.1,
        spin=spin,
        source_redshift=1.0,
        irradiation_efficiency=0.15,
    )
    images = source.brightness(torch.tensor([0.0, 5.0, 10.0], dtype=dtype))
    edges = torch.linspace(0.0, 2.0, 65, dtype=dtype)
    transfer_functions = source.transfer_function(edges)
    print("brightness", tuple(images.shape), images.sum(dim=(1, 2)))
    print("transfer functions", tuple(transfer_functions.shape))


if __name__ == "__main__":
    main()
