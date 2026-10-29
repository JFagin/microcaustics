"""Small CPU references for spatial microimages, independent of map photometry."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import microcaustics as mc
from microcaustics.microimages import MicroimageRequest


def point_lens_reference(beta):
    """Unit Einstein radius, lens at zero; unsigned fluxes of the two images."""
    u = torch.linalg.vector_norm(beta, dim=-1)
    root = torch.sqrt(u * u + 4)
    positions = torch.stack(((u + root) / 2, (u - root) / 2), -1)
    positions = positions[..., None] * beta[..., None, :] / u[..., None, None]
    mu = (u * u + 2) / (u * root)
    weights = torch.stack(((mu + 1) / 2, (mu - 1) / 2), -1)
    return positions, weights


def single_lens(sigma, cells):
    distances = mc.LensingDistances(1e25, 2e25, 1e25)
    radius_per_solar_mass = float(distances.einstein_radius_uas(1.0))
    stars = mc.PointMassField(
        torch.zeros(1),
        torch.zeros(1),
        mass_solar=torch.tensor([radius_per_solar_mass**-2]),
    ).resolve(distances)
    sim = mc.MicrolensingSimulation.create(
        mc.MacroLens(convergence=0.0, shear=0.0, smooth_matter_fraction=1.0),
        stars,
        runtime=mc.RuntimeConfig(
            device="cpu", backend="torch-eager", strict_backend=True
        ),
    )
    beta = torch.tensor([0.65, 0.2], dtype=torch.float64)
    grid = mc.PlaneGrid((48, 48), (8 * sigma, 8 * sigma), center_uas=(0.2, 0.65))
    geometry = mc.SourceGeometry(
        grid.shape,
        field_of_view_uas=grid.field_of_view_uas,
        bands_angstrom={"i": 7500.0},
    ).resolve(distances)
    x, y = grid.mesh(dtype=torch.float64)
    pixel_flux = torch.exp(-((x - beta[0]) ** 2 + (y - beta[1]) ** 2) / (2 * sigma**2))
    pixel_flux /= pixel_flux.sum()
    source = mc.StaticSource(
        image=(pixel_flux[..., None] / geometry.pixel_area_m2).float(),
        geometry=geometry,
    )
    config = mc.IPMConfig(
        rays=(cells + 1) ** 2,
        tiled=True,
        scout_ratio=1,
        refinement=2,
        virtual_refinement=4,
        dual_scout_scalar_correction=False,
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    )
    # Offset the rectangular grid to avoid putting a ray exactly on the lens.
    image = sim.magnification_map(
        mc.PlaneRegion((4.0, 4.0), center_uas=(0.007, 0.011)),
        grid,
        method=config,
        microimages=MicroimageRequest(source, distances),
    ).microimages
    # Integrate analytic point-lens images over the finite source, independently
    # of IPM. This reference includes the finite-source shift of the centroid.
    pos, mu = point_lens_reference(torch.stack((x, y), -1))
    weights = mu * pixel_flux[..., None]
    branch_flux = weights.sum((0, 1))
    branch_position = (weights[..., None] * pos).sum((0, 1)) / branch_flux[:, None]
    centroid = (weights[..., None] * pos).sum((0, 1, 2)) / branch_flux.sum()
    ny, nx = image.grid.shape
    dy, dx = image.grid.pixel_scale_uas
    xmin, _, ymin, _ = image.grid.bounds_uas
    xy = torch.stack(
        (
            xmin + (image.cell_indices % nx + 0.5) * dx,
            ymin + (image.cell_indices // nx + 0.5) * dy,
        ),
        -1,
    ).double()
    actual_flux, actual_position = [], []
    for positive in (True, False):
        mask = (xy @ beta > 0) == positive
        f = image.flux_jy[mask, 0].double()
        actual_flux.append(f.sum())
        actual_position.append((xy[mask] * f[:, None]).sum(0) / f.sum())
    return dict(
        image=image,
        flux=torch.stack(actual_flux),
        positions=torch.stack(actual_position),
        expected_flux=branch_flux,
        expected_positions=branch_position,
        expected_centroid=centroid,
        beta=beta,
    )


@pytest.fixture(scope="module")
def lens_samples():
    # The source grid stays fixed while the lens-plane grid is refined.
    return [single_lens(0.06, cells) for cells in (24, 48, 96)]


def test_single_lens_image_positions_fluxes_and_centroid(lens_samples):
    sample = lens_samples[-1]
    torch.testing.assert_close(
        sample["flux"], sample["expected_flux"], rtol=0.002, atol=0.0
    )
    torch.testing.assert_close(
        sample["positions"], sample["expected_positions"], rtol=0.0, atol=0.006
    )
    torch.testing.assert_close(
        sample["image"].centroid_uas[0].double(),
        sample["expected_centroid"],
        rtol=0.0,
        atol=0.001,
    )
    positions, flux = point_lens_reference(sample["beta"])
    point_centroid = (positions * flux[:, None]).sum(0) / flux.sum()
    torch.testing.assert_close(
        sample["image"].centroid_uas[0].double(), point_centroid, rtol=0.0, atol=0.01
    )


def test_lens_plane_resolution_improves_flux_and_centroid(lens_samples):
    flux_errors = [
        float(((s["flux"] / s["expected_flux"]) - 1).abs().max()) for s in lens_samples
    ]
    centroid_errors = [
        float(
            torch.linalg.vector_norm(
                s["image"].centroid_uas[0] - s["expected_centroid"]
            )
        )
        for s in lens_samples
    ]
    # Compare endpoints rather than require strictly monotonic cell-center errors.
    assert flux_errors[-1] < flux_errors[0] * 0.5, flux_errors
    assert centroid_errors[-1] < centroid_errors[0] * 0.5, centroid_errors


def test_compact_source_approaches_analytic_point_source_centroid(lens_samples):
    compact = lens_samples[-1]
    extended = single_lens(0.12, 96)
    positions, flux = point_lens_reference(compact["beta"])
    point_centroid = (positions * flux[:, None]).sum(0) / flux.sum()
    compact_error = torch.linalg.vector_norm(
        compact["image"].centroid_uas[0] - point_centroid
    )
    extended_error = torch.linalg.vector_norm(
        extended["image"].centroid_uas[0] - point_centroid
    )
    assert compact_error < 0.4 * extended_error


def test_moving_lens_reused_scout_matches_independent_centroids():
    from test_microimages import setup

    from microcaustics.dynamic import dynamic_maps

    sim, source, distances, grid, region, method = setup("cpu", stars=8)
    times = (0.0, 10.0, 20.0, 30.0, 40.0)
    request = MicroimageRequest(source, distances)
    independent = [
        sim.magnification_map(
            region, grid, method=method, time_days=t, microimages=request
        ).microimages
        for t in times
    ]
    settings = mc.DynamicConfig(
        temporal_batch_size=3,
        scout_refresh_frames=2,
        tuning=mc.AutoTuningConfig(enabled=False),
    )
    frames = list(
        dynamic_maps(
            sim,
            region,
            grid,
            times,
            method=method,
            config=settings,
            microimages=request,
        )
    )
    assert len(frames) == len(times)
    for actual, reference in zip(frames, independent, strict=True):
        torch.testing.assert_close(
            actual.microimages.total_flux_jy,
            reference.total_flux_jy,
            rtol=3e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            actual.microimages.centroid_uas, reference.centroid_uas, rtol=0.0, atol=2e-4
        )
        # The endpoint union can retain faint cells that a single-epoch scout
        # omits. Bound their total image contribution rather than demand equal
        # sparse support or relative agreement with individual zero pixels.
        image_error = (
            (actual.microimages.to_dense() - reference.to_dense()).abs().sum((0, 1))
        )
        assert torch.all(image_error / reference.total_flux_jy < 3e-4)
    assert (
        torch.linalg.vector_norm(
            independent[-1].centroid_uas - independent[0].centroid_uas
        )
        > 1e-3
    )


def crop_frame():
    return mc.MicroimageFrame(
        torch.tensor([11, 23, 34]),
        torch.tensor([[1.0, 0.0], [2.0, 4.0], [0.0, 3.0]]),
        mc.PlaneGrid((6, 8), (3.0, 4.0), center_uas=(10.0, 20.0)),
        0.0,
        0.0,
        (5000.0, 7000.0),
        torch.zeros(2),
    )


@pytest.mark.parametrize("bounds", [(17.0, 20.0, 8.0, 10.0), (18.0, 22.0, 8.5, 11.5)])
def test_crop_clips_bounds_without_changing_multiband_input(bounds):
    frame = crop_frame()
    before = frame.to_dense().clone()
    original_centroid = frame.centroid_uas.clone()
    crop = frame.crop(bounds)
    xmin, _, ymin, _ = frame.grid.bounds_uas
    x0, _, y0, _ = crop.grid.bounds_uas
    dy, dx = frame.grid.pixel_scale_uas
    row, col = round((y0 - ymin) / dy), round((x0 - xmin) / dx)
    ny, nx = crop.grid.shape
    torch.testing.assert_close(crop.to_dense(), before[row : row + ny, col : col + nx])
    torch.testing.assert_close(frame.to_dense(), before)
    torch.testing.assert_close(frame.centroid_uas, original_centroid)
    assert crop.grid.pixel_scale_uas == frame.grid.pixel_scale_uas


@pytest.mark.parametrize("empty", ["no_cells", "zero_flux", "empty_crop"])
def test_empty_image_or_empty_crop_has_zero_flux_and_nan_centroid(empty):
    frame = crop_frame()
    if empty == "no_cells":
        frame = replace(
            frame,
            cell_indices=torch.empty(0, dtype=torch.long),
            flux_jy=torch.empty((0, 2)),
        )
        crop = frame.crop()
    elif empty == "zero_flux":
        crop = replace(frame, flux_jy=torch.zeros_like(frame.flux_jy)).crop()
    else:
        crop = frame.crop((18.0, 18.5, 8.5, 9.0))
    assert crop.to_dense().shape[-1] == 2
    assert not torch.count_nonzero(crop.to_dense())
    assert torch.isnan(crop.centroid_uas).all()


def test_plot_defaults_preserve_native_spacing_without_explicit_bounds():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from microcaustics.plotting import plot_microimages

    frame = crop_frame()
    frame = replace(
        frame,
        grid=mc.PlaneGrid((1002, 1004), (501.0, 502.0)),
        cell_indices=torch.tensor([1005, 1905, 3512]),
    )
    # Must fail if default rendering starts silently merging cells again.
    with patch.object(
        mc.MicroimageFrame, "rebin", side_effect=AssertionError("unexpected binning")
    ):
        fig, ax = plot_microimages(frame, band=1, log=False)
    try:
        artist = ax.images[0]
        values = np.asarray(artist.get_array())
        xmin, xmax, ymin, ymax = artist.get_extent()
        dy, dx = frame.grid.pixel_scale_uas
        assert (xmax - xmin) / values.shape[1] == pytest.approx(dx)
        assert (ymax - ymin) / values.shape[0] == pytest.approx(dy)
        assert values.sum() * dx * dy / 1e6 == pytest.approx(
            float(frame.total_flux_jy[1])
        )
    finally:
        plt.close(fig)
