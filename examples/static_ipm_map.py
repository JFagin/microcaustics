"""Generate a static magnification map with configurable tiled IPM."""

from __future__ import annotations

import torch

import microcaustics as mc


def main() -> None:
    """Run a compact IPM example suitable for CPU or GPU installations."""

    system = mc.MicrolensingSystem(
        macro=mc.MacroLens(convergence=0.35, shear=0.15),
        distances=mc.LensingDistances(8.0e24, 1.6e25, 9.0e24),
        stars=mc.PointMassField(
            x_uas=torch.tensor([-1.1, 0.3, 1.4]),
            y_uas=torch.tensor([0.7, -0.4, 0.1]),
            mass_solar=torch.tensor([0.002739, 0.001636, 0.002264]),
        ),
    )
    result = system.magnification_map(
        map_width_uas=2.5,
        map_pixels=64,
        method=mc.IPMConfig(
            rays=4_096,
            scout_ratio=2,
            refinement=2,
            virtual_refinement=4,
            tiled=True,
            cell_chunk_size=4_096,
        ),
    )
    print(result.values)
    print(result.metadata)
    print(result.timing)


if __name__ == "__main__":
    main()
