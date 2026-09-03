"""Evaluate a physical multiband thin disk independently of microlensing."""

import microcaustics as mc

model = mc.ThinDiskModel(
    black_hole_mass_solar=1.0e9,
    eddington_ratio=0.1,
    bands_angstrom={"u": 3671.0, "g": 4827.0, "r": 6223.0, "i": 7546.0},
    source_grid_shape=256,
    spin=0.7,
    inclination_deg=30.0,
    position_angle_deg=15.0,
    relativity="approximate",
)
disk = model.pixelate(source_redshift=1.7, H0=70.0, Om0=0.3)
curve = mc.source_light_curve(disk, [0.0])
print(disk.geometry.shape)
print(dict(zip(curve.band_names, curve.flux[0].tolist(), strict=True)))
