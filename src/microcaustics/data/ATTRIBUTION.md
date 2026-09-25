# Bundled spectral data

`bandpasses/lsst2023-*.ecsv` are copied from the BSD-licensed `speclite`
distribution. They represent LSST throughputs tag 1.9, including an airmass
1.2 reference atmosphere. Their original metadata and source URLs are retained
in each ECSV file. The complete license is in `SPECLITE_LICENSE.rst`.

The files in `quasar_spectrum/` are the empirical quasar emission-line,
reddening, and S0 host templates used by the model adapted from Temple et al.
(2021, MNRAS, 508, 737; arXiv:2109.04472). Their numerical contents are
distributed unchanged. The physical-continuum normalization follows the
spectral workflow described by Fagin et al. (2024; arXiv:2410.18423).
The original qsogen MIT license is in `QSOGEN_LICENSE.md`.
