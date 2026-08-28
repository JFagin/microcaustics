# microcaustics notebooks

These notebooks use only the installed public package API. Start Jupyter from
the package root after installing `microcaustics[notebooks]`:

```bash
pip install -e ".[notebooks]"
jupyter lab examples/notebooks
```

The distributed notebooks include saved outputs from a complete execution on
an NVIDIA CUDA/Triton runtime. The runtime identity printed in each
performance-sensitive notebook is part of the saved record; rerunning on a
different device replaces those outputs with results from that environment.
Generated GIF, NPZ, PNG, and FITS side products are kept outside the notebook
files and ignored by Git.

The first notebook reports the operating system, PyTorch/CUDA versions, GPU
model, first-call time, and warmed steady-state time. Its first-call number may
reuse an on-disk compiler cache and is therefore not described as pure compile
time.

Triton needs a writable compiler cache. Its ordinary user cache is normally
appropriate. On a managed system, set `TRITON_CACHE_DIR` to a writable local
directory before starting Jupyter; do not place it on a slow network drive.

The Weisenbach-IPM and SIM5 notebooks are optional. Their distributed outputs
show the external comparisons used during validation. To reproduce those
outputs, configure the external products with `MICROCAUSTICS_WEISENBACH_NPZ`,
`MICROCAUSTICS_WEISENBACH_FRAME_NPZ`, and `MICROCAUSTICS_SIM5_DIR`. They
explain what is being compared and stop cleanly when those files are absent.

## Suggested order

1. `00_runtime_and_static_maps` — runtime selection, GPU identity, and honest
   first-call versus warmed timing.
2. `01_q2237_production_light_curve_and_gif` — the primary end-to-end use
   case: a complete 10-year Q2237 image-B-like production calculation with a
   circular Salpeter field, full-Kerr continuum disk, daily light curves with
   and without intrinsic variability, production caustics and center labels,
   and a 147-frame evolving-map GIF.
3. `02_static_irs_and_ipm` — registered $1024^2$ full-field IRS/IPM maps,
   followed by a matched-density rectangular-aperture example that exposes
   rather than renormalizes possible light loss.
4. `03_stellar_populations_and_mass_functions` — Salpeter, Kroupa,
   fixed-mass, custom, and direct-catalog populations; Einstein radii,
   velocities, realized compact convergence, and a genuine GPU batch of
   unrelated static magnification maps using conservative `k=1` scouting.
5. `04_far_field_approximation` — the local-exact complex-Taylor lens
   equation, every `FarFieldApproxConfig` hyperparameter, Q2237 method schematic,
   direct-raytrace validation, and practical accuracy/runtime tuning.
6. `05_dynamic_maps_and_light_curves` — a ten-year moving-lens calculation
   with 147 sparse maps, daily finite-source flux, and selective map retention.
7. `06_caustics_and_labels` — registered magnification, binary-parity,
   nearest-caustic-distance, and signed-winding maps from a complete caustic
   field, configurable production anchor/gauge probes, and a ten-year
   magnification/binary/distance-map GIF.
8. `07_relativistic_disks_and_transfer_functions` — Kerr disk images, a
   fixed-normalization variable-disk GIF, a ten-year series of microlensed
   response functions and mean lags, and a transfer-function GIF. The public
   `KerrDiskModel` keeps screen rotation, ray tracing, observer delays, and
   lamppost normalization inside the scientific package.
9. `08_continuum_reverberation_mapping` — full-Kerr lamppost transfer
   functions and daily multiband continuum-reverberation light curves without
   any microlensing calculation.
10. `09_multi_image_light_curves_and_observations` — a ten-year two-image API
   introduction with optional real Rubin OpSim WFD cadence input.
11. `10_expanding_supernovae` — an evolving photosphere with arbitrary bands.
12. `11_custom_sources_and_variability` — the user-source protocol followed
    by a ten-year dynamic light curve generated from that custom source.
13. `12_weisenbach_ipm_visual_validation` and
    `13_sim5_gr_visual_validation` — optional external validation, including
    independently generated maps, residuals, and light curves on an asserted
    100%-coverage native-frame Weisenbach aperture.
14. `14_realistic_strong_lens_image` — solved macroimages, host/nucleus/lens
    light, band-dependent PSFs, noise, and separate HST-like g/r/i images in
    mag arcsec$^{-2}$.
15. `15_end_to_end_lensed_quasar` — a solved four-image, ten-year system with
    independent Salpeter fields, production-scale dynamic maps, daily disk
    variability, arrival delays, and mock survey observations.
16. `16_accuracy_and_performance` — CUDA-aware convergence measurements,
    first-call versus warmed timing, numerical-parameter selection, dynamic
    fusion, memory reporting, and lossless autotuning.
17. `17_streaming_and_exporting_results` — map observers, finite-source
    streaming, selective map retention, registered NumPy products, and
    optional FITS export.
18. `18_q2237_training_set` — five independently seeded Q2237 image-B labeled
    light curves, first-call versus warmed generation, resumable products,
    and the equivalent ordinary multi-GPU Python command.
19. `19_randomized_training_set` — explicit demonstration priors over local
    lens and accretion-disk parameters, five labeled examples, and a
    multi-GPU sharded generator that reuses compiled kernels per worker.
20. `20_single_point_lens_validation` — numerical IPM magnification, critical
    curve, and caustic compared directly with the analytic point-lens
    equations.
The primary Q2237 tutorial and the final three advanced notebooks are intended
to be run on a CUDA system before a performance-sensitive study. They select
strict Triton execution when CUDA and Triton are available, print the actual
GPU and backend, and otherwise use an explicit portable fallback so their
numerical workflows remain inspectable on CPU-only machines.
