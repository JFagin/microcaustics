# Example scripts

The tutorial notebooks are the best place to learn `microcaustics`. These
standalone scripts provide shorter examples that can be run from a terminal or
adapted for a batch job.

Install the package from the repository root before running a script.

```bash
python -m pip install -e ".[plot,science]"
python examples/static_ipm_map.py
```

## Maps and dynamic calculations

- `static_ipm_map.py` generates an independent IPM magnification map.
- `static_irs_map.py` generates an independent uniform-grid IRS map.
- `dynamic_finite_source.py` combines moving maps with a finite source.
- `dynamic_labels.py` adds source-center caustic labels to a dynamic sequence.
- `automatic_tuning.py` selects safe temporal and spatial batch sizes.
- `batched_light_curves.py` batches several sources through one shared map sequence.

The static scripts use `map_width_uas` and `map_pixels`, so a centered square
map does not require a geometry object. Advanced validation examples retain an
explicit `PlaneGrid` when they need rectangular, off-center, or exactly
registered grids.

## Sources and relativistic calculations

- `thin_disk_source.py` constructs a nonrelativistic thin disk.
- `kerr_thin_disk_source.py` constructs a full-Kerr disk.
- `delayed_kerr_variability.py` adds delayed intrinsic variability.
- `thermal_reprocessing.py` evaluates nonlinear thermal reprocessing.
- `supernova_source.py` constructs an expanding photosphere.

## Multi-image and survey workflows

- `multi_image_light_curves.py` evaluates several macroimages with a shared source.
- `multirate_lensed_quasar.py` combines sparse microlensing maps with finer source evolution.
- `macro_image_rendering.py` renders an instrument-independent strong-lens scene.

## Dataset generation

- `generate_q2237_training_set.py` produces independent realizations of one system.
- `generate_random_training_set.py` samples lens and source parameters from documented priors.
- `training_set_support.py` contains their shared worker and plotting utilities.

Both dataset commands accept one or more CUDA device indices. Each worker stays
alive across several realizations so compatible compiled kernels are reused.
`--curves-per-batch` additionally runs that many complete independent systems
concurrently on each GPU. It is separate from shared-map source batching and
automatically backs off after a CUDA OOM.
Run either command with `--help` to see its output, batching, seed, and device
options.

The complete notebook guide is available in
[`notebooks/README.md`](notebooks/README.md).
