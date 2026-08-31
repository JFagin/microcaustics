# Test and validation suites

The ordinary suite is self-contained and does not require paper outputs or
external research codes:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

With the package's test extras installed, the same suite can be run with
coverage:

```powershell
python -m pytest
python -m pytest --cov=microcaustics --cov-report=term-missing
```

The coverage configuration enforces at least 80% portable-source line
coverage. Triton kernel bodies are excluded from that line metric because they
are compiled rather than executed by the Python interpreter; their numerical
behavior is instead covered by strict CUDA parity tests against the readable
Torch/reference implementations.

Documentation tests require every public module, class, function, and method
to have a docstring. They also validate all Markdown Python blocks, scripts,
notebook code cells, MkDocs navigation entries, and generated API-reference
module targets.

CUDA/Triton tests skip automatically when a compatible CUDA runtime is not
available. They compare the fused kernels with the readable Torch paths for
ray tracing, Taylor far-field approximation, scouting, polygon accumulation, caustics, labels, and
temporal batching.

## Executable notebooks

Notebook JSON, imports, and code syntax are checked in the ordinary suite.
Execute every notebook headlessly with the optional notebook dependencies via:

```powershell
$env:MICROCAUSTICS_RUN_NOTEBOOKS = "1"
python -m unittest tests.test_release_workflows.NotebookExecutionTests -v
```

On a CUDA host, the final end-to-end, accuracy/performance, and streaming
notebooks must report CUDA in their executed outputs. The performance notebook
requests strict Triton execution whenever CUDA and Triton are available.

An opt-in repeated-run audit checks that live CUDA tensor memory stabilizes
after warmup:

```powershell
$env:MICROCAUSTICS_RUN_LONG_CUDA = "1"
python -m unittest tests.test_release_workflows.LongCudaStabilityTests -v
```

## Independent numerical oracles

Install the optional validation dependencies with:

```powershell
python -m pip install -e ".[test,validation]"
```

NumPy, SciPy, and Astropy checks run automatically when installed. GitHub
Actions installs lenstronomy and enables its comparisons in a dedicated oracle
job. Keeping the environment gate prevents lenstronomy's expensive first import
from slowing every operating-system and Python-version job. Run that same job
locally with:

```powershell
$env:MICROCAUSTICS_RUN_LENSTRONOMY = "1"
python -m pytest -q tests/test_external_oracles.py
```

These libraries are validation-only dependencies. None is imported by the
ordinary package import merely to support the comparisons.

## Weisenbach IPM comparison

The Weisenbach implementation and large paper products are not bundled. Point
the opt-in test at an aggregate created by the external paper adapter:

```powershell
$env:MICROCAUSTICS_WEISENBACH_AGGREGATE = "C:\path\to\luke_lightcurve_all_frames.npz"
python -m unittest tests.test_external_weisluke -v
```

The test verifies the fixed-aperture coordinate contract and hashes, recomputes
the light-curve residuals with the package's public validation API, and checks
the stored map metric. The default light-curve threshold is 20 mmag and can be
overridden with `MICROCAUSTICS_WEISENBACH_MAX_LC_RMSE_MMAG`.

An additional one-frame topology test constructs winding maps from Luke's
independent CCF caustics and from a complete microcaustics caustic field:

```powershell
$env:MICROCAUSTICS_WEISENBACH_CCF_FRAME = "C:\path\to\luke_far_field_single_frame.npz"
$env:MICROCAUSTICS_REFERENCE_CAUSTIC_FRAME = "C:\path\to\paper_method_map_caustic_comparison_data.npz"
python -m unittest tests.test_external_weisluke.WeisenbachWindingMapComparisonTests -v
```

Weisenbach CCF exports curves rather than a native signed winding product.
The test therefore compares winding modulo two, which is independent of curve
orientation. Raw signed winding integers require a shared determinant-side
orientation convention and are not asserted across the two codes.

Repository maintainers can additionally run the root-level legacy parity
harness against the pre-package monolith. It is intentionally outside this
installable package and is not part of the user-facing source distribution.

## Wheel installation smoke test

The ordinary metadata and CLI tests are part of the fast suite. Building a
wheel and installing it into a temporary environment is opt-in:

```powershell
$env:MICROCAUSTICS_RUN_BUILD_SMOKE = "1"
python -m unittest tests.test_packaging.WheelInstallationTests -v
```

This uses `pip wheel --no-build-isolation --no-deps`, installs the resulting
wheel away from the repository, imports the installed package, and runs
`microcaustics doctor`. It does not download dependencies.

## SIM5 comparison

SIM5 itself and its large comparison arrays are not bundled. Validate a frozen
paper summary explicitly with:

```powershell
$env:MICROCAUSTICS_SIM5_SUMMARY = "C:\path\to\paper_gr_sim5_summary.json"
python -m unittest tests.test_external_sim5 -v
```

The test checks matched-scope provenance, hit masks, image NMSE, integrated
flux, emission radius, and redshift-factor agreement over multiple systems.
