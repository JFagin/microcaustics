# Portability and troubleshooting

## Runtime matrix

| Platform | Device | Preferred backend | Notes |
|---|---|---|---|
| Linux | NVIDIA CUDA | Triton float32 | Upstream-supported fused IPM and Taylor far-field paths |
| Windows | NVIDIA CUDA | Triton float32 | Fast fused paths through the community `triton-windows` distribution |
| Windows or Linux | CPU | eager or compiled Torch | Portable reference and small jobs |
| macOS | Apple MPS | compiled/eager Torch | Float32. Support follows PyTorch |
| macOS | CPU | eager or compiled Torch | Portable fallback, including float64 |

Use `RuntimeConfig(device="auto", backend="auto")` for portable scripts. For
benchmark claims, request the intended backend with `strict_backend=True` so a
fallback cannot be mistaken for an accelerated timing.

`Backend.TORCH_COMPILE` compiles the exact point-mass reductions, Taylor far-field
queries, and IRS deposition. The portable exact IPM polygon clipper remains a
Python reference implementation, so IPM results requested with this backend
report `torch-compile-partial`. Their metadata lists ray tracing,
interpolation, and rasterization separately. CUDA float32 uses the fully fused
Triton IPM rasterizer. On Windows CPU, compiled Torch requires a discoverable
C++ compiler. Automatic selection uses eager Torch when that toolchain is not
available.

Run `microcaustics doctor` to report the platform, Torch, CUDA, Triton,
device-memory, and selected-runtime information.

## Installing optional features

Install the appropriate PyTorch build first using the live
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
Choosing a CUDA wheel is required for NVIDIA acceleration. Installing a CPU
wheel on a CUDA workstation does not make CUDA available to this package.

Upstream Triton officially supports Linux. On native Windows, use the
community-maintained
[`triton-windows`](https://github.com/triton-lang/triton-windows) wheels and
match their Triton minor version to the installed PyTorch minor version. Do
not install both upstream `triton` and `triton-windows` in one environment.
WSL2 users can instead follow the upstream Linux path. macOS and CPU runtimes
use the portable PyTorch implementations and do not require Triton.

Optional extras are separated so a map-only installation does not require
plotting or astronomy packages:

- `microcaustics[plot]` provides plotting helpers.
- `microcaustics[science]` provides Astropy and SciPy.
- `microcaustics[macro]` provides caustics-based macro models and rendering.
- `microcaustics[notebooks]` provides Jupyter and plotting.
- `microcaustics[validation]` provides independent scientific oracles.

After installation, run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import triton; print(triton.__version__)"
microcaustics doctor
```

The Triton import is expected to fail on a deliberately portable CPU/macOS
environment. On an accelerated environment, `doctor` should report
Expected output includes `cuda_available: true`, `triton_importable: true`, and automatic backend
`triton`. Use `strict_backend=True` for published timings or installation
smoke tests so an accelerated-kernel failure cannot be hidden by fallback.

## First calls are slower

The first accelerated call can include kernel compilation, allocator
initialization, and cache population. Report it as first-call time, not pure
compile time. Measure warmed steady-state calls separately with
`benchmark_callable`. Put `TRITON_CACHE_DIR` on a fast writable local disk.

## CUDA out of memory

`RuntimeConfig.memory_fraction` reserves headroom. The scheduler can reduce
IPM cell chunks and split temporal batches losslessly after an allocation
failure. Automatic tuning rejects configurations above its memory ceiling.
These mechanisms never change `N`, `k`, `r`, `v`, or physical geometry.

If memory use remains high, reduce the temporal batch, reduce the solver's
spatial chunk, stream maps into photometry instead of retaining a cube, and
save only selected diagnostics with a map observer.

## Unexpectedly slow execution

Check the actual backend in result metadata and the output of `doctor`. Ensure
strict Triton execution succeeds on CUDA float32, compiler caches are local,
timings exclude first-call work, and the calculation did not select a direct
reference path. Full-field methods, float64, labels, and diagnostic maps
intentionally cost more than a tiled float32 light-curve-only workflow.
