# microcaustics

`microcaustics` generates static and dynamic gravitational-microlensing maps,
finite-source light curves, caustics and crossing labels, relativistic source
models, and resolved multi-image simulations.

The package provides three execution layers through one public interface.

- Portable eager Torch runs on CPU, CUDA, and Apple devices.
- Cached compiled Torch kernels are used when a compatible compiler is available.
- Fused Triton kernels provide the CUDA production path.

Start with [Choosing a method](choosing_methods.md), then use the
[source](sources.md), [dynamic](dynamic.md), and [caustic](caustics.md) guides
for the products needed by a particular study. The [API reference](api.md) is
generated directly from the tested source docstrings.

Runnable scripts and CUDA-aware notebooks are distributed in the `examples`
directory. They use only the public package interface.

The numerical methods and paper benchmarks are described in
[Fagin et al. (2026)](https://arxiv.org/abs/2610.19703). The package also provides
optional spectrum and response-based source calculations beyond those used
for the paper's figures and benchmarks.
