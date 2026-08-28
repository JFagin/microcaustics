# microcaustics

`microcaustics` generates static and dynamic gravitational-microlensing maps,
finite-source light curves, caustics and crossing labels, relativistic source
models, and resolved multi-image simulations.

The package provides three execution layers from one public interface:

- portable eager Torch for CPU, CUDA, and Apple devices.
- cached compiled Torch kernels where a compatible compiler is available. And
- fused Triton kernels for the CUDA production path.

Start with [Choosing a method](choosing_methods.md), then use the
[source](sources.md), [dynamic](dynamic.md), and [caustic](caustics.md) guides
for the products needed by a particular study. The [API reference](api.md) is
generated directly from the tested source docstrings.

Runnable scripts and CUDA-aware notebooks are distributed in the `examples`
directory. They use only the public package interface.
