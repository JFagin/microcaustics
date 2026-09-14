# Package design

## Public vocabulary

The primary user object is `MicrolensingSimulation`. Public APIs use physical
names rather than generic orchestration terms. Results use names such as
`MagnificationMap`, `LightCurve`, and `CausticField`.

## Separation of concerns

1. Lens models describe macro parameters and point-mass populations.
2. Plane grids describe requested map coordinates independently of a source.
3. Solvers map lens-plane samples to source-plane products.
4. Sources provide brightness tensors without depending on a lens solver.
5. Caustic calculations consume a lens mapping and remain optional.
6. GR and reverberation modules remain usable without microlensing.
7. Multi-image simulations compose independent macroimages around one shared
   source. They do not duplicate map, source, or caustic implementations.

Solver configuration describes numerical intent rather than a particular
kernel. For example, IPM exposes the same `N`, `k`, `r`, and `v` controls on
CUDA/Triton, compiled or eager PyTorch, CPU, and Apple devices. Accelerated
specializations are implementation details and always retain a portable
correctness path.

## High-level system facade

`microcaustics.system` remains the stable home of `MicrolensingSystem`,
`MicrolensingRealization`, and `IntegrationDomain`. Cohesive implementation
details live in the private `microcaustics._system` package:


- `scheduling` resolves cadences, numerical controls, dynamic defaults,
  retained-map observers, and light-curve dispatch;
- `geometry` derives source and lens-plane regions; and
- `coordinates` handles coordinate-basis transformations and point-lens
  motion diagnostics.

Application code should continue importing the public classes from
`microcaustics`. The private modules exist to keep the facade readable and
to let batching and multi-image orchestration reuse policy without importing
it indirectly through the public system class module.

## Migration rule

Production code is extracted one component at a time. An extracted component
must match deterministic fixtures from the final paper pipeline before the
reference runner is changed to import it. After that change, the package version
becomes the sole implementation so the paper and public APIs cannot drift.

The installed package does not contain abandoned experimental algorithms,
generated benchmark output, or external comparison binaries. Maintained,
publication-oriented figure builders that operate on public result objects are
part of the optional `microcaustics.plotting` interface so notebooks and papers
share one tested plotting implementation.

## Migrated production components

The dynamic IPM path is now implemented behind the public solver contracts.
Its CUDA specialization fuses temporal Taylor far-field queries and direct-cell
rasterization while retaining the exact portable Sutherland--Hodgman reference
for validation and non-CUDA systems. Source scouting, tail padding, and memory
backoff are scheduler policies with explicit result metadata. None are hidden
inside a numerical kernel. Far-field coefficients are evaluated independently
at every physical map epoch.
