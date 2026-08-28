# Optional caustics interoperability

Most users do not need this adapter. Generate maps, light curves, caustics,
and labels through the ordinary microcaustics interfaces described elsewhere
in this guide. Those paths remain the package default.

This adapter has one narrow purpose. It lets developers insert an existing
microcaustics local lens mapping into software that specifically expects a
`caustics.ThinLens`. It does not replace the microcaustics ray tracer or make
the calculation more accurate or faster.

The numerical core does not require `caustics`. Install the optional adapter
and macro-image tools with:

```bash
python -m pip install "microcaustics[macro]"
```

The core package supports Python 3.10 and newer. This optional integration
requires Python 3.11 or newer because `caustics>=1.7` does not publish a
Python 3.10-compatible release.

`as_caustics_thin_lens` exposes one fixed microcaustics lens state as a real
`caustics.ThinLens`. This is useful when caustics should compose the local
microlens mapping with its simulators or other lens components, while
microcaustics continues to evaluate its accelerated local-exact/Taylor lens equation.

```python
import caustics
import microcaustics as mc
from microcaustics.solvers import TaylorFarFieldApproximation

far_field = TaylorFarFieldApproximation(simulation, lens_region, far_field_approx_config)
lens = mc.as_caustics_thin_lens(
    far_field,
    cosmology=caustics.FlatLambdaCDM(),
    origin_arcsec=(1.12, -0.37),
)

plane = caustics.SinglePlane(
    cosmology=caustics.FlatLambdaCDM(),
    lenses=[lens],
    z_l=0.5,
    z_s=2.0,
)
beta_x, beta_y = plane.raytrace(x=image_x_arcsec, y=image_y_arcsec)
```

The adapter converts caustics' arcseconds to the microarcseconds used by the
local microlensing calculation and converts the mapped coordinates back. The
`origin_arcsec` parameter places local `(0, 0)` at a resolved macroimage.
When the adapter is placed in `SinglePlane`, leave its own `z_l` and `z_s`
unset as above so the plane supplies them without overriding static values.

The adapted far-field state already contains the complete local lens equation:
smooth convergence, shear, exact nearby stars, and the Taylor far field. Do
not add another mass sheet or external shear for the same macroimage. Other
independent caustics lens components may still be composed normally.

This interface is opt-in. It does not select a different numerical backend,
change the package defaults, or move tensors off their device. A far-field state
built with eager PyTorch, `torch.compile`, or Triton retains that query path.
For a dynamic calculation, adapt the individual per-frame states. A batch
of different physical epochs is not one thin lens.
