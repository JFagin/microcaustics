# Streaming and saving data products

Dynamic map iterators yield one `MagnificationMap` at a time. Light-curve and
transfer-function methods consume each map immediately and retain compact
response arrays. This is preferred when a complete map cube is not a product.

## Retaining selected epochs

High-level light-curve methods accept `keep_maps_at_days`. The returned
`LightCurve.maps` mapping then contains only those requested epochs. This is
the simplest choice for a small gallery or checkpoint set.

## Map observers

Light-curve, labeled-light-curve, transfer-function, and multi-image workflows
accept map-observer callbacks. A callback receives the real frame index and
registered `MagnificationMap`. Padded temporal slots are never exposed. Use it
to save selected epochs, update online statistics, or pass maps elsewhere.

Keep observers fast. Copy retained arrays to CPU promptly and avoid blocking
network writes in the accelerator loop when throughput matters.

## What to save

For each retained map, preserve values and dtype, `PlaneGrid.shape`,
`field_of_view_uas`, `center_uas`, observer epoch, numerical configuration, actual
backend, approximation metadata, units, and normalization. For light curves,
preserve times, band names, lensed and unlensed fluxes, identifiers, and
provenance.

`save_magnification_map`, `load_magnification_map`, `save_light_curve`, and
`load_light_curve` provide compact compressed-NPZ round trips for the two most
common products. FITS integrates with astronomy tools. Chunked HDF5 or Zarr is
preferable for long sequences. Serialization is always explicit.

See `examples/notebooks/workflows/03_streaming_and_exporting_results.ipynb` for
selective retention, portable NPZ round trips, and optional FITS export.
