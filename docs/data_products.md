# Streaming and saving data products

Dynamic map iterators yield one `MagnificationMap` at a time. Light-curve and
transfer-function methods consume each map immediately and retain compact
response arrays. This is preferred when a complete map cube is not a product.

## Retaining selected epochs

High-level light-curve methods accept `keep_maps_at_days`. The returned
`LightCurve.maps` tuple contains only requested epochs that were evaluated.
Use `result.maps[0]` for the first retained map and `result.map_times_days` for
their chronologically ordered times. Unavailable epochs warn and are omitted.
This is the simplest choice for a small gallery or checkpoint set.

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

Light-curve archives preserve physical Jy fluxes and, when requested, the
source-center labels, crossing flags, distances, censoring flags, and separate
label epochs. After loading, use `result.labels.times_days` for that time axis.
It need not match the finer photometry cadence in `result.times_days`.

Full caustic geometry and retained magnification maps are not included in the
compact light-curve archive. Loaded labels have `caustics=None`. Save retained
maps separately with `save_magnification_map`. Older photometry-only archives
remain readable and return `labels=None`.

See `examples/notebooks/workflows/03_streaming_and_exporting_results.ipynb` for
selective retention, portable NPZ round trips, and optional FITS export.
