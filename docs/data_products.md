# Streaming and saving data products

Dynamic map iterators yield one `MagnificationMap` at a time. Light-curve and
transfer-function methods consume each map immediately and retain compact
response arrays. This is preferred when a complete map cube is not a product.

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

NPZ is convenient for compact examples. FITS integrates with astronomy tools.
Chunked HDF5 or Zarr is preferable for long sequences. The package does not
impose one storage dependency or silently serialize calculations.

See `examples/notebooks/17_streaming_and_exporting_results.ipynb` for a
CUDA-aware observer, NPZ reconstruction, and optional FITS export.
