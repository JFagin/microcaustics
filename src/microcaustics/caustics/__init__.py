"""Critical curves, caustics, winding numbers, and label products."""

from .anchor_gauge import (
    anchor_gauge_label_map,
    label_caustic_fields,
    production_anchor_gauge_points,
)
from .direct import direct_caustic_field, far_field_caustic_field
from .labels import (
    crossing_parity,
    distance_to_segments,
    orient_mapped_closed_segments,
    regular_grid_winding_number,
    winding_number,
)
from .marching import marching_squares_zero
from .production import (
    caustic_fields_from_far_fields,
    dynamic_labeled_caustics,
    dynamic_labeled_maps,
    multirate_labeled_light_curve,
    streaming_labeled_light_curve,
    streaming_labeled_light_curves,
)

__all__ = [
    "anchor_gauge_label_map",
    "caustic_fields_from_far_fields",
    "crossing_parity",
    "direct_caustic_field",
    "far_field_caustic_field",
    "distance_to_segments",
    "marching_squares_zero",
    "orient_mapped_closed_segments",
    "regular_grid_winding_number",
    "winding_number",
    "dynamic_labeled_caustics",
    "dynamic_labeled_maps",
    "multirate_labeled_light_curve",
    "streaming_labeled_light_curve",
    "streaming_labeled_light_curves",
    "label_caustic_fields",
    "production_anchor_gauge_points",
]
