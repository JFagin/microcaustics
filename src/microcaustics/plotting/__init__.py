"""Optional, composable Matplotlib plots for public simulation products.

Install the plotting dependency with ``pip install microcaustics[plot]``.
Plot functions return ``(figure, axes)`` and never call ``show()`` or rerun a
microlensing calculation.
"""

from ._common import (
    add_scale_bar,
    band_colors,
    finish_axis,
    hide_image_axes,
    panel_colorbar,
    publication_style,
)
from .animations import animate_standardized_source_bands, save_fixed_palette_gif
from .caustics import plot_anchor_gauge, plot_caustics
from .datasets import plot_labeled_map_gallery, plot_light_curve_dataset
from .light_curves import (
    plot_light_curve,
    plot_multi_image_light_curves,
    plot_photometric_observations,
)
from .maps import (
    plot_caustic_diagnostics,
    plot_distance_map,
    plot_label_map,
    plot_magnification_map,
    plot_map_comparison,
)
from .methods import (
    plot_far_field_method,
    plot_ipm_scout_method,
    plot_lens_field_strategies,
)
from .paper_methods import (
    render_live_paper_anchor_gauge,
    render_live_paper_ipm_schematic,
    render_paper_anchor_gauge,
    render_paper_ipm_schematic,
    render_paper_sim5_validation,
)
from .sources import (
    MeanSourceIsophotes,
    TemporalSourceStandardization,
    enclosed_flux_contour_levels,
    mean_source_isophotes,
    plot_rendered_macro_image,
    plot_source_brightness,
    plot_standardized_source_bands,
    standardize_source_over_time,
)
from .timing import plot_timing_breakdown, print_benchmark, runtime_description
from .transfer import (
    plot_mean_delays,
    plot_transfer_function,
    transfer_response_density,
)

__all__ = [
    "plot_anchor_gauge",
    "plot_caustics",
    "plot_caustic_diagnostics",
    "plot_distance_map",
    "plot_far_field_method",
    "plot_label_map",
    "plot_labeled_map_gallery",
    "plot_light_curve",
    "plot_light_curve_dataset",
    "plot_magnification_map",
    "plot_map_comparison",
    "plot_ipm_scout_method",
    "plot_lens_field_strategies",
    "plot_mean_delays",
    "plot_multi_image_light_curves",
    "plot_photometric_observations",
    "plot_rendered_macro_image",
    "plot_source_brightness",
    "plot_standardized_source_bands",
    "plot_timing_breakdown",
    "plot_transfer_function",
    "plot_far_field_method",
    "enclosed_flux_contour_levels",
    "mean_source_isophotes",
    "MeanSourceIsophotes",
    "standardize_source_over_time",
    "TemporalSourceStandardization",
    "transfer_response_density",
    "print_benchmark",
    "runtime_description",
    "render_paper_ipm_schematic",
    "render_paper_anchor_gauge",
    "render_live_paper_ipm_schematic",
    "render_live_paper_anchor_gauge",
    "render_paper_sim5_validation",
    "add_scale_bar",
    "band_colors",
    "finish_axis",
    "publication_style",
    "hide_image_axes",
    "panel_colorbar",
    "save_fixed_palette_gif",
    "animate_standardized_source_bands",
]
