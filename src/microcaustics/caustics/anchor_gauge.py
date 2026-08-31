"""Production anchor/gauge labels for finite caustic fields."""

from __future__ import annotations

import math
from time import perf_counter

import torch

from ..config import CausticConfig
from ..geometry import PlaneGrid, PlaneRegion
from ..results import (
    AnchorGaugeLabels,
    CausticField,
    LabeledCausticFrame,
    LabelMap,
    TimingBreakdown,
)
from ..runtime import warn_backend_fallback


def _boundary_probes(
    region: PlaneRegion,
    count: int,
    inset_fraction: float,
    phase: float,
    radial_jitter_fraction: float,
    radial_phase: float,
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Return deterministic, corner-avoiding probes around a rectangle."""

    count = int(count)
    index = torch.arange(count, device=device, dtype=torch.float64)
    perimeter_coordinate = torch.remainder((index + float(phase)) / count, 1.0) * 4.0
    edge = torch.floor(perimeter_coordinate).to(torch.int64)
    tangent = -1.0 + 2.0 * (perimeter_coordinate - torch.floor(perimeter_coordinate))
    golden = 0.6180339887498949
    tangent *= 0.84 + 0.08 * torch.sin(
        2.0 * torch.pi * ((index + 1.0) * golden + 0.11)
    )
    tangent = tangent.clamp(-0.98, 0.98)
    radial = (1.0 - float(inset_fraction)) * (
        1.0
        + float(radial_jitter_fraction)
        * torch.sin(2.0 * torch.pi * ((index + 1.0) * golden + radial_phase))
    )
    half_y, half_x = (0.5 * value for value in region.field_of_view_uas)
    boundary_x = radial * half_x
    boundary_y = radial * half_y
    tangent_x = tangent * half_x
    tangent_y = tangent * half_y
    x = torch.where(
        edge == 0,
        -boundary_x,
        torch.where(edge == 1, tangent_x, torch.where(edge == 2, boundary_x, -tangent_x)),
    )
    y = torch.where(
        edge == 0,
        tangent_y,
        torch.where(edge == 1, boundary_y, torch.where(edge == 2, -tangent_y, -boundary_y)),
    )
    center_y, center_x = region.center_uas
    points = torch.stack((x + center_x, y + center_y), dim=-1)
    return points.to(device=device, dtype=dtype)


def production_anchor_gauge_points(
    source_region: PlaneRegion,
    config: CausticConfig,
    *,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the validated offset nine-anchor/nine-gauge layout."""

    anchors = _boundary_probes(
        source_region,
        config.anchor_count,
        config.anchor_inset_fraction,
        config.anchor_phase,
        config.anchor_radial_jitter_fraction,
        0.17,
        device=device,
        dtype=dtype,
    )
    gauges = _boundary_probes(
        source_region,
        config.gauge_count,
        config.gauge_inset_fraction,
        config.gauge_phase,
        config.gauge_radial_jitter_fraction,
        0.63,
        device=device,
        dtype=dtype,
    )
    return anchors, gauges


def _crossing_counts_and_distances_portable(
    segments: torch.Tensor,
    valid_segments: torch.Tensor,
    anchors: torch.Tensor,
    crossing_points: torch.Tensor,
    distance_points: torch.Tensor,
    *,
    point_chunk_size: int,
    segment_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-path half-open crossings and distances without large tensors."""

    frames, segment_count = map(int, segments.shape[:2])
    counts = torch.zeros(
        (frames, crossing_points.shape[0], anchors.shape[0]),
        device=segments.device,
        dtype=torch.int32,
    )
    distance_squared = torch.full(
        (frames, distance_points.shape[0]),
        float("inf"),
        device=segments.device,
        dtype=segments.dtype,
    )
    epsilon = 1.0e-7
    distance_epsilon = 1.0e-24
    for segment_start in range(0, segment_count, int(segment_chunk_size)):
        segment_stop = min(segment_count, segment_start + int(segment_chunk_size))
        block = segments[:, segment_start:segment_stop]
        valid = valid_segments[:, segment_start:segment_stop]
        # Bound the temporary [frame, point, anchor, segment] predicate even
        # when a diagnostic map requests a very large nominal point chunk.
        pairs_per_point = max(1, int(anchors.shape[0]) * (segment_stop - segment_start))
        effective_point_chunk = min(
            int(point_chunk_size),
            max(1, 4_000_000 // pairs_per_point),
        )
        point_a = block[:, :, 0]
        point_b = block[:, :, 1]
        segment_delta = point_b - point_a
        for point_start in range(0, crossing_points.shape[0], effective_point_chunk):
            point_stop = min(crossing_points.shape[0], point_start + effective_point_chunk)
            query = crossing_points[point_start:point_stop]
            path = query[None, :, None, :] - anchors[None, None, :, :]
            anchor_to_a = point_a[:, None, None] - anchors[None, None, :, None]
            anchor_to_b = point_b[:, None, None] - anchors[None, None, :, None]
            path_x = path[..., 0, None]
            path_y = path[..., 1, None]
            segment_x = segment_delta[:, None, None, :, 0]
            segment_y = segment_delta[:, None, None, :, 1]
            denominator = path_x * segment_y - path_y * segment_x
            non_parallel = denominator.abs() > epsilon
            safe = torch.where(non_parallel, denominator, torch.ones_like(denominator))
            fraction = (
                anchor_to_a[..., 0] * segment_y
                - anchor_to_a[..., 1] * segment_x
            ) / safe
            side_a = path_x * anchor_to_a[..., 1] - path_y * anchor_to_a[..., 0]
            side_b = path_x * anchor_to_b[..., 1] - path_y * anchor_to_b[..., 0]
            intersects = (
                non_parallel
                & (fraction > epsilon)
                & (fraction < 1.0 - epsilon)
                & ((side_a > 0.0) != (side_b > 0.0))
                & valid[:, None, None, :]
            )
            counts[:, point_start:point_stop] += intersects.sum(dim=-1).to(torch.int32)
        distance_chunk = min(
            int(point_chunk_size),
            max(1, 8_000_000 // max(1, segment_stop - segment_start)),
        )
        for point_start in range(0, distance_points.shape[0], distance_chunk):
            point_stop = min(distance_points.shape[0], point_start + distance_chunk)
            query = distance_points[point_start:point_stop]
            relative = query[None, :, None] - point_a[:, None]
            length_squared = segment_delta.square().sum(dim=-1).clamp_min(
                distance_epsilon
            )
            projection = (
                relative * segment_delta[:, None]
            ).sum(dim=-1) / length_squared[:, None]
            closest = (
                point_a[:, None]
                + projection.clamp(0.0, 1.0)[..., None] * segment_delta[:, None]
            )
            residual = query[None, :, None] - closest
            candidate = residual.square().sum(dim=-1)
            candidate = torch.where(
                valid[:, None],
                candidate,
                torch.full_like(candidate, float("inf")),
            )
            distance_squared[:, point_start:point_stop] = torch.minimum(
                distance_squared[:, point_start:point_stop],
                candidate.amin(dim=-1),
            )
    return counts, torch.sqrt(distance_squared)


def _reference_offsets(
    pair_counts: torch.Tensor,
    invalid_pair_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the small redundant anchor gauge on CPU without repeated syncs."""

    parity = (pair_counts.detach().cpu().to(torch.int64) & 1).clone()
    count = int(parity.shape[0])
    parity.fill_diagonal_(0)
    pair_mask = ~torch.eye(count, dtype=torch.bool)
    if invalid_pair_counts is not None:
        pair_mask &= invalid_pair_counts.detach().cpu().reshape(count, count) == 0
    best = None
    best_score = -1
    for reference in range(count):
        gauge = parity[:, reference].clone()
        gauge[reference] = 0
        for _ in range(3):
            previous = gauge.clone()
            for anchor in range(count):
                votes = (parity[anchor] ^ gauge)[pair_mask[anchor]]
                if votes.numel() == 0:
                    continue
                ones = int(votes.sum())
                zeros = int(votes.numel()) - ones
                if ones != zeros:
                    gauge[anchor] = int(ones > zeros)
            if torch.equal(previous, gauge):
                break
        predicted = gauge[:, None] ^ gauge[None]
        score = int((predicted[pair_mask] == parity[pair_mask]).sum())
        if score > best_score:
            best = gauge.clone()
            best_score = score
    assert best is not None
    return ((best ^ best[0]) & 1).to(torch.int8)


def _reference_offsets_batched(
    pair_counts: torch.Tensor,
    invalid_pair_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the same redundant anchor gauge for a temporal CPU batch.

    This is the frame-vectorized form of :func:`_reference_offsets`.  The
    reference, fixed-point, anchor-update, and strict tie-breaking order are
    intentionally identical, so batching changes only Python dispatch cost.
    """

    parity = (pair_counts.detach().cpu().to(torch.int64) & 1).clone()
    if parity.ndim != 3 or parity.shape[-1] != parity.shape[-2]:
        raise ValueError("batched anchor pair counts must have shape [B,A,A]")
    frames, count, _ = parity.shape
    diagonal = torch.arange(count)
    parity[:, diagonal, diagonal] = 0
    pair_mask = (~torch.eye(count, dtype=torch.bool))[None].expand(
        frames, -1, -1
    ).clone()
    if invalid_pair_counts is not None:
        pair_mask &= (
            invalid_pair_counts.detach().cpu().reshape(frames, count, count) == 0
        )
    best = torch.zeros((frames, count), dtype=torch.int64)
    best_score = torch.full((frames,), -1, dtype=torch.int64)
    for reference in range(count):
        gauge = parity[:, :, reference].clone()
        gauge[:, reference] = 0
        for _ in range(3):
            previous = gauge.clone()
            for anchor in range(count):
                mask = pair_mask[:, anchor]
                votes = parity[:, anchor] ^ gauge
                ones = (votes * mask.to(votes.dtype)).sum(dim=-1)
                valid = mask.sum(dim=-1)
                zeros = valid - ones
                update = ones != zeros
                gauge[:, anchor] = torch.where(
                    update,
                    (ones > zeros).to(gauge.dtype),
                    gauge[:, anchor],
                )
            if torch.equal(previous, gauge):
                break
        predicted = gauge[:, :, None] ^ gauge[:, None, :]
        score = ((predicted == parity) & pair_mask).sum(dim=(-2, -1))
        improved = score > best_score
        best[improved] = gauge[improved]
        best_score[improved] = score[improved]
    return ((best ^ best[:, :1]) & 1).to(torch.int8)


def _majority(
    counts: torch.Tensor,
    offsets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return binary majority classes and winning vote counts."""

    offsets = offsets.detach().cpu().to(torch.int64)
    while offsets.ndim < counts.ndim:
        offsets = offsets.unsqueeze(-2)
    parity = (counts.detach().cpu().to(torch.int64) & 1) ^ offsets
    if valid_mask is None:
        mask = torch.ones_like(parity, dtype=torch.bool)
    else:
        mask = valid_mask.detach().cpu().to(torch.bool)
    ones = (parity * mask.to(parity.dtype)).sum(dim=-1)
    valid = mask.sum(dim=-1)
    zeros = valid - ones
    return (
        (ones > zeros).to(torch.int8),
        torch.maximum(ones, zeros).to(torch.int32),
        valid.to(torch.int32),
    )


def label_caustic_fields(
    fields: tuple[CausticField, ...],
    source_region: PlaneRegion,
    config: CausticConfig,
    *,
    previous_aligned_gauges: torch.Tensor | None = None,
    previous_gauge_distances_uas: torch.Tensor | None = None,
    previous_center_label: int | None = None,
    previous_center_distance_uas: float | None = None,
    diagnostic_grid: PlaneGrid | None = None,
    include_distance_map: bool = False,
) -> tuple[
    tuple[LabeledCausticFrame, ...],
    torch.Tensor,
    torch.Tensor,
    int,
    float,
]:
    """Label one temporal batch and align it to the preceding batch."""

    center_distance_cap_uas = 0.5 * min(source_region.field_of_view_uas)
    if not fields:
        empty = torch.empty(0, dtype=torch.int8)
        return (
            (),
            empty,
            empty.to(torch.float32),
            0 if previous_center_label is None else previous_center_label,
            center_distance_cap_uas
            if previous_center_distance_uas is None
            else float(previous_center_distance_uas),
        )
    device = fields[0].caustic_segments_uas.device
    dtype = fields[0].caustic_segments_uas.dtype
    anchors, gauges = production_anchor_gauge_points(
        source_region,
        config,
        device=device,
        dtype=dtype,
    )
    center_y, center_x = source_region.center_uas
    center = torch.tensor([[center_x, center_y]], device=device, dtype=dtype)
    query_points = torch.cat((center, gauges), dim=0)
    crossing_points = torch.cat((anchors, query_points), dim=0)
    maximum_segments = max(field.segment_count for field in fields)
    segments = torch.zeros(
        (len(fields), maximum_segments, 2, 2),
        device=device,
        dtype=dtype,
    )
    valid = torch.zeros(
        (len(fields), maximum_segments),
        device=device,
        dtype=torch.bool,
    )
    invalid_segments = torch.zeros_like(valid)
    for frame, field in enumerate(fields):
        count = field.segment_count
        if count:
            segments[frame, :count] = field.caustic_segments_uas
            valid[frame, :count] = True
            invalid = field.invalid_segment_mask
            if invalid is not None:
                invalid_segments[frame, :count] = invalid.to(
                    device=device,
                    dtype=torch.bool,
                )
    started = perf_counter()
    use_triton = False
    if device.type == "cuda" and dtype == torch.float32:
        try:
            from .triton_caustics import (
                batched_caustic_crossings_distances_triton,
                triton_caustics_available,
            )

            if triton_caustics_available():
                counts, distances = batched_caustic_crossings_distances_triton(
                    segments,
                    valid,
                    anchors,
                    crossing_points,
                    query_points,
                    block_segments=config.triton_segment_block,
                )
                use_triton = True
        except Exception as error:
            warn_backend_fallback("Triton anchor/gauge labeling", error)
            use_triton = False
    if not use_triton:
        counts, distances = _crossing_counts_and_distances_portable(
            segments,
            valid,
            anchors,
            crossing_points,
            query_points,
            point_chunk_size=config.point_chunk_size,
            segment_chunk_size=config.segment_chunk_size,
        )
    invalid_counts = None
    if bool(invalid_segments.any().detach().cpu()):
        if use_triton:
            invalid_counts, _ = batched_caustic_crossings_distances_triton(
                segments,
                invalid_segments,
                anchors,
                crossing_points,
                query_points,
                block_segments=config.triton_segment_block,
            )
        else:
            invalid_counts, _ = _crossing_counts_and_distances_portable(
                segments,
                invalid_segments,
                anchors,
                crossing_points,
                query_points[:0],
                point_chunk_size=config.point_chunk_size,
                segment_chunk_size=config.segment_chunk_size,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    label_seconds = perf_counter() - started
    # Transfer the small batched query result once.  Per-frame ``.cpu()``
    # calls would serialize the CUDA stream 147 times even though production
    # requests only the center and gauge diagnostics.
    counts_cpu = counts.detach().cpu()
    distances_cpu = distances.detach().cpu()
    invalid_counts_cpu = (
        None if invalid_counts is None else invalid_counts.detach().cpu()
    )
    anchors_cpu = anchors.detach().cpu()
    gauges_cpu = gauges.detach().cpu()
    pair_invalid = (
        None
        if invalid_counts_cpu is None
        else invalid_counts_cpu[:, : config.anchor_count]
    )
    offsets_cpu = _reference_offsets_batched(
        counts_cpu[:, : config.anchor_count],
        pair_invalid,
    )
    query_invalid = (
        None
        if invalid_counts_cpu is None
        else invalid_counts_cpu[:, config.anchor_count :] != 0
    )
    classes_cpu, votes_cpu, valid_counts_cpu = _majority(
        counts_cpu[:, config.anchor_count :],
        offsets_cpu,
        None if query_invalid is None else ~query_invalid,
    )
    outputs = []
    aligned_previous = previous_aligned_gauges
    distance_previous = previous_gauge_distances_uas
    center_previous = previous_center_label
    center_distance_previous = previous_center_distance_uas
    for frame, field in enumerate(fields):
        offsets = offsets_cpu[frame]
        classes = classes_cpu[frame]
        votes = votes_cpu[frame]
        valid_counts = valid_counts_cpu[frame]
        raw_center = int(classes[0])
        raw_gauges = classes[1:]
        frame_xor = 0
        if aligned_previous is not None:
            previous = aligned_previous.to(torch.int8)
            previous_distance = distance_previous
            current_distance = distances_cpu[frame, 1:]
            safe = torch.isfinite(current_distance)
            if previous_distance is not None:
                safe &= torch.isfinite(previous_distance)
                safe &= previous_distance > config.safe_gauge_distance_uas
            safe &= current_distance > config.safe_gauge_distance_uas
            if int(safe.sum()) < config.minimum_alignment_gauges:
                safe = torch.isfinite(current_distance)
            differences = raw_gauges[safe] ^ previous[safe]
            if differences.numel():
                if config.weighted_temporal_alignment:
                    weights = current_distance[safe].clamp_min(1.0e-6)
                    zero_weight = weights[differences == 0].sum()
                    one_weight = weights[differences == 1].sum()
                    frame_xor = int(one_weight > zero_weight)
                else:
                    frame_xor = int((differences == 1).sum() > (differences == 0).sum())
        aligned_gauges = raw_gauges ^ frame_xor
        center_label = raw_center ^ frame_xor
        crossing = center_previous is not None and center_label != center_previous
        if crossing and config.crossing_distance_uas is not None:
            endpoint_distance = float(distances_cpu[frame, 0])
            if center_distance_previous is not None:
                endpoint_distance = min(endpoint_distance, center_distance_previous)
            crossing = endpoint_distance <= config.crossing_distance_uas
        raw_center_distance_uas = float(distances_cpu[frame, 0])
        center_distance_censored = (
            not math.isfinite(raw_center_distance_uas)
            or raw_center_distance_uas > center_distance_cap_uas
        )
        center_distance_uas = min(
            raw_center_distance_uas,
            center_distance_cap_uas,
        )
        if not math.isfinite(center_distance_uas):
            center_distance_uas = center_distance_cap_uas
        labels = AnchorGaugeLabels(
            raw_center_label=raw_center,
            center_label=center_label,
            center_crossing=bool(crossing),
            center_distance_uas=center_distance_uas,
            center_vote_count=int(votes[0]),
            center_valid_count=int(valid_counts[0]),
            gauge_labels=aligned_gauges,
            gauge_distances_uas=distances_cpu[frame, 1:],
            gauge_vote_counts=votes[1:],
            gauge_valid_counts=valid_counts[1:],
            anchor_offsets=offsets,
            anchor_points_uas=anchors_cpu,
            gauge_points_uas=gauges_cpu,
            center_distance_censored=center_distance_censored,
            frame_xor=frame_xor,
            metadata={
                "method": "anchor_gauge_binary_majority",
                "source_center_uas": (float(center_x), float(center_y)),
                "center_distance_cap_uas": center_distance_cap_uas,
                "half_open_vertex_rule": True,
                "triton_fused_crossing_distance": use_triton,
                "caustic_segments": field.segment_count,
            },
            timing=TimingBreakdown(
                collected=field.timing.collected,
                steady_seconds=label_seconds / len(fields),
            ),
        )
        label_map = None
        distance_map = None
        if diagnostic_grid is not None:
            label_map = anchor_gauge_label_map(
                field,
                diagnostic_grid,
                anchors,
                offsets,
                frame_xor=frame_xor,
                config=config,
            )
            if include_distance_map:
                distance_map = field.distance_map(
                    diagnostic_grid,
                    point_chunk_size=config.point_chunk_size,
                    segment_chunk_size=config.segment_chunk_size,
                )
        outputs.append(LabeledCausticFrame(field, labels, label_map, distance_map))
        aligned_previous = aligned_gauges
        distance_previous = distances_cpu[frame, 1:]
        center_previous = center_label
        center_distance_previous = center_distance_uas
    assert (
        aligned_previous is not None
        and distance_previous is not None
        and center_previous is not None
        and center_distance_previous is not None
    )
    return (
        tuple(outputs),
        aligned_previous,
        distance_previous,
        int(center_previous),
        center_distance_previous,
    )


def anchor_gauge_label_map(
    field: CausticField,
    grid: PlaneGrid,
    anchors: torch.Tensor,
    offsets: torch.Tensor,
    *,
    frame_xor: int,
    config: CausticConfig,
) -> LabelMap:
    """Materialize the production majority label on a diagnostic grid."""

    x, y = grid.mesh(
        device=field.caustic_segments_uas.device,
        dtype=field.caustic_segments_uas.dtype,
    )
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    segments = field.caustic_segments_uas[None]
    valid = torch.ones(
        (1, field.segment_count),
        device=segments.device,
        dtype=torch.bool,
    )
    counts, _ = _crossing_counts_and_distances_portable(
        segments,
        valid,
        anchors.to(device=segments.device, dtype=segments.dtype),
        points,
        points[:0],
        point_chunk_size=config.point_chunk_size,
        segment_chunk_size=config.segment_chunk_size,
    )
    query_valid = None
    if field.invalid_segment_mask is not None and bool(field.invalid_segment_mask.any()):
        invalid = field.invalid_segment_mask.to(device=segments.device)[None]
        invalid_counts, _ = _crossing_counts_and_distances_portable(
            segments,
            invalid,
            anchors.to(device=segments.device, dtype=segments.dtype),
            points,
            points[:0],
            point_chunk_size=config.point_chunk_size,
            segment_chunk_size=config.segment_chunk_size,
        )
        query_valid = invalid_counts[0] == 0
    classes, _, _ = _majority(counts[0], offsets, query_valid)
    values = (classes ^ int(frame_xor)).reshape(grid.shape)
    return LabelMap(
        values,
        grid,
        metadata={
            "method": "anchor_gauge_binary_majority",
            "anchor_count": int(anchors.shape[0]),
            "frame_xor": int(frame_xor),
        },
    )
