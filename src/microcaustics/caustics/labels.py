"""Half-open segment-crossing predicates for caustic labels."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def orient_mapped_closed_segments(
    critical_segments_uas,
    caustic_segments_uas,
    *,
    closure_tolerance_uas: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect unordered mapped critical segments into positive closed loops.

    Connectivity is recovered in the lens plane, where each critical curve is
    closed. The corresponding source-plane segments are kept paired and each
    completed loop is given a positive signed-area orientation. This is useful
    for importing external critical-curve products whose segment ordering was
    not retained. Native package marching output is already oriented and does
    not require this repair.
    """

    critical = torch.as_tensor(critical_segments_uas)
    caustic = torch.as_tensor(
        caustic_segments_uas,
        device=critical.device,
        dtype=critical.dtype,
    )
    if critical.ndim != 3 or tuple(critical.shape[1:]) != (2, 2):
        raise ValueError("critical_segments_uas must have shape [segment, 2, 2]")
    if caustic.shape != critical.shape:
        raise ValueError("critical and caustic segment shapes must match")
    tolerance = float(closure_tolerance_uas)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("closure_tolerance_uas must be finite and positive")
    if critical.shape[0] == 0:
        return critical.clone(), caustic.clone()

    coordinates = critical.detach().to(device="cpu", dtype=torch.float64)
    bins: dict[tuple[int, int], list[int]] = {}
    representatives: list[tuple[float, float]] = []
    endpoint_nodes: list[tuple[int, int]] = []

    def node_for(x: float, y: float) -> int:
        key_x = math.floor(x / tolerance)
        key_y = math.floor(y / tolerance)
        best = None
        best_distance2 = tolerance * tolerance
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for candidate in bins.get((key_x + offset_x, key_y + offset_y), ()):
                    cx, cy = representatives[candidate]
                    distance2 = (x - cx) ** 2 + (y - cy) ** 2
                    if distance2 <= best_distance2:
                        best = candidate
                        best_distance2 = distance2
        if best is not None:
            return best
        index = len(representatives)
        representatives.append((x, y))
        bins.setdefault((key_x, key_y), []).append(index)
        return index

    for segment in coordinates:
        endpoint_nodes.append(
            tuple(node_for(float(point[0]), float(point[1])) for point in segment)
        )

    incident: dict[int, list[tuple[int, int]]] = {}
    for edge_index, nodes in enumerate(endpoint_nodes):
        incident.setdefault(nodes[0], []).append((edge_index, 0))
        incident.setdefault(nodes[1], []).append((edge_index, 1))
    invalid_degrees = [node for node, edges in incident.items() if len(edges) != 2]
    if invalid_degrees:
        raise ValueError(
            "segments do not form closed degree-two critical curves within "
            "closure_tolerance_uas"
        )

    visited = [False] * len(endpoint_nodes)
    oriented_critical: list[torch.Tensor] = []
    oriented_caustic: list[torch.Tensor] = []
    for seed in range(len(endpoint_nodes)):
        if visited[seed]:
            continue
        start_node = endpoint_nodes[seed][0]
        edge_index, enter_end = seed, 0
        critical_loop: list[torch.Tensor] = []
        caustic_loop: list[torch.Tensor] = []
        while not visited[edge_index]:
            visited[edge_index] = True
            if enter_end == 0:
                critical_loop.append(critical[edge_index])
                caustic_loop.append(caustic[edge_index])
            else:
                critical_loop.append(critical[edge_index].flip(0))
                caustic_loop.append(caustic[edge_index].flip(0))
            exit_node = endpoint_nodes[edge_index][1 - enter_end]
            candidates = [item for item in incident[exit_node] if not visited[item[0]]]
            if not candidates:
                if exit_node != start_node:
                    raise ValueError("critical-curve segment chain is open")
                break
            edge_index, enter_end = candidates[0]

        source_segments = torch.stack(caustic_loop)
        vertices = source_segments[:, 0]
        signed_area = 0.5 * torch.sum(
            vertices[:, 0] * torch.roll(vertices[:, 1], -1)
            - torch.roll(vertices[:, 0], -1) * vertices[:, 1]
        )
        lens_segments = torch.stack(critical_loop)
        if bool(signed_area < 0):
            lens_segments = lens_segments.flip((0, 1))
            source_segments = source_segments.flip((0, 1))
        oriented_critical.append(lens_segments)
        oriented_caustic.append(source_segments)

    return torch.cat(oriented_critical), torch.cat(oriented_caustic)


@torch.no_grad()
def winding_number(
    segments_uas,
    points_uas,
    *,
    point_chunk_size: int = 4096,
    segment_chunk_size: int = 16384,
) -> torch.Tensor:
    """Return signed winding numbers for consistently oriented closed segments.

    A half-open positive-x ray rule counts an upward oriented crossing as +1
    and a downward crossing as -1. Segment orientation is therefore material.
    Callers providing unrelated or unoriented open segments should use
    :func:`crossing_parity` instead.
    """

    segments = torch.as_tensor(segments_uas)
    if segments.ndim != 3 or tuple(segments.shape[1:]) != (2, 2):
        raise ValueError("segments_uas must have shape [segment, 2, 2]")
    points = torch.as_tensor(
        points_uas,
        device=segments.device,
        dtype=segments.dtype,
    )
    if points.shape[-1:] != (2,):
        raise ValueError("points_uas must have trailing Cartesian dimension 2")
    if int(point_chunk_size) < 1 or int(segment_chunk_size) < 1:
        raise ValueError("point and segment chunk sizes must be positive")
    output_shape = points.shape[:-1]
    flat_points = points.reshape(-1, 2)
    winding = torch.zeros(
        flat_points.shape[0],
        device=segments.device,
        dtype=torch.int64,
    )
    for point_start in range(0, flat_points.shape[0], int(point_chunk_size)):
        point_stop = min(flat_points.shape[0], point_start + int(point_chunk_size))
        query = flat_points[point_start:point_stop]
        counts = torch.zeros(query.shape[0], device=segments.device, dtype=torch.int64)
        point_x = query[:, 0, None]
        point_y = query[:, 1, None]
        for segment_start in range(0, segments.shape[0], int(segment_chunk_size)):
            segment_stop = min(
                segments.shape[0],
                segment_start + int(segment_chunk_size),
            )
            block = segments[segment_start:segment_stop]
            x0, y0 = block[None, :, 0, 0], block[None, :, 0, 1]
            x1, y1 = block[None, :, 1, 0], block[None, :, 1, 1]
            upward = (y0 <= point_y) & (point_y < y1)
            downward = (y1 <= point_y) & (point_y < y0)
            crossing_y = upward | downward
            safe_dy = torch.where(crossing_y, y1 - y0, torch.ones_like(y1))
            intersection_x = x0 + (point_y - y0) * (x1 - x0) / safe_dy
            right = intersection_x > point_x
            counts.add_((upward & right).sum(dim=1))
            counts.sub_((downward & right).sum(dim=1))
        winding[point_start:point_stop] = counts
    return winding.reshape(output_shape)


@torch.no_grad()
def crossing_parity(
    segments_uas,
    points_uas,
    *,
    point_chunk_size: int = 4096,
    segment_chunk_size: int = 16384,
) -> torch.Tensor:
    """Return modulo-two winding labels for source-plane points.

    A horizontal ray extending toward positive x is cast from each point. An
    edge crosses the ray when its y interval contains the query under the
    half-open rule ``min(y0,y1) <= y < max(y0,y1)`` and its intersection lies
    strictly to the right. The rule is orientation-independent and counts a
    shared vertex once rather than once per incident segment.

    Parameters
    ----------
    segments_uas:
        Caustic segments with shape ``[segment, 2, 2]``.
    points_uas:
        Query points with trailing Cartesian coordinate dimension two.
    point_chunk_size, segment_chunk_size:
        Working-set controls. They do not change the labels.
    """

    segments = torch.as_tensor(segments_uas)
    if segments.ndim != 3 or tuple(segments.shape[1:]) != (2, 2):
        raise ValueError("segments_uas must have shape [segment, 2, 2]")
    points = torch.as_tensor(
        points_uas,
        device=segments.device,
        dtype=segments.dtype,
    )
    if points.shape[-1:] != (2,):
        raise ValueError("points_uas must have trailing Cartesian dimension 2")
    if int(point_chunk_size) < 1 or int(segment_chunk_size) < 1:
        raise ValueError("point and segment chunk sizes must be positive")
    output_shape = points.shape[:-1]
    flat_points = points.reshape(-1, 2)
    parity = torch.zeros(flat_points.shape[0], device=segments.device, dtype=torch.int8)
    for point_start in range(0, flat_points.shape[0], int(point_chunk_size)):
        point_stop = min(flat_points.shape[0], point_start + int(point_chunk_size))
        query = flat_points[point_start:point_stop]
        counts = torch.zeros(query.shape[0], device=segments.device, dtype=torch.int64)
        point_x = query[:, 0, None]
        point_y = query[:, 1, None]
        for segment_start in range(0, segments.shape[0], int(segment_chunk_size)):
            segment_stop = min(
                segments.shape[0],
                segment_start + int(segment_chunk_size),
            )
            block = segments[segment_start:segment_stop]
            x0, y0 = block[None, :, 0, 0], block[None, :, 0, 1]
            x1, y1 = block[None, :, 1, 0], block[None, :, 1, 1]
            upward = (y0 <= point_y) & (point_y < y1)
            downward = (y1 <= point_y) & (point_y < y0)
            crossing_y = upward | downward
            safe_dy = torch.where(crossing_y, y1 - y0, torch.ones_like(y1))
            intersection_x = x0 + (point_y - y0) * (x1 - x0) / safe_dy
            counts.add_((crossing_y & (intersection_x > point_x)).sum(dim=1))
        parity[point_start:point_stop] = torch.remainder(counts, 2).to(torch.int8)
    return parity.reshape(output_shape)


@torch.no_grad()
def distance_to_segments(
    segments_uas,
    points_uas,
    *,
    point_chunk_size: int = 4096,
    segment_chunk_size: int = 16384,
) -> torch.Tensor:
    """Return minimum point-to-segment distances in microarcseconds.

    Work is chunked over both points and segments, so a diagnostic map does
    not materialize the full ``n_points x n_segments`` distance tensor.
    An empty segment set returns positive infinity for every query.
    """

    segments = torch.as_tensor(segments_uas)
    if segments.ndim != 3 or tuple(segments.shape[1:]) != (2, 2):
        raise ValueError("segments_uas must have shape [segment, 2, 2]")
    points = torch.as_tensor(
        points_uas,
        device=segments.device,
        dtype=segments.dtype,
    )
    if points.shape[-1:] != (2,):
        raise ValueError("points_uas must have trailing Cartesian dimension 2")
    if int(point_chunk_size) < 1 or int(segment_chunk_size) < 1:
        raise ValueError("point and segment chunk sizes must be positive")
    output_shape = points.shape[:-1]
    flat_points = points.reshape(-1, 2)
    result = torch.full(
        (flat_points.shape[0],),
        float("inf"),
        device=segments.device,
        dtype=segments.dtype,
    )
    for point_start in range(0, flat_points.shape[0], int(point_chunk_size)):
        point_stop = min(flat_points.shape[0], point_start + int(point_chunk_size))
        query = flat_points[point_start:point_stop]
        minimum_squared = torch.full(
            (query.shape[0],),
            float("inf"),
            device=segments.device,
            dtype=segments.dtype,
        )
        for segment_start in range(0, segments.shape[0], int(segment_chunk_size)):
            segment_stop = min(
                segments.shape[0],
                segment_start + int(segment_chunk_size),
            )
            block = segments[segment_start:segment_stop]
            start = block[None, :, 0, :]
            vector = block[None, :, 1, :] - start
            relative = query[:, None, :] - start
            denominator = vector.square().sum(dim=-1).clamp_min(
                torch.finfo(segments.dtype).tiny
            )
            fraction = (relative * vector).sum(dim=-1) / denominator
            closest = start + fraction.clamp(0.0, 1.0)[..., None] * vector
            squared = (query[:, None, :] - closest).square().sum(dim=-1)
            minimum_squared = torch.minimum(minimum_squared, squared.min(dim=1).values)
        result[point_start:point_stop] = torch.sqrt(minimum_squared)
    return result.reshape(output_shape)
