from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .models import FrameInfo, Segment


def expected_steps_from_duration(duration: float, fallback: int) -> int:
    if duration > 0:
        return max(1, math.ceil((duration / 60.0) * 2.5))
    return max(1, fallback)


def target_segment_count(expected_steps: int, oversegment: float, n_frames: int) -> int:
    return max(1, min(n_frames, math.floor(expected_steps * oversegment)))


def build_temporal_dendrogram(features: np.ndarray, frames: list[FrameInfo]) -> dict[str, Any]:
    n_frames = len(frames)
    if n_frames == 1:
        return {
            "root": 0,
            "nodes": {
                "0": {
                    "id": 0,
                    "children": [],
                    "distance": 0.0,
                    "start_index": 0,
                    "end_index": 0,
                }
            },
        }

    scaled = StandardScaler().fit_transform(features)
    connectivity = diags([np.ones(n_frames - 1), np.ones(n_frames - 1)], offsets=[-1, 1], shape=(n_frames, n_frames))
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        compute_distances=True,
        linkage="ward",
        connectivity=connectivity,
    )
    model.fit(scaled)

    nodes: dict[str, dict[str, Any]] = {}
    for idx in range(n_frames):
        nodes[str(idx)] = {
            "id": idx,
            "children": [],
            "distance": 0.0,
            "start_index": idx,
            "end_index": idx,
        }

    for merge_idx, (left_id, right_id) in enumerate(model.children_):
        node_id = n_frames + merge_idx
        left = nodes[str(int(left_id))]
        right = nodes[str(int(right_id))]
        nodes[str(node_id)] = {
            "id": node_id,
            "children": [int(left_id), int(right_id)],
            "distance": float(model.distances_[merge_idx]),
            "start_index": min(left["start_index"], right["start_index"]),
            "end_index": max(left["end_index"], right["end_index"]),
        }

    return {"root": int(2 * n_frames - 2), "nodes": nodes}


def partition_dendrogram(
    dendrogram: dict[str, Any],
    frames: list[FrameInfo],
    target_count: int,
    min_frames: int,
) -> list[Segment]:
    nodes = dendrogram["nodes"]
    active = [int(dendrogram["root"])]

    while len(active) < target_count:
        candidates = [
            node_id
            for node_id in active
            if nodes[str(node_id)]["children"]
            and _span_len(nodes[str(node_id)]) >= max(2, min_frames * 2)
        ]
        if not candidates:
            break
        split_id = max(candidates, key=lambda node_id: nodes[str(node_id)]["distance"])
        active.remove(split_id)
        children = nodes[str(split_id)]["children"]
        active.extend(int(child) for child in children)

    runs = sorted(
        [(nodes[str(node_id)]["start_index"], nodes[str(node_id)]["end_index"]) for node_id in active],
        key=lambda span: span[0],
    )
    runs = _merge_short_runs(runs, min_frames)
    return [_make_segment(i, start, end, frames) for i, (start, end) in enumerate(runs)]


def dendrogram_child_segments(
    dendrogram: dict[str, Any],
    segment: Segment,
    frames: list[FrameInfo],
) -> list[Segment]:
    nodes = dendrogram["nodes"]
    match = None
    for node in nodes.values():
        if node["start_index"] == segment.start_index and node["end_index"] == segment.end_index:
            match = node
            break
    if not match or not match["children"]:
        return midpoint_segments(segment, frames)

    children = sorted((nodes[str(int(child))] for child in match["children"]), key=lambda node: node["start_index"])
    return [
        _make_segment(idx, int(child["start_index"]), int(child["end_index"]), frames)
        for idx, child in enumerate(children)
    ]


def midpoint_segments(segment: Segment, frames: list[FrameInfo]) -> list[Segment]:
    if segment.end_index <= segment.start_index:
        return [segment]
    midpoint = (segment.start_index + segment.end_index + 1) // 2
    return [
        _make_segment(0, segment.start_index, midpoint - 1, frames),
        _make_segment(1, midpoint, segment.end_index, frames),
    ]


def _span_len(node: dict[str, Any]) -> int:
    return int(node["end_index"]) - int(node["start_index"]) + 1


def _merge_short_runs(runs: list[tuple[int, int]], min_frames: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and (end - start + 1) < min_frames:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    if len(merged) > 1:
        start, end = merged[-1]
        if (end - start + 1) < min_frames:
            prev_start, _ = merged[-2]
            merged[-2] = (prev_start, end)
            merged.pop()
    return merged


def _make_segment(segment_id: int, start: int, end: int, frames: list[FrameInfo]) -> Segment:
    end_ts = frames[end].timestamp
    if end + 1 < len(frames):
        end_ts = frames[end + 1].timestamp
    return Segment(
        id=segment_id,
        start_index=start,
        end_index=end,
        start_ts=frames[start].timestamp,
        end_ts=end_ts,
        frame_paths=[Path(frame.path) for frame in frames[start : end + 1]],
    )
