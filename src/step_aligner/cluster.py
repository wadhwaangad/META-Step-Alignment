from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .models import FrameInfo, Segment


def temporal_segments(
    features: np.ndarray,
    frames: list[FrameInfo],
    expected_steps: int,
    oversegment: float,
    min_frames: int,
) -> list[Segment]:
    n_frames = len(frames)
    target = max(1, min(n_frames, math.floor(expected_steps * oversegment)))
    if target == 1 or n_frames <= min_frames:
        return [_make_segment(0, 0, n_frames - 1, frames)]

    scaled = StandardScaler().fit_transform(features)
    connectivity = diags([np.ones(n_frames - 1), np.ones(n_frames - 1)], offsets=[-1, 1], shape=(n_frames, n_frames))
    model = AgglomerativeClustering(n_clusters=target, linkage="ward", connectivity=connectivity)
    labels = model.fit_predict(scaled)

    runs: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, n_frames):
        if labels[idx] != labels[idx - 1]:
            runs.append((start, idx - 1))
            start = idx
    runs.append((start, n_frames - 1))
    runs = _merge_short_runs(runs, min_frames)
    return [_make_segment(i, start, end, frames) for i, (start, end) in enumerate(runs)]


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
