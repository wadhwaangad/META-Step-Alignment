from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metadata:
    activity: str
    steps: list[str]


@dataclass(frozen=True)
class FrameInfo:
    index: int
    source_frame: int
    timestamp: float
    path: Path


@dataclass(frozen=True)
class Segment:
    id: int
    start_index: int
    end_index: int
    start_ts: float
    end_ts: float
    frame_paths: list[Path]


@dataclass(frozen=True)
class CaptionedSegment:
    id: int
    start_ts: float
    end_ts: float
    caption: str


@dataclass(frozen=True)
class GroupedStep:
    caption: str
    start_ts: float
    end_ts: float
