from __future__ import annotations

from pathlib import Path

import cv2

from .models import FrameInfo


def sample_frames(video_path: Path, frames_dir: Path, every_n: int, image_size: int) -> tuple[list[FrameInfo], float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0

    frame_infos: list[FrameInfo] = []
    source_frame = 0
    sampled_index = 0
    ok, frame = cap.read()
    while ok:
        if source_frame % every_n == 0:
            resized = resize_center_crop(frame, image_size)
            out_path = frames_dir / f"frame_{sampled_index:06d}.jpg"
            cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            frame_infos.append(
                FrameInfo(
                    index=sampled_index,
                    source_frame=source_frame,
                    timestamp=source_frame / fps,
                    path=out_path,
                )
            )
            sampled_index += 1
        source_frame += 1
        ok, frame = cap.read()

    cap.release()
    if not frame_infos:
        raise RuntimeError(f"No frames sampled from video: {video_path}")
    return frame_infos, duration


def resize_center_crop(frame, image_size: int):
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = frame[top : top + side, left : left + side]
    return cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
