from __future__ import annotations

from pathlib import Path

import numpy as np

from .cluster import (
    build_temporal_dendrogram,
    dendrogram_child_segments,
    expected_steps_from_duration,
    partition_dendrogram,
    target_segment_count,
)
from .clients import make_model_client
from .features import extract_features
from .io import load_metadata, read_json, write_json, write_jsonl
from .models import CaptionedSegment, GroupedStep, Segment
from .video import sample_frames


def run_pipeline(args) -> None:
    video_path = Path(args.video)
    metadata_path = Path(args.metadata)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_client = None

    try:
        metadata = load_metadata(metadata_path)

        frames_json = out_dir / "frames.json"
        if frames_json.exists() and not args.force:
            frames_data = read_json(frames_json)
            if args.resample_frames:
                frame_infos, duration = sample_frames(video_path, out_dir / "frames", args.frame_stride, args.image_size)
                write_json(frames_json, {"duration": duration, "frames": frame_infos})
            else:
                frame_infos, duration = _load_frames(frames_data)
        else:
            frame_infos, duration = sample_frames(video_path, out_dir / "frames", args.frame_stride, args.image_size)
            write_json(frames_json, {"duration": duration, "frames": frame_infos})

        features_path = out_dir / "features.npz"
        if features_path.exists() and not args.force:
            features = np.load(features_path)["features"]
        else:
            features = extract_features(
                frame_infos,
                backend=args.feature_backend,
                vjepa_repo=args.vjepa_repo,
                window_size=args.window_size,
                window_stride=args.window_stride,
                device=args.device,
            )
            np.savez_compressed(features_path, features=features)
            clear_cuda_cache()

        expected_steps = len(metadata.steps)
        if getattr(args, "adaptive_steps", False):
            expected_steps = expected_steps_from_duration(duration, fallback=max(1, len(metadata.steps)))
        target_count = target_segment_count(expected_steps, args.oversegment, len(frame_infos))

        dendrogram_path = out_dir / "dendrogram.json"
        if dendrogram_path.exists() and not args.force:
            dendrogram = read_json(dendrogram_path)
        else:
            dendrogram = build_temporal_dendrogram(features, frame_infos)
            write_json(dendrogram_path, dendrogram)

        segments_path = out_dir / "segments.json"
        if segments_path.exists() and not args.force:
            segments = _segments_from_json(read_json(segments_path))
        else:
            segments = partition_dendrogram(dendrogram, frame_infos, target_count, args.min_segment_frames)
            write_json(segments_path, segments)

        def get_model_client():
            nonlocal model_client
            if model_client is None:
                model_client = make_model_client(args, out_dir)
            return model_client

        captions_path = out_dir / "captions.json"
        if captions_path.exists() and not args.force:
            captions = _captions_from_json(read_json(captions_path))
        else:
            captions = caption_segments_recursive(
                get_model_client(),
                segments,
                args.caption_frames,
                args.max_caption_splits,
                dendrogram,
                frame_infos,
            )
            write_json(captions_path, captions)

        grouped_path = out_dir / "grouped_steps.json"
        if grouped_path.exists() and not args.force:
            grouped = _grouped_from_json(read_json(grouped_path))
        else:
            grouped = get_model_client().group_steps(metadata, captions)
            write_json(grouped_path, grouped)

        alignment_path = out_dir / "alignment.json"
        if alignment_path.exists() and not args.force:
            alignment = read_json(alignment_path)
        else:
            alignment = get_model_client().align_steps(metadata, grouped, duration)
            write_json(alignment_path, alignment)

        qa_path = out_dir / "qa.json"
        if qa_path.exists() and not args.force:
            qa = read_json(qa_path)
        else:
            qa = get_model_client().score_coherence(metadata, grouped)
            qa["local_checks"] = local_checks(grouped, duration)
            qa["segmentation"] = {"expected_steps": expected_steps, "target_segments": target_count}
            write_json(qa_path, qa)

        plan_path = out_dir / "plan.json"
        if plan_path.exists() and not args.force:
            plan = read_json(plan_path)
            if not is_synthetic_counterfactual_plan(plan):
                plan = get_model_client().summarize_plan(metadata, grouped)
                write_json(plan_path, plan)
        else:
            plan = get_model_client().summarize_plan(metadata, grouped)
            write_json(plan_path, plan)

        rows = []
        for idx, item in enumerate(grouped):
            rows.append(
                {
                    "start_ts": item.start_ts,
                    "end_ts": item.end_ts,
                    "caption": item.caption,
                    "step_index": alignment[idx] if idx < len(alignment) else None,
                }
            )
        write_jsonl(out_dir / "transcript.jsonl", rows)
        write_json(
            out_dir / "run_summary.json",
            {
                "video": str(video_path),
                "metadata": str(metadata_path),
                "out": str(out_dir),
                "qa": qa,
                "plan": plan,
                "segmentation": {"expected_steps": expected_steps, "target_segments": target_count},
            },
        )
    finally:
        if model_client is not None:
            close_model_client(model_client)


def close_model_client(model_client) -> None:
    close = getattr(model_client, "close", None) or getattr(model_client, "unload", None)
    if close is not None:
        close()
    else:
        clear_cuda_cache()


def clear_cuda_cache() -> None:
    try:
        import gc
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def caption_segments_recursive(
    gemini,
    segments: list[Segment],
    caption_frames: int,
    max_splits: int,
    dendrogram: dict,
    frames,
) -> list[CaptionedSegment]:
    captions: list[CaptionedSegment] = []
    next_id = 0

    def visit(segment: Segment, depth: int) -> None:
        nonlocal next_id
        captioned = gemini.caption_segment(segment, caption_frames)
        lines = [line.strip() for line in captioned.caption.splitlines() if line.strip()]
        can_split = depth < max_splits and len(lines) > 1 and len(segment.frame_paths) >= 4
        if not can_split:
            captions.append(
                CaptionedSegment(
                    id=next_id,
                    start_ts=captioned.start_ts,
                    end_ts=captioned.end_ts,
                    caption=captioned.caption,
                )
            )
            next_id += 1
            return

        for child in dendrogram_child_segments(dendrogram, segment, frames):
            if child.start_index == segment.start_index and child.end_index == segment.end_index:
                captions.append(
                    CaptionedSegment(
                        id=next_id,
                        start_ts=captioned.start_ts,
                        end_ts=captioned.end_ts,
                        caption=captioned.caption,
                    )
                )
                next_id += 1
                return
            visit(child, depth + 1)

    for segment in segments:
        visit(segment, 0)
    return captions


def local_checks(grouped: list[GroupedStep], duration: float) -> dict[str, object]:
    gaps = []
    previous = 0.0
    for item in grouped:
        if item.start_ts > previous + 0.5:
            gaps.append({"start_ts": previous, "end_ts": item.start_ts})
        previous = max(previous, item.end_ts)
    if duration and previous < duration - 0.5:
        gaps.append({"start_ts": previous, "end_ts": duration})
    return {"has_full_coverage": len(gaps) == 0, "gaps": gaps}


def is_synthetic_counterfactual_plan(plan: object) -> bool:
    """Return whether a cached plan uses the current user-facing interaction schema."""
    if not isinstance(plan, dict):
        return False
    mistake = plan.get("mistake")
    correction = plan.get("correction")
    return (
        plan.get("taxonomy_version") == 2
        and
        isinstance(plan.get("counterfactual_instruction"), str)
        and isinstance(mistake, dict)
        and isinstance(correction, dict)
        and isinstance(plan.get("steps"), list)
    )


def _load_frames(data):
    from .models import FrameInfo

    return [
        FrameInfo(index=int(row["index"]), source_frame=int(row["source_frame"]), timestamp=float(row["timestamp"]), path=Path(row["path"]))
        for row in data["frames"]
    ], float(data["duration"])


def _segments_from_json(rows) -> list[Segment]:
    return [
        Segment(
            id=int(row["id"]),
            start_index=int(row["start_index"]),
            end_index=int(row["end_index"]),
            start_ts=float(row["start_ts"]),
            end_ts=float(row["end_ts"]),
            frame_paths=[Path(path) for path in row["frame_paths"]],
        )
        for row in rows
    ]


def _captions_from_json(rows) -> list[CaptionedSegment]:
    return [
        CaptionedSegment(id=int(row["id"]), start_ts=float(row["start_ts"]), end_ts=float(row["end_ts"]), caption=str(row["caption"]))
        for row in rows
    ]


def _grouped_from_json(rows) -> list[GroupedStep]:
    return [GroupedStep(caption=str(row["caption"]), start_ts=float(row["start_ts"]), end_ts=float(row["end_ts"])) for row in rows]
