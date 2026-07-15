from __future__ import annotations

import re
import traceback
import unicodedata
from pathlib import Path
from typing import Iterable

from .gemini import GeminiClient
from .io import load_metadata, write_json
from .models import Metadata
from .pipeline import run_pipeline
from .site import build_site


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def run_video_folder(args) -> None:
    videos_dir = Path(args.videos)
    metadata_dir = Path(args.metadata_dir)
    out_root = Path(args.out)
    site_dir = Path(args.site_dir) if args.site_dir else None

    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(videos_dir, recursive=args.recursive))
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"No videos found in {videos_dir}. Put MP4/MOV/MKV files there first.")

    gemini = GeminiClient(args.gemini_model) if args.metadata_source == "gemini_plan" else None
    manifest = []
    failures = []
    for video_path in videos:
        video_id = safe_id(video_path.relative_to(videos_dir).with_suffix("").as_posix())
        run_dir = out_root / video_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            metadata_path = metadata_dir / f"{video_path.stem}.json"
            if metadata_path.exists():
                metadata = load_metadata(metadata_path)
            else:
                metadata = infer_metadata(video_path, args, gemini)
                metadata_path = run_dir / "metadata.json"
                write_json(metadata_path, metadata)

            pipeline_args = pipeline_args_from_folder_args(args, video_path, metadata_path, run_dir)
            run_pipeline(pipeline_args)
            manifest.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "run_dir": str(run_dir),
                    "activity": metadata.activity,
                    "n_reference_steps": len(metadata.steps),
                }
            )
            write_json(out_root / "manifest.json", manifest)
        except Exception as exc:
            failure = {
                "video_id": video_id,
                "video_path": str(video_path),
                "run_dir": str(run_dir),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            write_json(out_root / "failures.json", failures)
            write_json(run_dir / "failed.json", failure)
            if not args.skip_failed:
                raise
            print(f"[skip-failed] {video_path}: {exc}", flush=True)

    if site_dir is not None:
        build_args = type(
            "BuildSiteArgs",
            (),
            {
                "runs": str(out_root),
                "site_dir": str(site_dir),
                "title": args.site_title,
                "video_mode": args.site_video_mode,
            },
        )
        build_site(build_args)


def iter_videos(videos_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(videos_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def infer_metadata(video_path: Path, args, gemini: GeminiClient | None) -> Metadata:
    activity = safe_ascii(args.default_activity or title_from_filename(video_path), "Wearable task")
    if args.metadata_source == "none":
        return neutral_metadata(args.target_steps)
    if args.metadata_source == "gemini_plan":
        if gemini is None:
            raise ValueError("gemini_plan metadata requires a Gemini client")
        context = safe_ascii(f"Video filename: {video_path.name}\nFolder: {video_path.parent.name}", "Local wearable video")
        steps = gemini.generate_reference_steps(activity, context, args.target_steps)
        return Metadata(activity=activity, steps=steps)
    return Metadata(activity=activity, steps=[activity])


def neutral_metadata(target_steps: int) -> Metadata:
    n_steps = max(1, target_steps)
    return Metadata(
        activity="Unlabeled local video",
        steps=[f"Observed procedural phase {idx + 1}" for idx in range(n_steps)],
    )


def title_from_filename(path: Path) -> str:
    name = re.sub(r"[_-]+", " ", path.stem)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:1].upper() + name[1:] if name else "Wearable task"


def safe_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._/-]+", "-", value).strip("-")
    return value.replace("/", "__") or "video"


def safe_ascii(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    encoded = normalized.encode("ascii", "ignore").decode("ascii")
    encoded = re.sub(r"\s+", " ", encoded).strip()
    return encoded or fallback


def pipeline_args_from_folder_args(args, video_path: Path, metadata_path: Path, run_dir: Path):
    class PipelineArgs:
        pass

    pipeline_args = PipelineArgs()
    pipeline_args.video = str(video_path)
    pipeline_args.metadata = str(metadata_path)
    pipeline_args.out = str(run_dir)
    pipeline_args.force = args.force
    pipeline_args.resample_frames = args.resample_frames
    pipeline_args.frame_stride = args.frame_stride
    pipeline_args.image_size = args.image_size
    pipeline_args.feature_backend = args.feature_backend
    pipeline_args.vjepa_repo = args.vjepa_repo
    pipeline_args.window_size = args.window_size
    pipeline_args.window_stride = args.window_stride
    pipeline_args.device = args.device
    pipeline_args.oversegment = args.oversegment
    pipeline_args.min_segment_frames = args.min_segment_frames
    pipeline_args.gemini_model = args.gemini_model
    pipeline_args.caption_frames = args.caption_frames
    pipeline_args.max_caption_splits = args.max_caption_splits
    return pipeline_args
