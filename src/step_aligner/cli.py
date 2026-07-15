from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="step-align", description="Procedural video step alignment pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full pipeline for one video.")
    run.add_argument("--video", required=True, help="Path to input video.")
    run.add_argument("--metadata", required=True, help="Path to metadata JSON with activity and steps.")
    run.add_argument("--out", required=True, help="Output/cache directory.")
    _add_pipeline_options(run)
    run.set_defaults(func=_run_pipeline)

    folder = sub.add_parser("run-folder", help="Run every video in the built-in videos folder.")
    folder.add_argument("--videos", default="videos/raw", help="Folder containing local videos.")
    folder.add_argument("--metadata-dir", default="videos/metadata", help="Optional per-video metadata JSON folder.")
    folder.add_argument("--out", default="runs", help="Batch output/cache root.")
    folder.add_argument("--site-dir", default="docs", help="Rebuild the static site here after processing. Use empty string to skip.")
    folder.add_argument("--site-title", default="Step Alignment Results")
    folder.add_argument("--site-video-mode", choices=["copy", "link", "none"], default="copy", help="How generated site should reference source videos.")
    folder.add_argument("--recursive", action="store_true", help="Search nested video folders.")
    folder.add_argument("--limit", type=int, default=None, help="Maximum videos to process.")
    folder.add_argument("--skip-failed", action="store_true", help="Record failed videos and continue processing the rest.")
    folder.add_argument("--metadata-source", choices=["none", "gemini_plan", "default"], default="none")
    folder.add_argument("--default-activity", default=None, help="Activity name to use when no per-video metadata exists.")
    folder.add_argument("--target-steps", type=int, default=8)
    _add_pipeline_options(folder)
    folder.set_defaults(func=_run_video_folder)

    dataset = sub.add_parser("run-wearable", help="Run the pipeline over local facebook/wearable-ai files.")
    dataset.add_argument("--dataset-root", required=True, help="Local Hugging Face dataset root.")
    dataset.add_argument("--config", choices=["egoconv", "egolongqa", "egoproactive"], default="egoproactive")
    dataset.add_argument("--out", required=True, help="Batch output/cache root.")
    dataset.add_argument("--limit", type=int, default=None, help="Maximum rows to process.")
    dataset.add_argument("--skip-missing", action="store_true", help="Skip rows whose MP4s were not downloaded.")
    dataset.add_argument("--metadata-source", choices=["gemini_plan", "proactive_answers", "questions"], default="gemini_plan")
    dataset.add_argument("--target-steps", type=int, default=8)
    _add_pipeline_options(dataset)
    dataset.set_defaults(func=_run_wearable_dataset)

    site = sub.add_parser("build-site", help="Build a dark GitHub Pages static results site.")
    site.add_argument("--runs", required=True, help="Root containing completed run directories.")
    site.add_argument("--site-dir", required=True, help="Output directory for GitHub Pages files.")
    site.add_argument("--title", default="Wearable AI Step Alignment Results")
    site.add_argument("--video-mode", choices=["copy", "link", "none"], default="copy", help="How the site should reference source videos.")
    site.set_defaults(func=_build_site)
    return parser


def _add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Recompute all cached stages.")
    parser.add_argument("--resample-frames", action="store_true", help="Resample frames even when frames.json exists.")
    parser.add_argument("--frame-stride", type=int, default=4, help="Sample every Nth source video frame.")
    parser.add_argument("--image-size", type=int, default=256, help="Center-crop frame size.")
    parser.add_argument("--feature-backend", choices=["colorhist", "vjepa_hf"], default="vjepa_hf")
    parser.add_argument("--vjepa-repo", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oversegment", type=float, default=1.3)
    parser.add_argument("--min-segment-frames", type=int, default=4)
    parser.add_argument("--gemini-model", default="gemini-3.5-flash")
    parser.add_argument("--caption-frames", type=int, default=8)
    parser.add_argument("--max-caption-splits", type=int, default=1, help="Recursively split multi-action captions up to this depth.")


def _run_pipeline(args) -> None:
    from .pipeline import run_pipeline

    run_pipeline(args)


def _run_video_folder(args) -> None:
    from .folder import run_video_folder

    if args.site_dir == "":
        args.site_dir = None
    run_video_folder(args)


def _run_wearable_dataset(args) -> None:
    from .wearable import run_wearable_dataset

    run_wearable_dataset(args)


def _build_site(args) -> None:
    from .site import build_site

    build_site(args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
