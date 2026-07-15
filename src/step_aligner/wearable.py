from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .gemini import GeminiClient
from .io import write_json
from .models import Metadata
from .pipeline import run_pipeline


CONFIG_FILES = {
    "egoconv": "wearable_ai_2026_egoconv_val_700.jsonl",
    "egolongqa": "wearable_ai_2026_egolongqa_val_700.jsonl",
    "egoproactive": "wearable_ai_2026_egoproactive_val_700.jsonl",
}


def iter_rows(dataset_root: Path, config: str, limit: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    jsonl_path = dataset_root / config / CONFIG_FILES[config]
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                return
            yield idx, json.loads(line)


def row_video_path(dataset_root: Path, config: str, row: dict[str, Any]) -> Path:
    return dataset_root / config / "val" / row["video_path"]


def metadata_from_row(
    row: dict[str, Any],
    config: str,
    source: str,
    gemini: GeminiClient | None,
    target_steps: int,
) -> Metadata:
    activity = _activity_from_row(row, config)
    if source == "gemini_plan":
        if gemini is None:
            raise ValueError("gemini_plan metadata requires a Gemini client")
        return Metadata(activity=activity, steps=gemini.generate_reference_steps(activity, _context_from_row(row, config), target_steps))
    if source == "proactive_answers":
        steps = _steps_from_proactive_answers(row)
        if steps:
            return Metadata(activity=activity, steps=steps)
    if source == "questions":
        steps = _steps_from_questions(row, config)
        if steps:
            return Metadata(activity=activity, steps=steps)
    return Metadata(activity=activity, steps=[activity])


def run_wearable_dataset(args) -> None:
    dataset_root = Path(args.dataset_root)
    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    gemini = GeminiClient(args.gemini_model) if args.metadata_source == "gemini_plan" else None

    manifest = []
    for idx, row in iter_rows(dataset_root, args.config, args.limit):
        video_path = row_video_path(dataset_root, args.config, row)
        if not video_path.exists():
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing video for row {idx}: {video_path}")

        video_id = Path(row["video_path"]).stem
        item_dir = run_root / args.config / video_id
        item_dir.mkdir(parents=True, exist_ok=True)
        metadata = metadata_from_row(row, args.config, args.metadata_source, gemini, args.target_steps)
        metadata_path = item_dir / "metadata.json"
        write_json(metadata_path, metadata)
        write_json(item_dir / "dataset_row.json", row)

        pipeline_args = _pipeline_args_from_dataset_args(args, video_path, metadata_path, item_dir)
        run_pipeline(pipeline_args)
        manifest.append(
            {
                "config": args.config,
                "index": idx,
                "video_id": video_id,
                "video_path": str(video_path),
                "run_dir": str(item_dir),
                "activity": metadata.activity,
                "n_reference_steps": len(metadata.steps),
            }
        )
        write_json(run_root / "manifest.json", manifest)


def _pipeline_args_from_dataset_args(args, video_path: Path, metadata_path: Path, item_dir: Path):
    class PipelineArgs:
        pass

    pipeline_args = PipelineArgs()
    pipeline_args.video = str(video_path)
    pipeline_args.metadata = str(metadata_path)
    pipeline_args.out = str(item_dir)
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


def _activity_from_row(row: dict[str, Any], config: str) -> str:
    if config == "egoproactive":
        return str(row.get("task") or row.get("query") or row.get("domain") or "Wearable AI activity")
    if config == "egolongqa":
        return str(row.get("category") or row.get("question") or "Wearable AI long-form QA")
    return str(row.get("task") or row.get("questions", ["Wearable AI conversation"])[0])


def _context_from_row(row: dict[str, Any], config: str) -> str:
    if config == "egoproactive":
        answers = "\n".join(_steps_from_proactive_answers(row))
        return f"Query: {row.get('query', '')}\nDomain: {row.get('domain', '')}\nReference assistant moments:\n{answers}"
    if config == "egolongqa":
        return f"Question: {row.get('question', '')}\nAnswer: {row.get('answer', '')}\nOptions: {row.get('mcq_options', '')}"
    questions = "\n".join(str(q) for q in row.get("questions", []))
    answers = "\n".join(str(a) for a in row.get("answers", []))
    return f"Questions:\n{questions}\nAnswers:\n{answers}"


def _steps_from_proactive_answers(row: dict[str, Any]) -> list[str]:
    steps = []
    for answer in row.get("answers", []):
        answer = str(answer).strip()
        if not answer or answer == "$silent$":
            continue
        answer = answer.replace("$interrupt$", "").strip()
        if answer:
            steps.append(answer)
    return _dedupe_preserve_order(steps)


def _steps_from_questions(row: dict[str, Any], config: str) -> list[str]:
    if config == "egolongqa":
        values = [row.get("question"), row.get("answer")]
    else:
        values = row.get("questions", []) + row.get("answers", [])
    return _dedupe_preserve_order([str(value).strip() for value in values if str(value).strip()])


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        key = value.lower()
        if key not in seen:
            output.append(value)
            seen.add(key)
    return output
