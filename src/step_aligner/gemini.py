from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

from .models import CaptionedSegment, GroupedStep, Metadata, Segment


VLM_PROMPT = """These frames are from an egocentric (first-person) video of someone performing a hands-on task. Describe what you see, with an emphasis on any action the person is performing. If no action is visible, say "No active task".
If there are multiple distinct actions (different tool, object, or motion), list each on a separate line.
Output ONLY the activity description(s), one per line."""


class GeminiClient:
    def __init__(self, model: str, retries: int = 4, sleep: float = 2.0):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use Gemini: `pip install google-genai`") from exc
        self._client = genai.Client()
        self.model = model
        self.retries = retries
        self.sleep = sleep

    def caption_segment(self, segment: Segment, max_frames: int) -> CaptionedSegment:
        frame_paths = uniform_sample(segment.frame_paths, max_frames)
        parts: list[dict[str, Any]] = [{"type": "text", "text": VLM_PROMPT}]
        for frame in frame_paths:
            parts.append(
                {
                    "type": "image",
                    "data": base64.b64encode(frame.read_bytes()).decode("utf-8"),
                    "mime_type": "image/jpeg",
                }
            )
        text = self._call_text(parts).strip()
        return CaptionedSegment(id=segment.id, start_ts=segment.start_ts, end_ts=segment.end_ts, caption=text)

    def group_steps(self, metadata: Metadata, captions: list[CaptionedSegment]) -> list[GroupedStep]:
        activities_text = "\n".join(
            f"[{item.start_ts:.1f}-{item.end_ts:.1f}s] {item.caption}" for item in captions
        )
        if is_unlabeled_metadata(metadata):
            prompt = f"""You are a video activity annotator. You group timestamped visual observations into coherent task steps and output ONLY a JSON array.

The video has no trusted title or reference steps. Infer the procedure only from the raw observations.

Raw observations:
{activities_text}

Rules:
- Write each caption as a clear instruction someone could follow, not a description of what was observed.
- Merge adjacent observations that describe the same continuous action.
- Keep distinct actions as separate steps when the object, tool, or motion changes.
- Absorb "No active task" segments into the nearest active group when appropriate.
- Target approximately {len(metadata.steps)} groups, but use fewer or more if the video content warrants it.
- Output JSON: [{{"caption": "...", "start_ts": 0.0, "end_ts": 18.0}}, ...]"""
        else:
            steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(metadata.steps))
            prompt = f"""You are a video activity annotator. You group timestamped activities into coherent task steps and output ONLY a JSON array.

You are converting raw visual observations from a video into clear, actionable instructions for performing a task.

Activity: {metadata.activity}
Reference steps:
{steps_text}
Raw observations:
{activities_text}

Rules:
- Use the activity name and reference steps to understand what the task is about. Map generic visual descriptions to the correct task-specific terms.
- Write each caption as a clear instruction someone could follow to perform the task, not a description of what was observed.
- Merge adjacent observations that describe the same continuous action.
- Target approximately {len(metadata.steps)} groups; it is OK to have a few more or fewer if the video content warrants it.
- Reference steps annotated with (mistake, ...), (interruption, ...), or (fix) MUST each appear as their own separate segment.
- Absorb "No active task" segments into the nearest active group.
- Output JSON: [{{"caption": "...", "start_ts": 0.0, "end_ts": 18.0}}, ...]"""
        rows = _parse_json(self._call_text([{"type": "text", "text": prompt}]))
        return [GroupedStep(caption=str(row["caption"]), start_ts=float(row["start_ts"]), end_ts=float(row["end_ts"])) for row in rows]

    def generate_reference_steps(self, activity: str, context: str, target_steps: int) -> list[str]:
        prompt = f"""You are preparing a procedural reference outline for an egocentric wearable-camera video.

Activity:
{activity}

Dataset context:
{context}

Write {target_steps} concise procedural steps that could plausibly describe the main hands-on phases of this activity. Use imperative wording. Do not include timing, confidence, or commentary.

Output ONLY a JSON array of strings."""
        values = _parse_json(self._call_text([{"type": "text", "text": prompt}]))
        steps = [str(value).strip() for value in values if str(value).strip()]
        if not steps:
            raise RuntimeError("Gemini returned no reference steps")
        return steps

    def align_steps(self, metadata: Metadata, grouped: list[GroupedStep], video_duration: float) -> list[int]:
        if is_unlabeled_metadata(metadata):
            return list(range(1, len(grouped) + 1))
        steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(metadata.steps))
        segments_text = "\n".join(
            f"{i + 1}. [{item.start_ts:.1f}-{item.end_ts:.1f}s, {100 * item.start_ts / max(video_duration, 1):.1f}%] {item.caption}"
            for i, item in enumerate(grouped)
        )
        prompt = f"""You are given a list of procedural steps and a chronological list of video segment captions. Your task is to assign each segment caption to the best-matching step.

STEPS (numbered 1-{len(metadata.steps)}):
{steps_text}

SEGMENTS (chronological, {len(grouped)} total, video duration {video_duration:.1f}s):
{segments_text}

INSTRUCTIONS:
- For each segment, output the 1-based step number that best describes the observed action.
- Use 0 if a segment has no matching step.
- Consider semantic match and temporal position.
- Multiple consecutive segments can map to the same step.
- Not every step needs to have a matching segment.
Output ONLY a JSON array of integers - one per segment, in order. The array must have exactly {len(grouped)} elements."""
        values = _parse_json(self._call_text([{"type": "text", "text": prompt}]))
        if len(values) != len(grouped):
            raise RuntimeError(f"Gemini alignment returned {len(values)} values for {len(grouped)} segments")
        return [int(value) for value in values]

    def score_coherence(self, metadata: Metadata, grouped: list[GroupedStep]) -> dict[str, Any]:
        captions_text = "\n".join(f"{i + 1}. {item.caption}" for i, item in enumerate(grouped))
        if is_unlabeled_metadata(metadata):
            prompt = f"""You are evaluating whether a sequence of captions from an unlabeled egocentric video forms a useful procedural transcript.

There is no trusted activity title and no trusted reference-step list. Judge only the observed caption sequence.

OBSERVED CAPTIONS (chronological, {len(grouped)} total):
{captions_text}

Evaluate the caption sequence on:
1. COVERAGE (1-10): Do the captions cover the distinct visible hands-on phases, without excessive repetition?
2. ORDER (1-10): Does the sequence make temporal/procedural sense?
3. RELEVANCE (1-10): Are captions specific and actionable enough to help a person understand what happened?
Then give an OVERALL score (1-10).

Do not penalize the transcript for lacking a known task label. Penalize only missing phases, nonsense order, vague captions, or redundancy.

Output ONLY a JSON object:
{{"score": ..., "coverage_score": ..., "order_score": ..., "relevance_score": ..., "inferred_activity": "...", "reasoning": "...", "issues": [...]}}"""
        else:
            steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(metadata.steps))
            prompt = f"""You are evaluating whether a sequence of visual captions could serve as instructions for someone to perform the activity.

ACTIVITY: {metadata.activity}
REFERENCE PROCEDURAL STEPS:
{steps_text}
OBSERVED CAPTIONS (chronological, {len(grouped)} total):
{captions_text}

Evaluate the caption sequence on:
1. COVERAGE (1-10)
2. ORDER (1-10)
3. RELEVANCE (1-10)
Then give an OVERALL score (1-10).

Output ONLY a JSON object:
{{"score": ..., "coverage_score": ..., "order_score": ..., "relevance_score": ..., "reasoning": "...", "issues": [...]}}"""
        return dict(_parse_json(self._call_text([{"type": "text", "text": prompt}])))

    def _call_text(self, input_parts: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                if hasattr(self._client, "interactions"):
                    response = self._client.interactions.create(model=self.model, input=input_parts)
                    return response.output_text
                response = self._client.models.generate_content(model=self.model, contents=_legacy_contents(input_parts))
                return response.text
            except Exception as exc:
                last_error = exc
                time.sleep(self.sleep * (attempt + 1))
        raise RuntimeError(f"Gemini request failed after {self.retries} attempts: {last_error}")


def is_unlabeled_metadata(metadata: Metadata) -> bool:
    return metadata.activity == "Unlabeled local video" and all(
        step.startswith("Observed procedural phase ") for step in metadata.steps
    )


def uniform_sample(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    if max_items <= 1:
        return [paths[len(paths) // 2]]
    idxs = [round(i * (len(paths) - 1) / (max_items - 1)) for i in range(max_items)]
    return [paths[idx] for idx in idxs]


def _parse_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start = min([idx for idx in [text.find("["), text.find("{")] if idx != -1], default=-1)
        end = max(text.rfind("]"), text.rfind("}"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _legacy_contents(input_parts: list[dict[str, Any]]) -> list[Any]:
    parts: list[Any] = []
    for item in input_parts:
        if item["type"] == "text":
            parts.append(item["text"])
        elif item["type"] == "image":
            parts.append({"mime_type": item["mime_type"], "data": item["data"]})
    return parts
