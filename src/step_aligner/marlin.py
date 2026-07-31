from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, VLM_PROMPT
from .models import CaptionedSegment, GroupedStep, Metadata, Segment


class MarlinClient(GeminiClient):
    def __init__(
        self,
        model: str = "NemoStation/Marlin-2B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        segment_dir: str | Path = "marlin_segments",
        retries: int = 1,
        sleep: float = 0.0,
    ):
        self.model = model
        self.retries = retries
        self.sleep = sleep
        self.segment_dir = Path(segment_dir)
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self._device = device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Marlin backend requires torch, transformers, qwen-vl-utils, and pillow."
            ) from exc

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        device_map: str | dict[str, str] = {"": device} if device != "auto" else "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self._processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)

    def caption_segment(self, segment: Segment, max_frames: int) -> CaptionedSegment:
        frames = uniform_sample_paths(segment.frame_paths, max_frames)
        prompt = (
            "The attached images are uniformly sampled frames from one short segment of an egocentric video, "
            "shown in chronological order. Treat them as a temporal sequence.\n\n"
            f"{VLM_PROMPT}"
        )
        text = self._call_image_sequence_text(frames, prompt, max_new_tokens=768)
        return CaptionedSegment(id=segment.id, start_ts=segment.start_ts, end_ts=segment.end_ts, caption=strip_think(text).strip())

    def _call_text(self, input_parts: list[dict[str, Any]]) -> str:
        text = "\n".join(str(part.get("text", "")) for part in input_parts if part.get("type") == "text")
        return self._call_text_only(text, max_new_tokens=1536)

    def group_steps(self, metadata: Metadata, captions: list[CaptionedSegment]) -> list[GroupedStep]:
        try:
            return super().group_steps(metadata, captions)
        except Exception as exc:
            print(f"[marlin-fallback] grouping used caption groups because Marlin did not return JSON: {exc}", flush=True)
            return fallback_group_steps(captions)

    def align_steps(self, metadata: Metadata, grouped: list[GroupedStep], video_duration: float) -> list[int]:
        try:
            return super().align_steps(metadata, grouped, video_duration)
        except Exception as exc:
            print(f"[marlin-fallback] alignment used chronological order because Marlin did not return JSON: {exc}", flush=True)
            return list(range(1, len(grouped) + 1))

    def score_coherence(self, metadata: Metadata, grouped: list[GroupedStep]) -> dict[str, Any]:
        try:
            return super().score_coherence(metadata, grouped)
        except Exception as exc:
            print(f"[marlin-fallback] QA used local heuristic because Marlin did not return JSON: {exc}", flush=True)
            captions = [item.caption.strip() for item in grouped if item.caption.strip()]
            vague_count = sum(1 for text in captions if len(text.split()) < 5 or text.lower() in {"no active task", "unknown"})
            coverage = 8.0 if len(captions) >= 3 else max(3.0, float(len(captions) * 2))
            relevance = max(3.0, 8.0 - vague_count)
            order = 8.0
            score = round((coverage + order + relevance) / 3.0, 2)
            return {
                "score": score,
                "coverage_score": coverage,
                "order_score": order,
                "relevance_score": relevance,
                "reasoning": "Marlin did not return parseable JSON for QA, so this score was estimated from caption count, order, and specificity.",
                "issues": ["QA used a local fallback rather than a model-generated JSON evaluation."],
            }

    def summarize_plan(self, metadata: Metadata, grouped: list[GroupedStep]) -> dict[str, Any]:
        try:
            return super().summarize_plan(metadata, grouped)
        except Exception as exc:
            print(f"[marlin-fallback] plan used local outline because Marlin did not return JSON: {exc}", flush=True)
            outline = [conversationalize(item.caption) for item in grouped[:6] if item.caption.strip()]
            if not outline:
                outline = ["You'll move through the visible task phases in order, using the transcript as the main guide."]
            return {
                "title": metadata.activity,
                "overview": "You'll work through the main visible phases in the same order they appear in the video.",
                "materials": [],
                "outline": outline,
                "cautions": [],
            }

    def _call_text_only(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
        out = out[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        del inputs, out
        self._empty_cuda_cache()
        return strip_think(text).strip()

    def _call_image_sequence_text(self, frame_paths: list[Path], prompt: str, max_new_tokens: int) -> str:
        import torch

        content: list[dict[str, str]] = []
        for frame_path in frame_paths:
            content.append({"type": "image", "image": str(frame_path)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
        out = out[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        del inputs, out
        self._empty_cuda_cache()
        return strip_think(text).strip()

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._empty_cuda_cache()

    unload = close

    def _empty_cuda_cache(self) -> None:
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def uniform_sample_paths(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    if max_items <= 1:
        return [paths[len(paths) // 2]]
    idxs = [round(i * (len(paths) - 1) / (max_items - 1)) for i in range(max_items)]
    return [paths[idx] for idx in idxs]


def fallback_group_steps(captions: list[CaptionedSegment]) -> list[GroupedStep]:
    grouped: list[GroupedStep] = []
    for item in captions:
        caption = " ".join(item.caption.split())
        if not caption:
            caption = "Continue the visible task activity."
        grouped.append(GroupedStep(caption=caption, start_ts=item.start_ts, end_ts=item.end_ts))
    return grouped


def conversationalize(caption: str) -> str:
    caption = " ".join(caption.strip().split())
    if not caption:
        return "You'll continue through the next visible phase of the task."
    first = caption[:1].lower() + caption[1:]
    return f"You'll {first}" if not first.lower().startswith(("you'll", "you will")) else caption


def strip_think(text: str) -> str:
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*<think>\s*", "", text, flags=re.DOTALL)
    return text
