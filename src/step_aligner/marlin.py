from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, VLM_PROMPT
from .models import CaptionedSegment, Segment


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

    def _call_text_only(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return strip_think(text).strip()

    def _call_image_sequence_text(self, frame_paths: list[Path], prompt: str, max_new_tokens: int) -> str:
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
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return strip_think(text).strip()


def uniform_sample_paths(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    if max_items <= 1:
        return [paths[len(paths) // 2]]
    idxs = [round(i * (len(paths) - 1) / (max_items - 1)) for i in range(max_items)]
    return [paths[idx] for idx in idxs]


def strip_think(text: str) -> str:
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*<think>\s*", "", text, flags=re.DOTALL)
    return text
