from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import FrameInfo


def extract_features(
    frames: list[FrameInfo],
    backend: str,
    vjepa_repo: str,
    window_size: int,
    window_stride: int,
    device: str,
) -> np.ndarray:
    if backend == "colorhist":
        return extract_colorhist_features(frames)
    if backend == "vjepa_hf":
        return extract_vjepa_hf_window_features(frames, vjepa_repo, window_size, window_stride, device)
    raise ValueError(f"Unknown feature backend: {backend}")


def extract_colorhist_features(frames: list[FrameInfo]) -> np.ndarray:
    features = []
    for frame in frames:
        img = cv2.imread(str(frame.path))
        if img is None:
            raise RuntimeError(f"Could not read frame: {frame.path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist.astype(np.float32))
    return np.stack(features)


def extract_vjepa_hf_window_features(
    frames: list[FrameInfo],
    repo: str,
    window_size: int,
    window_stride: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "The vjepa_hf backend requires torch and transformers. Install with `pip install -e .[vjepa]` "
            "after installing the correct CUDA PyTorch build for your GPU."
        ) from exc

    model = AutoModel.from_pretrained(repo, torch_dtype=torch.float16 if device.startswith("cuda") else None)
    processor = AutoVideoProcessor.from_pretrained(repo)
    model.to(device)
    model.eval()

    accum: list[np.ndarray | None] = [None for _ in frames]
    counts = np.zeros(len(frames), dtype=np.float32)

    starts = list(range(0, max(1, len(frames) - window_size + 1), window_stride))
    if starts[-1] != max(0, len(frames) - window_size):
        starts.append(max(0, len(frames) - window_size))

    with torch.no_grad():
        for start in starts:
            end = min(len(frames), start + window_size)
            images = [Image.open(frame.path).convert("RGB") for frame in frames[start:end]]
            try:
                inputs = processor(videos=images, return_tensors="pt")
            except TypeError:
                inputs = processor(images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = _pool_model_output(outputs).detach().float().cpu().numpy()

            if pooled.ndim == 1:
                window_vecs = np.repeat(pooled[None, :], end - start, axis=0)
            elif pooled.shape[0] == end - start:
                window_vecs = pooled
            else:
                window_vecs = np.repeat(pooled.reshape(1, -1), end - start, axis=0)

            for offset, vec in enumerate(window_vecs):
                idx = start + offset
                accum[idx] = vec if accum[idx] is None else accum[idx] + vec
                counts[idx] += 1.0

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    filled = []
    for idx, item in enumerate(accum):
        if item is None:
            nearest = min(range(len(accum)), key=lambda j: abs(j - idx) if accum[j] is not None else 10**9)
            item = accum[nearest]
            counts[idx] = counts[nearest]
        filled.append((item / max(counts[idx], 1.0)).astype(np.float32))
    return np.stack(filled)


def _pool_model_output(outputs):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output.squeeze(0)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state.squeeze(0)
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        return hidden.mean(dim=0)
    first = outputs[0]
    if first.ndim >= 2:
        return first.squeeze(0).mean(dim=0)
    return first.squeeze(0)
