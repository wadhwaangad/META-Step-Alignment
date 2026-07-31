from __future__ import annotations

from pathlib import Path

from .gemini import GeminiClient
from .marlin import MarlinClient


def make_model_client(args, out_dir: str | Path):
    backend = getattr(args, "model_backend", "gemini")
    if backend == "gemini":
        return GeminiClient(args.gemini_model)
    if backend == "marlin":
        return MarlinClient(
            model=args.marlin_model,
            device=args.marlin_device,
            dtype=args.marlin_dtype,
            segment_dir=Path(out_dir) / "marlin_segments",
        )
    raise ValueError(f"Unknown model backend: {backend}")
