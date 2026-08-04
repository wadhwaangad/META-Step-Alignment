from __future__ import annotations

from pathlib import Path

from .gemini import GeminiClient
from .molmo import MolmoClient


def make_model_client(args, out_dir: str | Path):
    backend = getattr(args, "model_backend", "gemini")
    if backend == "gemini":
        return GeminiClient(args.gemini_model)
    if backend == "molmo":
        return MolmoClient(model=args.molmo_model)
    raise ValueError(f"Unknown model backend: {backend}")
