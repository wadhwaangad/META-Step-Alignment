from __future__ import annotations

import base64
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .gemini import GeminiClient


class MolmoClient(GeminiClient):
    """Molmo 2 client served through OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "allenai/molmo-2-8b",
        retries: int = 4,
        sleep: float = 2.0,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Molmo backend requires OPENROUTER_API_KEY. "
                "Set it with `$env:OPENROUTER_API_KEY=\"your_key_here\"`."
            )
        self.model = model
        self.retries = retries
        self.sleep = sleep
        self._api_key = api_key

    def _call_text(self, input_parts: list[dict[str, Any]]) -> str:
        content: list[dict[str, Any]] = []
        for item in input_parts:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{item['mime_type']};base64,{item['data']}",
                        },
                    }
                )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "X-OpenRouter-Title": "Step Aligner",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=120) as response:
                    data = json.load(response)
                text = data["choices"][0]["message"]["content"]
                if not text:
                    raise RuntimeError("OpenRouter returned an empty Molmo response")
                return str(text)
            except (HTTPError, URLError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.sleep * (attempt + 1))
        raise RuntimeError(f"Molmo request failed after {self.retries} attempts: {last_error}")

    # Captioning, grouping, alignment, QA, and plan generation are inherited
    # from GeminiClient; they all delegate to this client's _call_text method.
