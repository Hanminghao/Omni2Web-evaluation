"""Gemini API client used by all model-based scoring stages."""

import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from scorer._types import GeminiTaskClient
from scorer.rate_limit import RPMLimiter

DEFAULT_GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_JUDGE_MODEL = "gemini-3.1-pro-preview"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

load_dotenv(Path.cwd() / ".env")
_LIMITERS: dict[tuple[str, int], RPMLimiter] = {}


def _parts(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    result = []
    for item in content or []:
        if item.get("type") == "text":
            result.append({"text": item.get("text", "")})
        elif item.get("type") == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            if not url.startswith("data:") or "," not in url:
                raise ValueError("Gemini judge images must be base64 data URLs")
            header, payload = url.split(",", 1)
            mime = header[5:].split(";", 1)[0] or "image/png"
            base64.b64decode(payload, validate=True)
            result.append({"inlineData": {"mimeType": mime, "data": payload}})
    return result


class GeminiClient:
    def __init__(self, api_key: str, model: str, rpm: int,
                 api_root: str | None = None, stream: bool | None = None):
        self.api_key = api_key
        self.api_root = (api_root or os.getenv("GEMINI_API_ROOT")
                         or DEFAULT_GEMINI_API_ROOT).rstrip("/")
        self.stream = (_env_flag("GEMINI_API_STREAM") if stream is None else stream)
        self.limiter = _LIMITERS.setdefault((model, rpm), RPMLimiter(rpm))

    @staticmethod
    def _stream_content(response: requests.Response) -> tuple[str, dict | None]:
        chunks = []
        usage = None
        for raw_line in response.iter_lines(decode_unicode=True):
            line = (raw_line or "").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usageMetadata") or usage
            for candidate in event.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if not part.get("thought"):
                        chunks.append(part.get("text", ""))
        return "".join(chunks), usage

    def call_text(self, messages, model=DEFAULT_JUDGE_MODEL, max_tokens=None,
                  include_thoughts=False, retries=3, backoff=2.0, timeout=300):
        contents = []
        system_parts = []
        for message in messages:
            parts = _parts(message.get("content"))
            if message.get("role") == "system":
                system_parts.extend(parts)
            else:
                role = "model" if message.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": parts})
        payload = {"contents": contents, "generationConfig": {}}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if include_thoughts:
            payload["generationConfig"]["thinkingConfig"] = {"includeThoughts": True}
        method = "streamGenerateContent?alt=sse" if self.stream else "generateContent"
        url = f"{self.api_root}/models/{model}:{method}"
        last_error = "request failed"
        for attempt in range(max(1, retries)):
            try:
                with self.limiter.slot():
                    response = requests.post(
                        url,
                        headers={"x-goog-api-key": self.api_key,
                                 "Content-Type": "application/json"},
                        json=payload,
                        timeout=timeout, stream=self.stream,
                    )
                if response.status_code == 200:
                    if self.stream:
                        text, usage = self._stream_content(response)
                    else:
                        data = response.json()
                        text = "".join(
                            part.get("text", "")
                            for candidate in data.get("candidates", [])
                            for part in candidate.get("content", {}).get("parts", [])
                            if not part.get("thought")
                        )
                        usage = data.get("usageMetadata")
                    if text.strip():
                        return {"content": text, "reasoning": "", "success": True,
                                "usage": usage}
                    last_error = "empty response"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    if response.status_code not in RETRYABLE_STATUS:
                        break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max(1, retries):
                time.sleep(min(60.0, backoff * (2 ** attempt)) + random.uniform(0, 1))
        return {"content": "", "reasoning": "", "success": False,
                "error": last_error, "usage": None}


def make_judge_client(judge_model: str = DEFAULT_JUDGE_MODEL, retries: int = 3,
                      backoff: float = 2.0, rpm: int = 60) -> GeminiTaskClient:
    del retries, backoff
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is required")
    return GeminiTaskClient(GeminiClient(key, judge_model, rpm))


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
