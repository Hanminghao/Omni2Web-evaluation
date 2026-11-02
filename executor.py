"""Fixed Track C executor using the official DeepSeek API."""

from __future__ import annotations

import json
import os
import random
import time

import requests

from scorer.rate_limit import RPMLimiter

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class DeepSeekExecutor:
    def __init__(self, api_key: str | None = None, timeout: int = 600,
                 retries: int = 8, max_tokens: int = 65536,
                 rpm: int = 60, stream: bool = False):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.stream = stream
        self.limiter = RPMLimiter(rpm)

    def _decode_stream(self, response: requests.Response) -> tuple[str, str | None, dict | None]:
        chunks = []
        finish_reason = None
        usage = None
        for raw_line in response.iter_lines(decode_unicode=True):
            line = (raw_line or "").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                chunks.append(delta.get("content") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
        return "".join(chunks), finish_reason, usage

    def generate(self, prompt: str) -> dict:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "request failed"
        for attempt in range(max(1, self.retries)):
            try:
                with self.limiter.slot():
                    response = requests.post(
                        DEEPSEEK_URL, headers=headers, json=payload,
                        timeout=self.timeout, stream=self.stream,
                    )
                if response.status_code == 200:
                    if self.stream:
                        text, finish_reason, usage = self._decode_stream(response)
                    else:
                        data = response.json()
                        choices = data.get("choices") or []
                        message = choices[0].get("message") or {} if choices else {}
                        text = message.get("content") or ""
                        finish_reason = choices[0].get("finish_reason") if choices else None
                        usage = data.get("usage")
                    if text.strip():
                        return {
                            "ok": True,
                            "response": text,
                            "finish_reason": finish_reason,
                            "usage": usage,
                        }
                    last_error = "empty response"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    if response.status_code not in _RETRYABLE:
                        break
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max(1, self.retries):
                time.sleep(min(60.0, 2 ** attempt) + random.uniform(0, 1))
        return {"ok": False, "response": "", "error": last_error}
