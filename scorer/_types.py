from typing import Protocol


class TextClient(Protocol):
    def call_text(self, messages, model, max_tokens=None, include_thoughts=False,
                  retries=3, backoff=2.0, timeout=300): ...


class GeminiTaskClient:
    def __init__(self, client: TextClient):
        self.client = client
