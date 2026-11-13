"""Minimal guarded DeepSeek HTTP client using Python's standard library."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from urllib import error, request
from typing import Any


class DeepSeekError(RuntimeError):
    """Raised for configuration, transport, or response errors."""


@dataclass(frozen=True)
class DeepSeekConfig:
    model: str = "deepseek-flash"
    endpoint: str = "https://api.deepseek.com/chat/completions"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 60.0


class DeepSeekClient:
    """A client that never sends a request unless explicitly enabled."""

    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config

    @staticmethod
    def api_key_configured() -> bool:
        return bool(os.environ.get("DEEPSEEK_API_KEY"))

    def complete(self, prompt: str, *, allow_live_request: bool = False) -> dict[str, Any]:
        if not allow_live_request:
            raise DeepSeekError(
                "Live DeepSeek requests are disabled. Set allow_live_request=True "
                "only after explicit approval."
            )
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY is not configured.")

        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "thinking": {"type": "disabled"},
            }
        ).encode("utf-8")
        http_request = request.Request(
            self.config.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raise DeepSeekError(f"DeepSeek HTTP error {exc.code}.") from exc
        except error.URLError as exc:
            raise DeepSeekError("DeepSeek request could not be sent.") from exc
        except TimeoutError as exc:
            raise DeepSeekError("DeepSeek request timed out.") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DeepSeekError("DeepSeek returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise DeepSeekError("DeepSeek response must be a JSON object.")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekError("DeepSeek response has no choices.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise DeepSeekError("DeepSeek response has no textual message content.")
        return result
