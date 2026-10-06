"""A small structured-output wrapper and an OpenCode Go adapter."""

import json
import os
from abc import ABC, abstractmethod
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class LLMResponseError(ValueError):
    """The provider returned incomplete, malformed, or invalid output."""


class LLM(ABC):
    def generate[T: BaseModel](
        self, *, system: str, prompt: str, response_model: type[T]
    ) -> T:
        """Request JSON matching a Pydantic model and validate it without retries."""
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        instructions = (
            f"{system}\n\nReturn exactly one JSON object matching the schema below. "
            "Do not include Markdown fences, commentary, or additional text.\n\n"
            f"JSON schema:\n{schema}"
        )
        content = self._complete(system=instructions, prompt=prompt)
        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            raise LLMResponseError(
                f"Invalid {response_model.__name__} response: {error}"
            ) from error

    @abstractmethod
    def _complete(self, *, system: str, prompt: str) -> str:
        """Return the provider's text response, raising on incomplete output."""


class OpenCodeGoLLM(LLM):
    """Use a Go model that supports the Chat Completions endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 32768,
    ) -> None:
        load_dotenv(".env", override=False)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENCODE_GO_API_KEY")
        if not self.api_key:
            raise ValueError("Pass api_key or set OPENCODE_GO_API_KEY in .env or the environment.")
        if not model.strip():
            raise ValueError("Specify an OpenCode Go Chat Completions model ID.")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.session_id = str(uuid4())

    def _complete(self, *, system: str, prompt: str) -> str:
        response = httpx.post(
            "https://opencode.ai/zen/go/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "ai-camera-roll/0.1.0",
                "x-opencode-session": self.session_id,
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            choice = response.json()["choices"][0]
            finish_reason = choice["finish_reason"]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise LLMResponseError("Malformed OpenCode Go completion response.") from error
        if finish_reason == "length":
            raise LLMResponseError(
                "OpenCode Go truncated the response. Increase max_tokens or reduce the request size."
            )
        if finish_reason != "stop":
            raise LLMResponseError(f"OpenCode Go completion ended with {finish_reason!r}.")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("OpenCode Go returned no text content.")
        return content
