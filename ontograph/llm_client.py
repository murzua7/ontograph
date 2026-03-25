"""LLM client with multiple backends.

Modes:
  - "anthropic": Direct Anthropic API (requires ANTHROPIC_API_KEY)
  - "openai": OpenAI-compatible API (works with Ollama at localhost:11434)
  - "ollama": Shortcut for OpenAI mode pointing at Ollama
  - "mock": Returns predefined responses (for testing)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict[str, int] | None = None


def _extract_json(text: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


class LLMClient:
    """Multi-backend LLM client for structured extraction."""

    def __init__(
        self,
        mode: str = "anthropic",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.mode = mode
        self._mock_responses: list[str] = []
        self._mock_index = 0

        if mode == "ollama":
            self.mode = "openai"
            self.base_url = base_url or "http://localhost:11434/v1"
            self.api_key = "ollama"
            self.model = model or "llama3.1:8b"
        elif mode == "openai":
            self.base_url = base_url or "https://api.openai.com/v1"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.model = model or "gpt-4o-mini"
        elif mode == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.model = model or "claude-sonnet-4-20250514"
            self.base_url = None
        elif mode == "mock":
            self.model = "mock"
            self.base_url = None
            self.api_key = ""
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_mock_responses(self, responses: list[str]) -> None:
        """Set predefined responses for mock mode."""
        self._mock_responses = responses
        self._mock_index = 0

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        """Send a chat completion request."""
        if self.mode == "mock":
            return self._mock_chat()
        elif self.mode == "anthropic":
            return self._anthropic_chat(messages, temperature)
        else:
            return self._openai_chat(messages, temperature)

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict | list:
        """Send a chat request and parse the response as JSON."""
        response = self.chat(messages, temperature)
        return _extract_json(response.content)

    def _mock_chat(self) -> LLMResponse:
        if self._mock_index < len(self._mock_responses):
            content = self._mock_responses[self._mock_index]
            self._mock_index += 1
        else:
            content = '{"entities": [], "relations": []}'
        return LLMResponse(content=content, model="mock")

    def _anthropic_chat(self, messages: list[dict[str, str]], temperature: float) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic SDK required. Install with: uv add anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Convert to Anthropic format (separate system from user/assistant)
        system = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system if system else anthropic.NOT_GIVEN,
            messages=api_messages,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )

    def _openai_chat(self, messages: list[dict[str, str]], temperature: float) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI SDK required. Install with: uv add openai")

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            usage=usage,
        )
