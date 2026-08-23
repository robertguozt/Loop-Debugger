"""Model access.

The loop talks to this interface, never to the SDK directly. That is what
makes the harness testable: `tests/` drives the whole loop with a scripted
model and no network, which is the only way to get deterministic coverage of
brake behaviour.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-5"


@dataclass
class Reply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class Model(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]]) -> Reply: ...


class AnthropicModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 4000,
    ):
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            raise SystemExit(
                "The anthropic package is required. Run: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                "Set ANTHROPIC_API_KEY in your environment, or pass --api-key."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict[str, Any]]) -> Reply:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return Reply(
            text=text,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
        )


class ScriptedModel:
    """Replays a fixed list of responses. Used by the test suite."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, system: str, messages: list[dict[str, Any]]) -> Reply:
        self.calls.append(messages)
        text = self._replies.pop(0) if self._replies else "{}"
        return Reply(text=text, tokens_in=100, tokens_out=50)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in prose and fences often enough that a bare
    json.loads is a reliability bug, not a style preference.
    """
    text = text.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth, in_str, esc = None, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"no JSON object found in model reply: {text[:200]!r}")
