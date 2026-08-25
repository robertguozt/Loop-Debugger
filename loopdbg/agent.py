"""The autonomous control loop.

:meth:`DebugAgent.run_debug_cycle` drives a multi-turn Anthropic tool-use
conversation: it sends the transcript, parses ``tool_use`` blocks out of the
response, executes the matching Python tool, returns ``tool_result`` blocks, and
repeats until the model calls ``report_resolution`` or the step budget runs out.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast

from .config import AgentConfig
from .prompts import PHASE_HINTS, SYSTEM_PROMPT, initial_task_message
from .tools import TERMINAL_TOOL, TOOL_SCHEMAS, ToolBox, ToolOutcome

log = logging.getLogger(__name__)

MessageParam = dict[str, Any]


class Phase(StrEnum):
    """Coarse state of the investigation, inferred from which tools have run."""

    REPRODUCE = "reproduce"
    LOCALIZE = "localize"
    PATCH = "patch"
    VERIFY = "verify"
    DONE = "done"


class StopReason(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    MAX_STEPS = "max_steps"
    API_ERROR = "api_error"
    NO_TOOL_CALL = "no_tool_call"


@dataclass(slots=True)
class StepRecord:
    """One tool invocation, kept for the run report."""

    index: int
    phase: str
    tool: str
    input_preview: str
    is_error: bool
    duration_s: float


@dataclass(slots=True)
class DebugReport:
    """Everything a caller needs to judge the run."""

    stop_reason: StopReason
    resolved: bool
    steps_used: int
    phase: str
    root_cause: str = ""
    fix_summary: str = ""
    verification: str = ""
    files_changed: list[str] = field(default_factory=list)
    diff: str = ""
    transcript: list[StepRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    error: str = ""

    def to_json(self, indent: int = 2) -> str:
        payload = asdict(self)
        payload["stop_reason"] = self.stop_reason.value
        return json.dumps(payload, indent=indent)


class MessagesClient(Protocol):
    """Structural type for ``anthropic.Anthropic().messages``."""

    def create(self, **kwargs: Any) -> Any: ...


class AnthropicLike(Protocol):
    messages: MessagesClient


class DebugAgent:
    """Owns the conversation, the tool box, and the step budget."""

    def __init__(
        self,
        config: AgentConfig,
        toolbox: ToolBox | None = None,
        client: AnthropicLike | None = None,
    ) -> None:
        self.config = config
        self.toolbox = toolbox or ToolBox(config)
        self._client = client
        self.phase: Phase = Phase.REPRODUCE
        self.messages: list[MessageParam] = []
        self.transcript: list[StepRecord] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._consecutive_patches = 0

    # ------------------------------------------------------------- client ----
    @property
    def client(self) -> AnthropicLike:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError("install the SDK with `pip install anthropic`") from exc
            self._client = cast(AnthropicLike, anthropic.Anthropic(api_key=self.config.api_key))
        client = self._client
        assert client is not None
        return client

    # --------------------------------------------------------------- loop ----
    def run_debug_cycle(self, test_target: str, failure_hint: str | None = None) -> DebugReport:
        """Drive the loop until the failure is resolved or the budget is spent."""
        started = time.monotonic()
        self.messages = [{"role": "user", "content": initial_task_message(test_target, failure_hint)}]
        budget = self.config.model.max_steps
        stop = StopReason.MAX_STEPS
        report_payload: dict[str, Any] = {}
        error_text = ""
        nudges = 0
        steps_used = 0

        try:
            for step in range(1, budget + 1):
                steps_used = step
                try:
                    response = self._call_model()
                except Exception as exc:  # noqa: BLE001 - surface API failures as a report
                    log.exception("model call failed on step %s", step)
                    stop, error_text = StopReason.API_ERROR, f"{type(exc).__name__}: {exc}"
                    break

                self._account(response)
                assistant_blocks = _blocks(response)
                if not assistant_blocks:
                    # An empty content array is rejected by the API on the next
                    # turn, which would poison the rest of the conversation.
                    assistant_blocks = [{"type": "text", "text": "(no content returned)"}]
                self.messages.append({"role": "assistant", "content": assistant_blocks})
                self._log_text(step, assistant_blocks)

                tool_uses = [b for b in assistant_blocks if _block_type(b) == "tool_use"]
                if not tool_uses:
                    # The model answered in prose. Nudge it twice, then give up.
                    if nudges >= 2:
                        stop = StopReason.NO_TOOL_CALL
                        break
                    nudges += 1
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue by calling a tool. "
                                + PHASE_HINTS.get(self.phase.value, "")
                                + " When finished, call report_resolution."
                            ),
                        }
                    )
                    continue
                nudges = 0

                if self._consecutive_patches >= 3:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have applied three patches without a passing run. Stop patching. "
                                "Go back to PHASE 2, widen the search, and state what evidence "
                                "contradicts your current hypothesis before editing again."
                            ),
                        }
                    )
                    self._consecutive_patches = 0

                results: list[dict[str, Any]] = []
                terminated = False
                for block in tool_uses:
                    name = _block_name(block)
                    payload = _block_input(block)
                    if name == TERMINAL_TOOL:
                        report_payload = payload
                        stop = StopReason.RESOLVED if payload.get("resolved") else StopReason.UNRESOLVED
                        results.append(_tool_result(_block_id(block), "acknowledged", False))
                        terminated = True
                        continue
                    outcome = self._run_tool(step, name, payload)
                    results.append(_tool_result(_block_id(block), outcome.content, outcome.is_error))
                self.messages.append({"role": "user", "content": results})
                self._compact_history()
                if terminated:
                    break
            else:
                stop = StopReason.MAX_STEPS
        finally:
            diff = self.toolbox.diff()
            self.toolbox.close()

        self.phase = Phase.DONE
        return DebugReport(
            stop_reason=stop,
            resolved=bool(report_payload.get("resolved", False)) and stop is StopReason.RESOLVED,
            steps_used=steps_used,
            phase=self.phase.value,
            root_cause=str(report_payload.get("root_cause", "")),
            fix_summary=str(report_payload.get("fix_summary", "")),
            verification=str(report_payload.get("verification", "")),
            files_changed=[str(f) for f in report_payload.get("files_changed", [])]
            or [p.filepath for p in self.toolbox.patches],
            diff=diff,
            transcript=self.transcript,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            wall_time_s=time.monotonic() - started,
            error=error_text,
        )

    # --------------------------------------------------------------- model ---
    def _call_model(self) -> Any:
        cfg = self.config.model
        return self.client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=TOOL_SCHEMAS,
            messages=_wire_messages(self.messages),
        )

    def _account(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        # Cached prefix tokens are billed separately and excluded from
        # input_tokens, so a naive sum understates the real total every turn.
        for field_name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            self._input_tokens += int(getattr(usage, field_name, 0) or 0)
        self._output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    @staticmethod
    def _log_text(step: int, blocks: Sequence[Any]) -> None:
        for block in blocks:
            if _block_type(block) == "text":
                text = _block_text(block).strip()
                if text:
                    log.info("[step %s] %s", step, text[:600])

    # ---------------------------------------------------------------- tools --
    def _run_tool(self, step: int, name: str, payload: dict[str, Any]) -> ToolOutcome:
        began = time.monotonic()
        outcome = self.toolbox.dispatch(name, payload)
        elapsed = time.monotonic() - began
        self._advance_phase(name, outcome)
        self.transcript.append(
            StepRecord(
                index=step,
                phase=self.phase.value,
                tool=name,
                input_preview=_preview(payload),
                is_error=outcome.is_error,
                duration_s=elapsed,
            )
        )
        log.info("[step %s] %s -> %s (%.2fs)", step, name, "error" if outcome.is_error else "ok", elapsed)
        outcome.content = _truncate(outcome.content, self.config.model.max_tool_result_chars)
        return outcome

    def _advance_phase(self, name: str, outcome: ToolOutcome) -> None:
        if name == "run_test_suite":
            self.phase = Phase.VERIFY if outcome.metadata.get("passed") else Phase.LOCALIZE
            self._consecutive_patches = 0
        elif name in {"read_file", "search_codebase"}:
            self.phase = Phase.PATCH if self.phase is Phase.VERIFY else Phase.LOCALIZE
        elif name == "apply_unified_diff_or_patch" and not outcome.is_error:
            self.phase = Phase.VERIFY
            self._consecutive_patches += 1

    # -------------------------------------------------------------- history --
    def _compact_history(self) -> None:
        """Keep the transcript bounded: collapse old tool payloads into stubs.

        The first user turn (the task) and the most recent
        ``history_window_turns`` messages stay verbatim. Everything between is
        replaced by a one-line placeholder, which preserves the tool_use ->
        tool_result pairing the API requires while dropping the bulk.
        """
        window = self.config.model.history_window_turns
        if len(self.messages) <= window + 1:
            return
        for message in self.messages[1:-window]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            collapsed: list[Any] = []
            for block in content:
                btype = _block_type(block)
                if btype == "tool_result" and not _already_stub(block):
                    collapsed.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": _dict_get(block, "tool_use_id"),
                            "content": "[elided: superseded tool output]",
                            "is_error": bool(_dict_get(block, "is_error") or False),
                        }
                    )
                elif btype == "text":
                    collapsed.append({"type": "text", "text": _truncate(_block_text(block), 400)})
                else:
                    collapsed.append(block)
            message["content"] = collapsed


# --------------------------------------------------------------------------- #
# Block helpers: the SDK returns objects, our own stubs return dicts.
# --------------------------------------------------------------------------- #


def _dict_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _block_type(block: Any) -> str:
    return str(_dict_get(block, "type", ""))


def _block_name(block: Any) -> str:
    return str(_dict_get(block, "name", ""))


def _block_id(block: Any) -> str:
    return str(_dict_get(block, "id", ""))


def _block_text(block: Any) -> str:
    return str(_dict_get(block, "text", "") or "")


def _block_input(block: Any) -> dict[str, Any]:
    raw = _dict_get(block, "input", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _blocks(response: Any) -> list[Any]:
    content = _dict_get(response, "content", []) or []
    return list(content)


def _already_stub(block: Any) -> bool:
    content = _dict_get(block, "content")
    return isinstance(content, str) and content.startswith("[elided:")


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def _wire_messages(messages: Iterable[MessageParam]) -> list[MessageParam]:
    """Strip bookkeeping keys the API does not accept."""
    wire: list[MessageParam] = []
    for message in messages:
        clean = {k: v for k, v in message.items() if not k.startswith("_")}
        content = clean.get("content")
        if isinstance(content, list):
            clean["content"] = [_serialise(block) for block in content]
        wire.append(clean)
    return wire


def _serialise(block: Any) -> Any:
    if isinstance(block, dict):
        return block
    dumper = getattr(block, "model_dump", None)
    if callable(dumper):
        return {k: v for k, v in dumper(exclude_none=True).items() if k != "cache_control"}
    return block


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = limit // 2
    return f"{text[:keep]}\n... [{len(text) - limit} chars elided] ...\n{text[-keep:]}"


def _preview(payload: dict[str, Any], limit: int = 160) -> str:
    try:
        rendered = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        rendered = str(payload)
    return rendered[:limit]
