"""Console x.ai protocol layer — Responses API payload builder and SSE adapter.

Routes through console.x.ai/v1/responses instead of grok.com/rest/app-chat.
Uses standard SSE event: + data: format with 14+ event types.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import orjson

from app.platform.logging.logger import logger


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_console_payload(
    *,
    model: str,
    input_data: list[dict],
    instructions: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    stream: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning: dict | None = None,
    agent_count: int | None = None,
) -> dict[str, Any]:
    """Build a JSON payload for POST console.x.ai/v1/responses.

    Only sends fields that are non-None to avoid 422 from unknown/out-of-range values.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": input_data,
    }

    if stream:
        payload["stream"] = True
    if instructions:
        payload["instructions"] = instructions
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning:
        payload["reasoning"] = reasoning
    if agent_count is not None:
        payload["agent_count"] = agent_count
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    logger.debug(
        "console payload built: model={} input_len={} stream={}",
        model, len(input_data), stream,
    )
    return payload


def build_console_headers(
    token: str,
    *,
    x_cluster: str = "https://us-east-1.api.x.ai",
) -> dict[str, str]:
    """Build HTTP headers for console.x.ai requests."""
    _ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    )
    return {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Authorization": "Bearer anonymous",
        "Content-Type": "application/json",
        "Cookie": f"sso={token}",
        "Origin": "https://console.x.ai",
        "Referer": "https://console.x.ai/team/chat-playground",
        "x-cluster": x_cluster,
        "User-Agent": _ua,
        "Sec-CH-UA": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    }


# ---------------------------------------------------------------------------
# SSE line classifier
# ---------------------------------------------------------------------------

def classify_console_line(line: str) -> tuple[str, dict | None]:
    """Parse a console.v1.responses SSE line.

    Format is standard SSE:
      event: response.output_text.delta
      data: {"type": "...", ...}

    Returns (event_type, parsed_data) or ("skip", None) for non-data lines.
    Handles the two-line (event + data) convention by caching the last event name.
    """
    # Normalize bytes → str (curl_cffi aiter_lines may return bytes)
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")

    if not line or not line.strip():
        return "skip", None

    stripped = line.strip()

    # data: [DONE]
    if stripped == "data: [DONE]":
        return "done", None

    # event: <type>
    if stripped.startswith("event:"):
        return "event", stripped[6:].strip()

    # data: {json}
    if stripped.startswith("data:"):
        raw = stripped[5:].strip()
        if not raw:
            return "skip", None
        try:
            data = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            logger.warning("console sse json parse error: line={!r} error={}", stripped[:200], exc)
            return "skip", None
        return "data", data

    # Unknown / comment lines
    if stripped.startswith(":"):
        return "skip", None

    logger.debug("console sse unhandled line: {!r}", stripped[:200])
    return "skip", None


# ---------------------------------------------------------------------------
# Console stream adapter
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConsoleFrameEvent:
    """A single parsed event from the console Responses API stream."""
    kind: str = ""
    content: str = ""
    annotation_data: dict | None = None
    reasoning_text: str = ""
    tool_call_data: dict | None = None
    search_sources: list[dict] | None = None
    raw: dict | None = None


@dataclass(slots=True)
class ConsoleStreamAdapter:
    """Accumulate console Responses API SSE events into structured output.

    Tracks text buffer, thinking buffer, annotations, search sources,
    and tool calls across multiple SSE events in a single response stream.
    """

    text_buf: list[str] = field(default_factory=list)
    thinking_buf: list[str] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    search_sources: list[dict] = field(default_factory=list)
    _seen_urls: set[str] = field(default_factory=set)
    tool_calls: list[dict] = field(default_factory=list)
    finished: bool = False
    _last_event: str = ""

    def feed(self, event_type: str, data: dict | None) -> ConsoleFrameEvent | None:
        """Feed one parsed SSE event. Returns a ConsoleFrameEvent or None."""
        if event_type == "done":
            self.finished = True
            return ConsoleFrameEvent(kind="done")

        if event_type != "data" or data is None:
            return None

        ev = ConsoleFrameEvent(raw=data)
        dtype = data.get("type", "")

        # ---- response.output_text.delta ----
        if dtype == "response.output_text.delta":
            delta = data.get("delta", "")
            self.text_buf.append(delta)
            ev.kind = "text"
            ev.content = delta
            return ev

        # ---- response.reasoning_summary_text.delta ----
        if dtype == "response.reasoning_summary_text.delta":
            delta = data.get("delta", "")
            self.thinking_buf.append(delta)
            ev.kind = "thinking"
            ev.content = delta
            ev.reasoning_text = delta
            return ev

        # ---- response.output_text.annotation.added ----
        if dtype == "response.output_text.annotation.added":
            ann = data.get("annotation", {})
            if ann:
                self.annotations.append(ann)
                ev.kind = "annotation"
                ev.annotation_data = ann

                # Extract URL for search_sources (handle multi-agent start_index==0)
                url_citation = ann.get("url_citation", {})
                url = url_citation.get("url", "") or ann.get("url", "")
                title = url_citation.get("title", "") or ann.get("title", "")
                if url and url not in self._seen_urls:
                    self._seen_urls.add(url)
                    src: dict[str, str] = {"url": url, "type": "web"}
                    if title:
                        src["title"] = title
                    self.search_sources.append(src)
            return ev

        # ---- response.completed ----
        if dtype == "response.completed":
            self.finished = True
            ev.kind = "completed"
            # Extract additional search_sources from the completed response
            resp = data.get("response", data)
            self._extract_final_search_sources(resp)
            if self.search_sources:
                ev.search_sources = list(self.search_sources)
            return ev

        # ---- response.incomplete / response.failed ----
        if dtype in ("response.incomplete", "response.failed"):
            self.finished = True
            ev.kind = "error"
            ev.content = data.get("error", {}).get("message", "") if isinstance(data.get("error"), dict) else ""
            return ev

        # ---- error (top-level) ----
        if dtype == "error":
            self.finished = True
            ev.kind = "error"
            ev.content = data.get("message", "")
            return ev

        # ---- function_call arguments delta ----
        if dtype == "response.function_call_arguments.delta":
            ev.kind = "tool_call"
            ev.tool_call_data = {
                "item_id": data.get("item_id", ""),
                "name": data.get("name", ""),
                "delta": data.get("delta", ""),
            }
            return ev

        # ---- function_call argument done ----
        if dtype == "response.function_call_arguments.done":
            ev.kind = "tool_call_done"
            ev.tool_call_data = {
                "item_id": data.get("item_id", ""),
                "name": data.get("name", ""),
                "arguments": data.get("arguments", ""),
            }
            return ev

        # ---- output_item events (signal only, content comes in sub-events) ----
        if dtype in (
            "response.created",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.done",
            "response.output_text.done",
            "response.function_call_arguments.done",
        ):
            ev.kind = "signal"
            return ev

        # Unknown — log and skip
        logger.debug("console sse unhandled event type: {}", dtype)
        return None

    def _extract_final_search_sources(self, response: dict) -> None:
        """Extract search_sources from response.output annotations (post-stream)."""
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                for ann in part.get("annotations", []):
                    url_citation = ann.get("url_citation", {})
                    url = url_citation.get("url", "") or ann.get("url", "")
                    if url and url not in self._seen_urls:
                        self._seen_urls.add(url)
                        src: dict[str, str] = {"url": url, "type": "web"}
                        title = url_citation.get("title", "") or ann.get("title", "")
                        if title:
                            src["title"] = title
                        self.search_sources.append(src)

    def search_sources_list(self) -> list[dict]:
        """Return deduplicated search sources."""
        return list(self.search_sources)

    def annotations_list(self) -> list[dict]:
        """Return collected annotations."""
        return list(self.annotations)

    @property
    def image_urls(self) -> list[tuple[str, str]]:
        """Console model doesn't use image_urls pattern (images handled natively)."""
        return []


# ---------------------------------------------------------------------------
# Input conversion helpers
# ---------------------------------------------------------------------------

def messages_to_console_input(messages: list[dict]) -> list[dict]:
    """Convert Chat Completions messages to Responses API input array.

    Handles system → instructions split, multimodal content,
    tool calls, and tool results.
    """
    input_items: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # System messages become instructions, not input items
            continue

        item: dict[str, Any] = {"role": role}

        if role == "assistant" and "tool_calls" in msg:
            # Tool calls → function_call items
            for tc in msg["tool_calls"]:
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                })
            # Also include text content if present
            if content:
                if isinstance(content, str):
                    item["content"] = [{"type": "input_text", "text": content}]
                elif isinstance(content, list):
                    item["content"] = _normalize_content_parts(content)
            if item.get("content"):
                input_items.append(item)
        elif role == "tool":
            # Tool result → function_call_output
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": str(content) if content else "",
            })
        else:
            # User / assistant text / multimodal
            if isinstance(content, str):
                item["content"] = [{"type": "input_text", "text": content}]
            elif isinstance(content, list):
                item["content"] = _normalize_content_parts(content)
            input_items.append(item)

    return input_items


def extract_instructions(messages: list[dict]) -> str | None:
    """Extract and concatenate system messages into a single instructions string."""
    parts = [msg["content"] for msg in messages if msg.get("role") == "system" and msg.get("content")]
    return "\n".join(parts) if parts else None


def _normalize_content_parts(parts: list[dict]) -> list[dict]:
    """Normalize Chat Completions content parts to Responses API format."""
    result: list[dict] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")

        if ptype == "text":
            result.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = part.get("image_url", {})
            if isinstance(url, dict):
                url_val = url.get("url", "")
            else:
                url_val = str(url)
            if url_val:
                result.append({
                    "type": "input_image",
                    "image_url": url_val,
                })
        elif ptype == "input_text":
            result.append(part)
        elif ptype == "input_image":
            result.append(part)
        else:
            # Pass through unknown types
            result.append(part)
    return result


def convert_openai_tools_to_console(tools: list[dict] | None) -> list[dict]:
    """Convert OpenAI Chat Completions tools -> console (Responses API) tools.

    OpenAI Chat:  {"type": "function", "function": {"name", "description", "parameters"}}
    Console API:  {"type": "function", "name", "description", "parameters"}
    OpenAI web_search_preview is normalized to console web_search.
    """
    if not tools:
        return []
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        tool_type = t.get("type")
        if tool_type in {"web_search_preview", "web_search_preview_2025_03_11"}:
            normalized = dict(t)
            normalized["type"] = "web_search"
            out.append(normalized)
            continue
        if tool_type != "function":
            out.append(dict(t))
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else None
        if fn:
            out.append({
                "type": "function",
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {},
            })
        else:
            out.append(dict(t))
    return out


def convert_openai_tool_choice(tool_choice: Any) -> Any:
    """Convert OpenAI tool_choice -> console tool_choice.

    OpenAI:  "none" | "auto" | "required" | {"type":"function","function":{"name":"x"}}
    Console: "none" | "auto" | "required" | {"type":"function","name":"x"}
    """
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else None
        if fn:
            return {"type": "function", "name": fn.get("name") or ""}
        return dict(tool_choice)
    return tool_choice


def inject_web_search_tool(tools: list[dict] | None) -> list[dict]:
    """Ensure a web_search tool is present in the tools list."""
    existing = list(tools or [])
    for t in existing:
        if isinstance(t, dict) and t.get("type") == "web_search":
            return existing
    existing.append({"type": "web_search"})
    return existing


__all__ = [
    "build_console_payload",
    "build_console_headers",
    "classify_console_line",
    "ConsoleFrameEvent",
    "ConsoleStreamAdapter",
    "messages_to_console_input",
    "extract_instructions",
    "convert_openai_tools_to_console",
    "convert_openai_tool_choice",
    "inject_web_search_tool",
]
