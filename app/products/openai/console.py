"""Console product layer — bridges Chat Completions / Responses API to console.x.ai.

Provides two main entry points:
  - _console_completions:   Chat Completions → console.x.ai/v1/responses → Chat Completions
  - _console_responses_dispatch: Responses API → console.x.ai/v1/responses (transparent proxy)
"""

import asyncio
import os
from typing import Any, AsyncGenerator

import orjson
from curl_cffi.const import CurlOpt

from app.platform.logging.logger import logger
from app.platform.config.snapshot import get_config
from app.platform.errors import UpstreamError
from app.platform.tokens import (
    estimate_prompt_tokens,
    estimate_tokens,
)
from app.dataplane.proxy.adapters.session import (
    ResettableSession,
    build_session_kwargs,
)
from app.dataplane.proxy import get_proxy_runtime
from app.dataplane.reverse.runtime.endpoint_table import CONSOLE_RESPONSES
from app.dataplane.reverse.protocol.xai_console import (
    build_console_payload,
    build_console_headers,
    classify_console_line,
    ConsoleStreamAdapter,
    messages_to_console_input,
    extract_instructions,
    convert_openai_tools_to_console,
    convert_openai_tool_choice,
    inject_web_search_tool,
)
from ._format import (
    make_response_id,
    make_stream_chunk,
    make_thinking_chunk,
    make_chat_response,
    build_usage,
    make_resp_id,
    make_resp_object,
    build_resp_usage,
    format_sse,
)


def _upstream_body_excerpt(exc: UpstreamError, *, limit: int = 240) -> str:
    details = getattr(exc, "details", {})
    if not isinstance(details, dict):
        return "-"
    body = str(details.get("body", "") or "").replace("\n", "\\n")
    return body[:limit] or "-"


def _transport_upstream_error(exc: BaseException, *, context: str) -> UpstreamError:
    if isinstance(exc, UpstreamError):
        return exc
    body = str(exc).replace("\n", "\\n")[:400]
    return UpstreamError(f"{context}: {exc}", status=0, body=body)


def _log_task_exception(task: asyncio.Task) -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.warning("bg task failed: task={} error={}", task.get_name(), exc)


# ---------------------------------------------------------------------------
# Low-level: raw SSE stream from console.x.ai
# ---------------------------------------------------------------------------

def _inject_env_proxy(session_kwargs: dict) -> dict:
    """Inject CONSOLE_PROXY_URL into session kwargs if no proxy set.

    Uses a dedicated env var so it doesn't interfere with tiktoken/requests
    which would otherwise pick up a global HTTPS_PROXY.
    """
    # Check if any proxy is already configured
    existing = (
        session_kwargs.get("proxy") or
        session_kwargs.get("proxies") or
        (session_kwargs.get("curl_options") or {}).get(CurlOpt.PROXY)
    )
    if existing:
        return session_kwargs

    proxy_url = os.environ.get("CONSOLE_PROXY_URL")
    if not proxy_url:
        return session_kwargs

    # Set both high-level and low-level proxy options
    kwargs = dict(session_kwargs)
    kwargs.setdefault("proxies", {"https": proxy_url})
    opts = dict(kwargs.get("curl_options") or {})
    opts.setdefault(CurlOpt.PROXY, proxy_url)
    kwargs["curl_options"] = opts
    return kwargs


async def _console_post_json(
    token: str,
    payload: dict[str, Any],
    timeout_s: float = 120.0,
) -> dict:
    """POST to console.x.ai/v1/responses and return the JSON response body."""
    proxy = await get_proxy_runtime()
    lease = await proxy.acquire()

    payload_bytes = orjson.dumps(payload)
    headers = build_console_headers(token)
    session_kwargs = _inject_env_proxy(build_session_kwargs(lease=lease))

    async with ResettableSession(**session_kwargs) as session:
        try:
            response = await session.post(
                CONSOLE_RESPONSES,
                headers=headers,
                data=payload_bytes,
                timeout=timeout_s,
                stream=False,
            )
        except Exception as exc:
            raise _transport_upstream_error(exc, context="Console transport failed") from exc

        if response.status_code not in (200, 402):
            try:
                body = response.content.decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
            raise UpstreamError(
                f"Console upstream returned {response.status_code}: {body}",
                status=response.status_code,
                body=body,
            )

        if response.status_code == 402:
            body = response.content.decode("utf-8", "replace")[:400]
            raise UpstreamError(
                "Console trial credits exhausted",
                status=402,
                body=body,
            )

        return orjson.loads(response.content)


async def _console_stream(
    token: str,
    payload: dict[str, Any],
    timeout_s: float = 120.0,
) -> AsyncGenerator[str, None]:
    """POST to console.x.ai/v1/responses and yield raw SSE lines (stream=True)."""
    proxy = await get_proxy_runtime()
    lease = await proxy.acquire()

    payload_bytes = orjson.dumps(payload)
    headers = build_console_headers(token)
    session_kwargs = _inject_env_proxy(build_session_kwargs(lease=lease))

    async with ResettableSession(**session_kwargs) as session:
        try:
            response = await session.post(
                CONSOLE_RESPONSES,
                headers=headers,
                data=payload_bytes,
                timeout=timeout_s,
                stream=True,
            )
        except Exception as exc:
            raise _transport_upstream_error(exc, context="Console transport failed") from exc

        if response.status_code not in (200, 402):
            try:
                body = response.content.decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
            raise UpstreamError(
                f"Console upstream returned {response.status_code}: {body}",
                status=response.status_code,
                body=body,
            )

        if response.status_code == 402:
            body = response.content.decode("utf-8", "replace")[:400]
            raise UpstreamError(
                "Console trial credits exhausted",
                status=402,
                body=body,
            )

        try:
            async for line in response.aiter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                yield line
        except Exception as exc:
            raise _transport_upstream_error(exc, context="Console stream read failed") from exc



def _drop_multi_agent_client_tools(
    upstream_model: str,
    tools: list[dict] | None,
    tool_choice: Any,
) -> tuple[list[dict] | None, Any]:
    """Drop client-side function tools rejected by multi-agent console models."""
    if "multi-agent" not in upstream_model or not tools:
        return tools, tool_choice

    kept: list[dict] = []
    dropped = 0
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            dropped += 1
            continue
        kept.append(tool)

    if not dropped:
        return tools, tool_choice

    logger.warning(
        "console multi-agent dropped unsupported client function tools: model={} dropped={}",
        upstream_model,
        dropped,
    )
    return kept, None

# ---------------------------------------------------------------------------
# Chat Completions → Console bridge
# ---------------------------------------------------------------------------

async def _console_completions(
    *,
    token: str,
    model: str,
    upstream_model: str,
    messages: list[dict],
    stream: bool | None,
    emit_think: bool | None = None,
    reasoning_effort_level: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    timeout_s: float = 120.0,
) -> dict | AsyncGenerator[str, None]:
    """Chat Completions through console.x.ai/v1/responses.

    Returns either a dict (non-streaming) or an async generator of SSE strings.
    """
    instructions = extract_instructions(messages)
    input_data = messages_to_console_input(messages)
    # Convert tools to console format (flatten nested function), then drop
    # function tools for multi-agent models because console rejects them.
    converted_tools = convert_openai_tools_to_console(tools)
    converted_tool_choice = convert_openai_tool_choice(tool_choice)
    converted_tools, converted_tool_choice = _drop_multi_agent_client_tools(
        upstream_model, converted_tools, converted_tool_choice)
    resolved_tools = inject_web_search_tool(converted_tools)

    # Only send reasoning.effort when explicitly provided by the caller.
    # Some console models (grok-4.20-reasoning etc.) reject the parameter.
    # NOTE: do NOT send {"effort": "none"} — the API rejects it as invalid.
    reasoning: dict | None = None
    if reasoning_effort_level in ("low", "medium", "high"):
        reasoning = {"effort": reasoning_effort_level}

    payload = build_console_payload(
        model=upstream_model,
        input_data=input_data,
        instructions=instructions,
        tools=resolved_tools,
        tool_choice=converted_tool_choice,
        stream=stream if stream is not None else True,
        temperature=temperature,
        top_p=top_p,
        reasoning=reasoning,
    )

    if stream:
        return _console_stream_completions(
            token=token,
            model=model,
            upstream_model=upstream_model,
            messages=messages,
            payload=payload,
            emit_think=emit_think,
            timeout_s=timeout_s,
        )

    # ---- Non-streaming (response is standard JSON, not SSE) ----
    resp = await _console_post_json(token, payload, timeout_s)

    full_text = ""
    full_think = ""
    sources: list[dict] = []
    _seen_urls: set[str] = set()
    for item in resp.get("output", []):
        if item.get("type") == "reasoning":
            for summary in item.get("summary", []):
                full_think += summary.get("text", "")
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    full_text += part.get("text", "")
                for ann in part.get("annotations", []):
                    url = ann.get("url_citation", {}).get("url", "") or ann.get("url", "")
                    if url and url not in _seen_urls:
                        _seen_urls.add(url)
                        src = {"url": url, "type": "web"}
                        title = ann.get("url_citation", {}).get("title", "") or ann.get("title", "")
                        if title:
                            src["title"] = title
                        sources.append(src)
    logger.debug("cc non-stream result: text_len={} think_len={} sources={}", len(full_text), len(full_think), len(sources))
    full_think = full_think or None
    sources = sources or None

    response = make_chat_response(
        model=model,
        content=full_text,
        reasoning_content=full_think or None,
        search_sources=sources or None,
    )
    pt = estimate_prompt_tokens(str(messages))
    ct = estimate_tokens(full_text)
    rt = estimate_tokens(full_think) if full_think else 0
    response["usage"] = build_usage(pt, ct + rt, reasoning_tokens=rt)
    return response


async def _console_stream_completions(
    token: str,
    model: str,
    upstream_model: str,
    messages: list[dict],
    payload: dict[str, Any],
    emit_think: bool | None,
    timeout_s: float,
) -> AsyncGenerator[str, None]:
    """Streaming path: console SSE → Chat Completions SSE chunks."""
    response_id = make_response_id()
    adapter = ConsoleStreamAdapter()
    think_buf: list[str] = []
    text_buf: list[str] = []
    thinking_started = False
    reasoning_started = False
    finished = False

    try:
        async for line in _console_stream(token, payload, timeout_s):
            if finished:
                break

            event_type, data = classify_console_line(line)
            if event_type == "done":
                finished = True
                break

            if event_type == "event":
                adapter._last_event = data
                continue

            ev = adapter.feed(event_type, data)
            if ev is None:
                continue

            if ev.kind == "thinking":
                if emit_think:
                    think_buf.append(ev.content)
                    chunk = make_thinking_chunk(response_id, model, ev.content)
                    yield f"data: {orjson.dumps(chunk).decode()}\n\n"

            elif ev.kind == "text":
                text_buf.append(ev.content)
                chunk = make_stream_chunk(response_id, model, ev.content)
                yield f"data: {orjson.dumps(chunk).decode()}\n\n"

            elif ev.kind == "done":
                finished = True

            elif ev.kind == "completed":
                finished = True
                sources = adapter.search_sources_list()

            elif ev.kind == "error":
                logger.warning("console stream error: {}", ev.content)
                finished = True

        # Final chunk
        full_text = "".join(text_buf)
        # Keep Chat Completions streaming chunks strict for clients such as
        # LobeChat. Citation URLs are already streamed as text by console.x.ai;
        # do not add non-standard root search_sources or delta.annotations.
        final_chunk = make_stream_chunk(
            response_id, model, "",
            is_final=True,
            finish_reason="stop",
        )
        pt = estimate_prompt_tokens(str(messages))
        ct = estimate_tokens(full_text)
        rt = estimate_tokens("".join(think_buf)) if think_buf else 0
        final_chunk["usage"] = build_usage(pt, ct + rt, reasoning_tokens=rt)
        yield f"data: {orjson.dumps(final_chunk).decode()}\n\n"
        yield "data: [DONE]\n\n"

    except UpstreamError:
        raise


# ---------------------------------------------------------------------------
# Responses API → Console proxy
# ---------------------------------------------------------------------------

def _extract_sources_from_resp(resp: dict) -> list[dict]:
    """Extract search_sources from a Responses API response object."""
    sources: list[dict] = []
    seen: set[str] = set()
    for item in resp.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            for ann in part.get("annotations", []):
                url = ann.get("url_citation", {}).get("url", "") or ann.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    src: dict = {"url": url, "type": "web"}
                    title = ann.get("url_citation", {}).get("title", "") or ann.get("title", "")
                    if title:
                        src["title"] = title
                    sources.append(src)
    return sources


async def _console_responses_dispatch(
    *,
    token: str,
    model: str,
    upstream_model: str,
    input_val: str | list,
    instructions: str | None,
    stream: bool,
    emit_think: bool,
    reasoning_effort_level: str | None = None,
    temperature: float,
    top_p: float,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    timeout_s: float = 120.0,
) -> dict | AsyncGenerator[str, None]:
    """Transparent proxy: Responses API → console.x.ai/v1/responses.

    For streaming: passes through SSE events, intercepts response.completed
    to inject search_sources.
    For non-streaming: passes through JSON, injects search_sources.
    """
    # Build input array if string
    input_data: list[dict]
    if isinstance(input_val, str):
        input_data = [{"role": "user", "content": [{"type": "input_text", "text": input_val}]}]
    else:
        input_data = input_val

    # Convert tools to console format, then drop function tools for
    # multi-agent models because console rejects client-side tools.
    converted_tools = convert_openai_tools_to_console(tools)
    converted_tool_choice = convert_openai_tool_choice(tool_choice)
    converted_tools, converted_tool_choice = _drop_multi_agent_client_tools(
        upstream_model, converted_tools, converted_tool_choice)
    resolved_tools = inject_web_search_tool(converted_tools)

    reasoning: dict | None = None
    if reasoning_effort_level in ("low", "medium", "high"):
        reasoning = {"effort": reasoning_effort_level}
    elif emit_think:
        reasoning = {"effort": "high"}

    payload = build_console_payload(
        model=upstream_model,
        input_data=input_data,
        instructions=instructions,
        tools=resolved_tools,
        tool_choice=converted_tool_choice,
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        reasoning=reasoning,
    )

    if stream:
        return _console_responses_stream(
            token=token,
            upstream_model=upstream_model,
            payload=payload,
            emit_think=emit_think,
            timeout_s=timeout_s,
        )

    # ---- Non-streaming (response is standard JSON, not SSE) ----
    resp = await _console_post_json(token, payload, timeout_s)

    # Build output + inject search_sources from annotations
    sources = _extract_sources_from_resp(resp)
    for item in resp.get("output", []):
        if item.get("type") == "message" and sources:
            item["search_sources"] = sources
    resp_id = resp.get("id", make_resp_id("resp"))
    resp_model = resp.get("model", model)
    resp_status = resp.get("status", "completed")
    resp_usage = resp.get("usage") or build_resp_usage(
        estimate_prompt_tokens(str(input_data)),
        estimate_tokens(""),
    )
    return make_resp_object(resp_id, resp_model, resp_status, resp.get("output", []), resp_usage)


async def _console_responses_stream(
    token: str,
    upstream_model: str,
    payload: dict[str, Any],
    emit_think: bool,
    timeout_s: float,
) -> AsyncGenerator[str, None]:
    """Streaming path: transparently proxy SSE events, inject search_sources."""
    response_id = make_resp_id("resp")
    adapter = ConsoleStreamAdapter()
    reasoning_id = make_resp_id("rs")
    message_id = make_resp_id("msg")
    reasoning_started = False
    message_started = False
    msg_idx = 0
    output_count = 0

    try:
        async for line in _console_stream(token, payload, timeout_s):
            event_type, data = classify_console_line(line)

            if event_type == "done":
                yield "data: [DONE]\n\n"
                return

            if event_type == "event":
                adapter._last_event = data
                continue

            if event_type != "data" or data is None:
                continue

            dtype = data.get("type", "")

            # Track output_item.added to count output positions
            if dtype == "response.output_item.added":
                item = data.get("item", data)
                item_type = item.get("type", "")
                if item_type == "reasoning" and emit_think:
                    reasoning_started = True
                if item_type == "message":
                    message_started = True
                    msg_idx = output_count

            # Track output_item.done
            if dtype == "response.output_item.done":
                output_count += 1

            # Feed through adapter FIRST to accumulate search_sources
            # (multi-agent annotations only available inside feed() on completed)
            ev = adapter.feed(event_type, data)
            if ev and ev.kind == "error":
                logger.warning("console responses stream error: {}", ev.content)

            # Forward all events as-is, just inject search_sources on completed
            if dtype == "response.completed":
                sources = adapter.search_sources_list()
                if sources:
                    resp = data.get("response", data)
                    output_items = resp.get("output", [])
                    for item in output_items:
                        if item.get("type") == "message":
                            item["search_sources"] = sources
                yield format_sse(dtype, data)
            else:
                yield format_sse(dtype, data)

    except UpstreamError:
        raise


def _resolve_reasoning_effort(emit_think: bool | None, effort_level: str | None = None) -> str | None:
    """Map emit_think flag to reasoning.effort value.

    If *effort_level* is provided (one of "low", "medium", "high"), use it directly
    when thinking is enabled. Otherwise fall back to "high"/"none".
    """
    if effort_level in ("low", "medium", "high"):
        return effort_level
    if emit_think is True:
        return "high"
    if emit_think is False:
        return "none"
    cfg = get_config()
    return "high" if cfg.get_bool("features.thinking", True) else "none"


__all__ = [
    "_console_completions",
    "_console_responses_dispatch",
    "_console_stream",
]
