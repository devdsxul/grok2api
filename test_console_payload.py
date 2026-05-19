#!/usr/bin/env python3
"""Test console.x.ai payload — run INSIDE Docker container on VPS.

Usage:
  # Copy to container and run
  docker cp test_console_payload.py grok2api:/app/
  docker exec grok2api python /app/test_console_payload.py sso=<token>

  # With specific model / effort
  docker exec grok2api python /app/test_console_payload.py sso=<token> model=grok-4.3 effort=high

  # With function tools
  docker exec grok2api python /app/test_console_payload.py sso=<token> model=grok-4.3 tools=1

Output: exact upstream request payload and full HTTP response body.
"""

import json
import os
import sys

# Use curl_cffi (already installed in container) for proper TLS fingerprint + proxy
from curl_cffi import requests

# ---- Build the exact same payload as the app ----

def build_console_payload(
    *,
    model: str,
    input_data: list[dict],
    instructions: str | None = None,
    tools: list[dict] | None = None,
    tool_choice=None,
    stream: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning: dict | None = None,
) -> dict:
    payload: dict = {"model": model, "input": input_data}
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
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return payload


def build_console_headers(token: str) -> dict[str, str]:
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
        "x-cluster": "https://us-east-1.api.x.ai",
        "User-Agent": _ua,
    }


def convert_openai_tools_to_console(tools: list[dict] | None) -> list[dict]:
    if not tools:
        return []
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
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


# ---- Parse CLI args ----
args = {}
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        args[k] = v

sso_token = args.get("sso", os.environ.get("SSO_TOKEN", ""))
if not sso_token:
    print("Usage: python test_console_payload.py sso=<token> [model=grok-4.3] [effort=high] [tools=1]")
    sys.exit(1)

model = args.get("model", "grok-4.3")
effort = args.get("effort")
test_with_tools = args.get("tools", "0") == "1"

# ---- Construct payload ----
input_data = [
    {
        "role": "user",
        "content": [{"type": "input_text", "text": "say hi in 3 words"}],
    }
]

reasoning = None
if effort:
    reasoning = {"effort": effort}

tools_arr = None
tool_choice = None
if test_with_tools:
    tools_arr = convert_openai_tools_to_console([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ])

payload = build_console_payload(
    model=model,
    input_data=input_data,
    tools=tools_arr,
    tool_choice=tool_choice,
    stream=False,  # non-streaming for simpler error inspection
    temperature=0.8,
    top_p=0.95,
    reasoning=reasoning,
)

# ---- Print what we're sending ----
print("=" * 70)
print(f"TEST: model={model} effort={effort or '(none)'} tools={test_with_tools}")
print("=" * 70)
print("\n>>> REQUEST PAYLOAD:")
print(json.dumps(payload, indent=2, ensure_ascii=False))

# ---- Send request ----
headers = build_console_headers(sso_token)

url = "https://console.x.ai/v1/responses"
print(f"\n>>> POST {url}")
print(f"    Cookie: sso={sso_token[:8]}...{sso_token[-4:]}")

# Try with proxy from CONSOLE_PROXY_URL env if set
proxy_url = os.environ.get("CONSOLE_PROXY_URL", "")
session_kwargs = {}
if proxy_url:
    session_kwargs["proxy"] = proxy_url
    print(f"    Proxy: {proxy_url}")

try:
    resp = requests.post(
        url,
        data=json.dumps(payload),
        headers=headers,
        impersonate="chrome120",
        timeout=30,
        **session_kwargs,
    )

    print(f"\n<<< STATUS: {resp.status_code}")
    body = resp.text
    try:
        parsed = json.loads(body)
        print(f"<<< BODY (pretty):")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(f"<<< BODY (raw):")
        print(body[:2000])

except Exception as e:
    print(f"\n<<< ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
