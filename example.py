#!/usr/bin/env python3
"""
example.py — Connect to the Codex LLM through the local subscription proxy.

What this does
--------------
1. Starts the Node.js OAuth bridge (auto-installs its npm deps on first run).
2. Builds a ``ChatGPTCodexOpenAI`` client (a drop-in for ``openai.OpenAI``)
   backed by your ChatGPT/Codex subscription.
3. Sends a one-shot prompt, then a multi-turn follow-up, then a tool-call demo.

Prerequisites
-------------
* Node.js >= 20 on PATH (the bridge runs on Node; ``npm install`` is automatic).
* A one-time browser login:   python codex_auth.py
* The ``openai`` Python package:   pip install -r requirements.txt

Run it
------
    python example.py

Choose the model with the CODEX_MODEL env var (default: gpt-5.4):
    CODEX_MODEL=gpt-5.4 python example.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

from codex_llm_proxy import (
    ChatGPTCodexOpenAI,
    CodexSubscriptionError,
    CodexSubscriptionNotAuthenticatedError,
    extract_output_text,
    get_shared_subscription_client,
)

MODEL = os.getenv("CODEX_MODEL", "gpt-5.4")


def build_client() -> ChatGPTCodexOpenAI:
    """Start the shared OAuth bridge and return a Codex-backed client.

    Exits with a friendly message if the user has not authenticated yet.
    """
    print(">> Starting the Codex OAuth bridge (Node.js)...")
    bridge = get_shared_subscription_client()

    # Fail fast with a clear message if there are no stored credentials.
    try:
        bridge.get_api_key()
    except CodexSubscriptionNotAuthenticatedError:
        print(
            "\nNo Codex credentials found.\n"
            "Authenticate once with:\n\n    python codex_auth.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f">> Authenticated. Using model: {MODEL}\n")
    return ChatGPTCodexOpenAI(get_token_fn=bridge.get_api_key)


def demo_one_shot(client: ChatGPTCodexOpenAI) -> "object":
    """Single-turn prompt. Returns the raw response (for chaining a follow-up)."""
    print("=" * 70)
    print("1) One-shot completion")
    print("=" * 70)

    response = client.responses.create(
        model=MODEL,
        instructions="You are a concise, helpful assistant.",
        input=[
            {
                "role": "user",
                "content": "In one sentence, what is an LLM proxy?",
            }
        ],
    )
    print("Assistant:", extract_output_text(response).strip(), "\n")
    return response


def demo_multi_turn(client: ChatGPTCodexOpenAI, previous_response_id: str) -> None:
    """Follow-up turn. ``previous_response_id`` lets the proxy reconstruct history."""
    print("=" * 70)
    print("2) Multi-turn follow-up (uses previous_response_id)")
    print("=" * 70)

    response = client.responses.create(
        model=MODEL,
        instructions="You are a concise, helpful assistant.",
        previous_response_id=previous_response_id,
        input=[
            {
                "role": "user",
                "content": "Now rewrite that as a haiku.",
            }
        ],
    )
    print("Assistant:", extract_output_text(response).strip(), "\n")


def demo_tool_call(client: ChatGPTCodexOpenAI) -> None:
    """Show that the Responses API tool-calling surface works through the proxy."""
    print("=" * 70)
    print("3) Tool calling")
    print("=" * 70)

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]

    response = client.responses.create(
        model=MODEL,
        instructions="Use the provided tools when they are relevant.",
        tools=tools,
        input=[{"role": "user", "content": "What's the weather in Paris?"}],
    )

    tool_calls = [
        item
        for item in (getattr(response, "output", None) or [])
        if (getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None))
        == "function_call"
    ]

    if not tool_calls:
        # The model answered directly instead of calling the tool.
        print("Assistant:", extract_output_text(response).strip(), "\n")
        return

    for call in tool_calls:
        name = getattr(call, "name", None) or call.get("name")
        raw_args = getattr(call, "arguments", None) or call.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args)
        except (TypeError, json.JSONDecodeError):
            parsed = raw_args
        print(f"Model requested tool: {name}({parsed})\n")


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "WARNING"),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        client = build_client()
        first = demo_one_shot(client)
        response_id = getattr(first, "id", None)
        if response_id:
            demo_multi_turn(client, response_id)
        demo_tool_call(client)
    except CodexSubscriptionError as exc:
        print(f"\nCodex proxy error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

    print("Done.")


if __name__ == "__main__":
    main()
