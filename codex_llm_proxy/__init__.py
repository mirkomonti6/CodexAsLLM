"""
codex_llm_proxy — use a ChatGPT/Codex subscription as a local LLM proxy.

This package starts a small Node.js OAuth bridge (``codex_oauth_bridge/``) that
manages the ChatGPT/Codex OAuth lifecycle, then exposes ``ChatGPTCodexOpenAI`` —
a drop-in for ``openai.OpenAI`` that talks to
``https://chatgpt.com/backend-api/codex/responses`` using the OAuth JWT.

Quick start:
    from codex_llm_proxy import (
        ChatGPTCodexOpenAI,
        get_shared_subscription_client,
        extract_output_text,
    )

    bridge = get_shared_subscription_client()      # starts the Node bridge
    client = ChatGPTCodexOpenAI(get_token_fn=bridge.get_api_key)
    resp = client.responses.create(
        model="gpt-5.4",
        instructions="You are a helpful assistant.",
        input=[{"role": "user", "content": "Hello!"}],
    )
    print(extract_output_text(resp))

First-time authentication (one-time browser login):
    python codex_auth.py
"""

from __future__ import annotations

from typing import Any

from codex_llm_proxy.subscription_client import (
    ChatGPTCodexOpenAI,
    CodexSubscriptionAuthError,
    CodexSubscriptionClient,
    CodexSubscriptionError,
    CodexSubscriptionNotAuthenticatedError,
    get_shared_subscription_client,
)

__all__ = [
    "ChatGPTCodexOpenAI",
    "CodexSubscriptionClient",
    "CodexSubscriptionError",
    "CodexSubscriptionAuthError",
    "CodexSubscriptionNotAuthenticatedError",
    "get_shared_subscription_client",
    "extract_output_text",
]


def extract_output_text(response: Any) -> str:
    """Concatenate all text from a Responses API result's ``message`` items.

    Works whether ``response.output`` items are pydantic models (normal SDK
    objects) or plain dicts (the proxy rebuilds some items from raw stream
    events, so both forms can appear). Returns an empty string if there is no
    assistant text (e.g. the turn produced only tool calls).
    """

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    parts: list[str] = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        content = _get(item, "content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            text = _get(block, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
