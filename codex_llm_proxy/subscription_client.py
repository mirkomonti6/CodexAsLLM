"""
subscription_client.py

Manages the Node.js OAuth bridge subprocess and exposes ``ChatGPTCodexOpenAI`` —
a drop-in for ``openai.OpenAI`` that calls ``chatgpt.com/backend-api/codex/responses``
directly with the OAuth JWT, giving standard Responses API behavior
(parallel tool calls, ``resp_...`` IDs, fast) without the codex app-server subprocess.

In other words: this turns a ChatGPT/Codex *subscription* into a usable LLM
endpoint — a local "Codex as LLM proxy".

Usage:
    from codex_llm_proxy import (
        ChatGPTCodexOpenAI,
        get_shared_subscription_client,
    )

    bridge = get_shared_subscription_client()          # starts the Node bridge
    client = ChatGPTCodexOpenAI(get_token_fn=bridge.get_api_key)
    resp = client.responses.create(
        model="gpt-5.4",
        instructions="You are a helpful assistant.",
        input=[{"role": "user", "content": "Hello!"}],
    )

First-time auth (one-time browser login):
    python codex_auth.py
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CodexSubscriptionError(RuntimeError):
    """Base exception for Codex subscription client errors."""


class CodexSubscriptionAuthError(CodexSubscriptionError):
    """OAuth flow failed or was not started."""


class CodexSubscriptionNotAuthenticatedError(CodexSubscriptionError):
    """No valid credentials on disk. Run: python codex_auth.py"""


# ---------------------------------------------------------------------------
# Direct ChatGPT Codex Responses API client
# Endpoint: https://chatgpt.com/backend-api/codex/responses
# This is the same Responses API format (parallel tool calls, resp_... IDs)
# used by OpenClaw's openai-codex-responses transport.
# ---------------------------------------------------------------------------

_CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# openai SDK adds these internal-only fields during parsing; the chatgpt.com
# endpoint rejects them as unknown parameters when sent back as input history.
_SDK_INTERNAL_FIELDS = {"parsed_arguments"}


def _clean_for_api(obj: Any) -> Any:
    """Recursively strip SDK-internal fields that the chatgpt.com endpoint rejects."""
    if isinstance(obj, dict):
        return {
            k: _clean_for_api(v)
            for k, v in obj.items()
            if k not in _SDK_INTERNAL_FIELDS
        }
    if isinstance(obj, list):
        return [_clean_for_api(i) for i in obj]
    return obj


def _extract_account_id(jwt_token: str) -> str:
    """Extract chatgpt_account_id from the OAuth JWT payload."""
    parts = jwt_token.split(".")
    if len(parts) != 3:
        raise CodexSubscriptionError("Invalid JWT token format")
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        account_id = payload.get("https://api.openai.com/auth", {}).get(
            "chatgpt_account_id"
        )
        if not account_id:
            raise CodexSubscriptionError("No chatgpt_account_id in JWT payload")
        return account_id
    except (json.JSONDecodeError, Exception) as exc:
        raise CodexSubscriptionError(
            f"Failed to extract account ID from token: {exc}"
        ) from exc


def _prepare_codex_stream_kwargs(kwargs: dict) -> dict:
    """Apply the chatgpt.com codex endpoint's hard constraints to create() kwargs.

    Mutates ``kwargs`` in place (and returns it for convenience):
    - ``store`` must be False.
    - Streaming is managed internally, so ``stream`` is dropped.
    - ``max_output_tokens`` and ``truncation`` are not supported by the endpoint
      and are dropped (forwarding them yields HTTP 400 "Unsupported parameter").
    - ``instructions`` is required; a default is supplied when absent.
    """
    kwargs["store"] = False
    kwargs.pop("stream", None)  # streaming managed internally
    kwargs.pop("max_output_tokens", None)  # not supported by chatgpt.com endpoint
    kwargs.pop("truncation", None)  # not supported by chatgpt.com endpoint
    # chatgpt.com/backend-api/codex/responses requires `instructions`
    if "instructions" not in kwargs:
        kwargs["instructions"] = "You are a helpful assistant."
    return kwargs


class _ChatGPTCodexResponsesProxy:
    """
    Calls chatgpt.com/backend-api/codex/responses via openai.OpenAI with the
    required custom headers. Fetches a fresh token on every call.

    Handles two endpoint constraints transparently:
    - store=False is required
    - stream=True is required (handled via low-level create(stream=True); the
      final response is captured from the response.completed event and its
      output assembled from the streamed response.output_item.done events)
    - previous_response_id is NOT supported: conversation history is tracked
      locally and reconstructed into the full input on each turn.
    """

    def __init__(self, get_token_fn, timeout: float = 120.0):
        self._get_token = get_token_fn
        self._timeout = timeout
        self._lock = threading.Lock()
        # Maps response_id → flattened input list for the *next* turn
        # (full prior input + serialised output items from that response)
        self._history: "OrderedDict[str, list]" = OrderedDict()
        self._history_max = int(os.getenv("CODEX_HISTORY_MAX", "1000"))

    def create(self, **kwargs) -> Any:
        # No global lock: the chatgpt.com call is parallel-safe and a fresh
        # openai.OpenAI() is built per call. Only the final _history write
        # is guarded — see the with-lock block at the end.
        import openai

        # Per-call HTTP deadline (e.g. context reranker). Not sent to the API.
        call_timeout = float(kwargs.pop("request_timeout", self._timeout))

        token = self._get_token()
        account_id = _extract_account_id(token)
        client = openai.OpenAI(
            api_key=token,
            base_url=_CHATGPT_CODEX_BASE_URL,
            default_headers={
                "chatgpt-account-id": account_id,
                "originator": "pi",
                "OpenAI-Beta": "responses=experimental",
            },
            timeout=call_timeout,
        )
        _prepare_codex_stream_kwargs(kwargs)

        # previous_response_id is not supported by chatgpt.com endpoint.
        # Reconstruct the full conversation history locally instead.
        prev_id = kwargs.pop("previous_response_id", None)
        raw_input = kwargs.get("input", [])
        current_input: list = (
            [{"role": "user", "content": raw_input}]
            if isinstance(raw_input, str)
            else list(raw_input)
        )
        if prev_id and prev_id in self._history:
            kwargs["input"] = self._history[prev_id] + current_input
        else:
            kwargs["input"] = current_input

        # Chain validator: when previous_response_id is set, walk the
        # reconstructed input and verify every prior function_call has a
        # matching function_call_output in the chain. A gap here triggers
        # a 400 "No tool output found for function call X" from the API.
        # We log a WARNING so the cascade is debuggable *before* the API
        # rejects the call; the call still proceeds (logging only).
        if prev_id and prev_id in self._history:
            try:
                _chain = kwargs.get("input") or []
                _fc_ids: list = []
                _fco_ids: set = set()
                for _it in _chain:
                    if not isinstance(_it, dict):
                        continue
                    _t = _it.get("type")
                    if _t == "function_call":
                        _cid = _it.get("call_id")
                        if _cid:
                            _fc_ids.append(_cid)
                    elif _t == "function_call_output":
                        _cid = _it.get("call_id")
                        if _cid:
                            _fco_ids.add(_cid)
                _missing = [cid for cid in _fc_ids if cid not in _fco_ids]
                if _missing:
                    logger.warning(
                        "[codex_subscription] chain_gap_detected: "
                        "previous_response_id=%s missing function_call_output "
                        "for call_ids=%s (will likely 400)",
                        prev_id,
                        _missing[:5],
                    )
            except Exception:
                # Validation is best-effort; never block the call.
                pass

        _inst = kwargs.get("instructions") or ""
        if not isinstance(_inst, str):
            _inst = str(_inst)
        _tl = kwargs.get("tools")
        _n_tools = len(_tl) if isinstance(_tl, list) else 0
        _inp_list = kwargs.get("input")
        _n_in = len(_inp_list) if isinstance(_inp_list, list) else 1
        _model = kwargs.get("model", "")
        logger.info(
            "[codex_subscription] stream_request: model=%r timeout_s=%.1f "
            "instructions_chars=%d input_messages=%d n_tools=%d "
            "reconstructed_history=%s",
            _model,
            call_timeout,
            len(_inst),
            _n_in,
            _n_tools,
            "yes" if (prev_id and prev_id in self._history) else "no",
        )
        _t_stream = time.perf_counter()
        # Assemble the response from raw stream events ourselves.
        #
        # We deliberately avoid the high-level client.responses.stream() helper:
        # on openai>=2.x it eagerly runs parse_response() on the
        # `response.completed` event, which does `for output in response.output`.
        # The chatgpt.com/backend-api/codex endpoint sends that completed event
        # with output=None, so the helper raises "'NoneType' object is not
        # iterable" *during stream consumption* — before we can read the streamed
        # items or call get_final_response(). (The original workaround here was
        # written for openai SDK 1.93, before parse_response ran during accumulation.)
        #
        # The low-level create(stream=True) yields raw events constructed via
        # construct_type() (unvalidated), so output=None is tolerated. We capture
        # the final Response from the completed event and rebuild its output from
        # the response.output_item.done events below (see the patch block further down).
        _collected_items: dict = {}
        _stream_event_count = 0
        _completed_response = None
        with client.responses.create(**kwargs, stream=True) as stream:
            for _ev in stream:
                _stream_event_count += 1
                _ev_type_str = getattr(_ev, "type", "") or ""
                if _ev_type_str == "response.output_item.done":
                    _item = getattr(_ev, "item", None)
                    _idx = getattr(_ev, "output_index", None)
                    if _item is not None and _idx is not None:
                        _raw = (
                            _item.model_dump(mode="json")
                            if hasattr(_item, "model_dump")
                            else None
                        )
                        if _raw:
                            _collected_items[_idx] = _raw
                elif _ev_type_str == "response.completed":
                    _completed_response = getattr(_ev, "response", None)
        response = _completed_response
        if response is None:
            raise CodexSubscriptionError(
                "codex stream ended without a response.completed event "
                f"(events={_stream_event_count}, collected_items={len(_collected_items)})"
            )
        _elapsed_s = time.perf_counter() - _t_stream
        _out = getattr(response, "output", None)
        _out_n = len(_out) if isinstance(_out, list) else 0
        _rid = getattr(response, "id", "") or ""
        _st = getattr(response, "status", "") or ""
        logger.info(
            "[codex_subscription] stream_done: elapsed_s=%.2f model=%r response_id=%s… status=%r "
            "output_items=%d stream_events=%d",
            _elapsed_s,
            _model,
            str(_rid)[:20],
            _st,
            _out_n,
            _stream_event_count,
        )

        if not getattr(response, "output", None) and _collected_items:
            from openai.types.responses.parsed_response import (
                ParsedResponseFunctionToolCall,
                ParsedResponseOutputMessage,
            )

            _ParsedMsgNone = ParsedResponseOutputMessage[type(None)]
            _patched = [_collected_items[k] for k in sorted(_collected_items)]
            _rebuilt: list = []
            for _item_dict in _patched:
                try:
                    if _item_dict.get("type") == "function_call":
                        _rebuilt.append(
                            ParsedResponseFunctionToolCall.model_validate(_item_dict)
                        )
                    elif _item_dict.get("type") == "message":
                        _rebuilt.append(_ParsedMsgNone.model_validate(_item_dict))
                    else:
                        _rebuilt.append(_item_dict)
                except Exception:
                    _rebuilt.append(_item_dict)
            response.output = _rebuilt
            logger.info(
                "[codex_subscription] Patched %d output items from stream events (SDK assembly bug workaround)",
                len(_patched),
            )

            # Some failures arrive as a Cloudflare HTML challenge delivered as an
            # output "message" body instead of raising. Detect it and fail fast so
            # we never pass the raw HTML downstream as if it were a model response.
            _html_hit = False
            for _d in _patched:
                if not isinstance(_d, dict) or _d.get("type") != "message":
                    continue
                _content = _d.get("content")
                _text = ""
                if isinstance(_content, str):
                    _text = _content
                elif isinstance(_content, list):
                    for _blk in _content:
                        if isinstance(_blk, dict):
                            _t = _blk.get("text") or ""
                            if isinstance(_t, str):
                                _text += _t
                if "<html" in _text.lower() or "challenge-error-text" in _text:
                    _html_hit = True
                    break

            if _html_hit:
                raise CodexSubscriptionAuthError(
                    "ChatGPT Codex endpoint returned a Cloudflare challenge (HTML). "
                    "This backend cannot proceed until the challenge is resolved; "
                    "try re-auth via `python codex_auth.py`, "
                    "or switch to a different LLM backend."
                )

        # Persist history so the next turn can use previous_response_id
        if getattr(response, "id", None):
            output_items = []
            for item in getattr(response, "output", []):
                raw = (
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else (item if isinstance(item, dict) else None)
                )
                if raw is not None:
                    output_items.append(_clean_for_api(raw))
            with self._lock:
                self._history[response.id] = list(kwargs["input"]) + output_items
                while len(self._history) > self._history_max:
                    self._history.popitem(last=False)

        return response


class ChatGPTCodexOpenAI:
    """
    Drop-in for openai.OpenAI backed by chatgpt.com/backend-api/codex/responses.
    Provides standard Responses API behavior: parallel tool calls, resp_... IDs, fast.
    Token is refreshed from the Node bridge on every call.
    """

    def __init__(self, get_token_fn, timeout: float = 120.0):
        self.responses = _ChatGPTCodexResponsesProxy(get_token_fn, timeout)


# ---------------------------------------------------------------------------
# CodexSubscriptionClient
# ---------------------------------------------------------------------------

# The Node bridge lives at <project_root>/codex_oauth_bridge by default. Allow an
# explicit override via CODEX_OAUTH_BRIDGE_DIR so the package keeps working when
# installed outside the project tree.
_BRIDGE_DIR = Path(
    os.getenv("CODEX_OAUTH_BRIDGE_DIR")
    or (Path(__file__).resolve().parent.parent / "codex_oauth_bridge")
)
_NPM_SENTINEL = _BRIDGE_DIR / "node_modules" / "@mariozechner" / "pi-ai" / "README.md"


class CodexSubscriptionClient:
    """
    Manages the lifecycle of the Node.js OAuth bridge and exposes helpers
    for the auth flow and token retrieval.
    """

    TOKEN_PATH = Path.home() / ".config" / "codex_oauth_tokens.json"
    BRIDGE_DIR = _BRIDGE_DIR

    def __init__(
        self,
        *,
        port: int = int(os.getenv("CODEX_OAUTH_BRIDGE_PORT", "7777")),
        startup_timeout: float = 30.0,
        request_timeout: float = 30.0,
    ):
        self._port = port
        self._base_url = f"http://127.0.0.1:{port}"
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        atexit.register(self.stop)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Install npm deps (if needed) and start the Node bridge."""
        with self._lock:
            self._install_deps_if_needed()
            self._spawn()

    def stop(self) -> None:
        """Terminate the Node bridge subprocess."""
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None

    def _ensure_running(self) -> None:
        """Restart bridge if it has crashed. Must be called under self._lock."""
        if self._proc is None or self._proc.poll() is not None:
            logger.info("[codex_subscription] Bridge not running, (re)starting...")
            self._spawn()

    def _spawn(self) -> None:
        """Start the Node.js bridge. Assumes self._lock is held."""
        node_bin = shutil.which("node")
        if not node_bin:
            raise CodexSubscriptionError(
                "Node.js is required for codex_subscription backend. "
                "Install Node.js >= 20 from https://nodejs.org"
            )

        env = {**os.environ, "CODEX_OAUTH_BRIDGE_PORT": str(self._port)}
        self._proc = subprocess.Popen(
            [node_bin, "server.js"],
            cwd=str(self.BRIDGE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Forward stderr in background thread so it appears in the Python log
        stderr_thread = threading.Thread(
            target=self._stderr_loop,
            args=(self._proc,),
            daemon=True,
            name="codex-bridge-stderr",
        )
        stderr_thread.start()

        try:
            self._wait_for_health(time.monotonic() + self._startup_timeout)
        except Exception as exc:
            self._proc.terminate()
            self._proc = None
            raise CodexSubscriptionError(
                f"Codex OAuth bridge failed to start: {exc}"
            ) from exc

    def _stderr_loop(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stderr:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    logger.debug("[codex-bridge] %s", decoded)
                    if os.getenv("CODEX_OAUTH_DEBUG", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }:
                        try:
                            sys.stderr.write(decoded + "\n")
                            sys.stderr.flush()
                        except Exception:
                            pass
        except Exception:
            pass

    def _wait_for_health(self, deadline: float) -> None:
        delay = 0.2
        while True:
            try:
                resp = self._raw_get("/health", timeout=3.0)
                if resp.get("status") == "ok":
                    return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise CodexSubscriptionError(
                    f"Bridge did not become healthy within {self._startup_timeout}s"
                )
            time.sleep(min(delay, deadline - time.monotonic()))
            delay = min(delay * 1.5, 2.0)

    def _install_deps_if_needed(self) -> None:
        if _NPM_SENTINEL.exists():
            return
        npm_bin = shutil.which("npm")
        if not npm_bin:
            raise CodexSubscriptionError(
                "npm is required to install the Codex OAuth bridge dependencies. "
                "Install Node.js >= 20 (which includes npm) from https://nodejs.org"
            )
        logger.info(
            "[codex_subscription] Running npm install in %s ...", self.BRIDGE_DIR
        )
        result = subprocess.run(
            [npm_bin, "install"],
            cwd=str(self.BRIDGE_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CodexSubscriptionError(f"npm install failed:\n{result.stderr}")
        logger.info("[codex_subscription] npm install complete.")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _raw_get(self, path: str, timeout: Optional[float] = None) -> dict:
        timeout = timeout or self._request_timeout
        req = urllib.request.Request(self._base_url + path)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        with self._lock:
            self._ensure_running()
        try:
            return self._raw_get(path)
        except urllib.error.HTTPError as exc:
            body = {}
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass
            if exc.code == 401:
                raise CodexSubscriptionNotAuthenticatedError(
                    "No Codex OAuth credentials found. "
                    "Run: python codex_auth.py"
                ) from exc
            raise CodexSubscriptionError(
                f"Bridge returned HTTP {exc.code}: {body.get('error', exc.reason)}"
            ) from exc

    def _post(self, path: str, body: dict, timeout: Optional[float] = None) -> dict:
        with self._lock:
            self._ensure_running()
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        effective_timeout = timeout if timeout is not None else self._request_timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_resp = {}
            try:
                body_resp = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass
            raise CodexSubscriptionError(
                f"Bridge returned HTTP {exc.code}: {body_resp.get('error', exc.reason)}"
            ) from exc

    # ------------------------------------------------------------------
    # Public auth API
    # ------------------------------------------------------------------

    def start_auth(self) -> str:
        """
        Trigger the browser OAuth flow.
        Returns the authorize_url that the user must open.
        Raises CodexSubscriptionAuthError if the bridge reports an error.
        """
        # Bridge waits up to 20s for PKCE setup + OAuth server start before responding
        result = self._post("/start-auth", {}, timeout=30.0)
        if "error" in result:
            err = result["error"]
            if err == "port_1455_in_use":
                raise CodexSubscriptionAuthError(
                    "Port 1455 is required for the OAuth callback (hard-coded by the library). "
                    "Ensure no other process is using port 1455 and try again."
                )
            raise CodexSubscriptionAuthError(f"start_auth failed: {err}")
        url = result.get("authorize_url")
        if not url:
            raise CodexSubscriptionAuthError(
                "Bridge did not return an authorize_url. "
                "Check the bridge logs for details."
            )
        return url

    def poll_auth_status(self) -> dict:
        """Returns { status: "pending"|"complete"|"failed", error?: str }"""
        return self._get("/auth-status")

    def wait_for_auth(self, timeout: float = 300.0) -> None:
        """
        Block until OAuth flow completes or timeout is reached.
        Raises CodexSubscriptionAuthError on failure.
        """
        deadline = time.monotonic() + timeout
        while True:
            status_resp = self.poll_auth_status()
            status = status_resp.get("status", "pending")
            if status == "complete":
                return
            if status == "failed":
                raise CodexSubscriptionAuthError(
                    f"OAuth flow failed: {status_resp.get('error', 'unknown')}"
                )
            if time.monotonic() >= deadline:
                raise CodexSubscriptionAuthError(
                    f"Timed out waiting for OAuth flow after {timeout}s"
                )
            time.sleep(1.5)

    def get_api_key(self) -> str:
        """
        Return a valid (auto-refreshed) access token.
        Raises CodexSubscriptionNotAuthenticatedError if no credentials exist.
        """
        result = self._get("/get-api-key")
        api_key = result.get("api_key")
        if not api_key:
            raise CodexSubscriptionNotAuthenticatedError(
                "No Codex OAuth credentials found. "
                "Run: python codex_auth.py"
            )
        return api_key


# ---------------------------------------------------------------------------
# Process-wide shared bridge singleton
# ---------------------------------------------------------------------------
#
# Only ONE OAuth bridge should run per process: it binds a fixed port
# (CODEX_OAUTH_BRIDGE_PORT, default 7777) and the OAuth callback port 1455.
# Multiple CodexSubscriptionClient() instances each calling start() would
# collide on the port (the Node bridge exits on EADDRINUSE). Obtain the client
# through get_shared_subscription_client() so callers share a single bridge.

_shared_client: Optional["CodexSubscriptionClient"] = None
_shared_client_lock = threading.Lock()


def get_shared_subscription_client() -> "CodexSubscriptionClient":
    """Return the process-wide CodexSubscriptionClient, starting it once.

    Thread-safe and idempotent: the first caller starts the Node bridge; later
    callers reuse it. ``start()`` is safe to call again (it no-ops if already
    running), but we gate construction so only one bridge process is spawned.
    """
    global _shared_client
    with _shared_client_lock:
        if _shared_client is None:
            client = CodexSubscriptionClient()
            client.start()
            _shared_client = client
        return _shared_client
