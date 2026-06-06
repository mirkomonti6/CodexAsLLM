# USE Codex Subscription as API LLMs

Use your **ChatGPT / Codex subscription** as a local, OpenAI-compatible LLM that you can consume developing your applications.

This is a self-contained implementation of the "codex as LLM proxy" mechanism: a tiny
Node.js OAuth bridge plus a Python client that speaks the OpenAI **Responses API**
against `https://chatgpt.com/backend-api/codex/responses` using your subscription's
OAuth token — no `OPENAI_API_KEY`, no per-token API billing, no `codex` app-server
subprocess.

```
   your Python code
        │
        ▼
  ChatGPTCodexOpenAI            (drop-in for openai.OpenAI — .responses.create(...))
        │  fresh OAuth JWT per call
        ▼
  codex_oauth_bridge  ──────►  ~/.config/codex_oauth_tokens.json   (auto-refreshed)
   (Node, 127.0.0.1:7777)
        │  the JWT
        ▼
  https://chatgpt.com/backend-api/codex/responses
```

---

## What's in here

| Path | Purpose |
|------|---------|
| `codex_oauth_bridge/` | Node.js HTTP bridge that runs the ChatGPT/Codex OAuth flow and hands out auto-refreshed tokens. Uses [`@mariozechner/pi-ai`](https://www.npmjs.com/package/@mariozechner/pi-ai). |
| `codex_llm_proxy/subscription_client.py` | The Python core: `CodexSubscriptionClient` (manages the bridge) and `ChatGPTCodexOpenAI` (the OpenAI-compatible client). |
| `codex_llm_proxy/__init__.py` | Public API + `extract_output_text()` helper. |
| `codex_auth.py` | One-time OAuth login CLI (`--check`, `--revoke`). |
| `example.py` | **Runnable example** that connects to the Codex LLM. |
| `requirements.txt` / `pyproject.toml` | Python deps / installable package metadata. |

---

## Prerequisites

* **Python 3.9+**
* **Node.js >= 20** on your `PATH` (the bridge runs on Node; `npm install` runs
  automatically the first time the bridge starts).
* A **ChatGPT/Codex subscription** to log into during auth.

---

## Setup

```bash
# 1. (optional but recommended) create a virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 2. install the Python dependency
pip install -r requirements.txt
#    ...or install the package itself:
#    pip install -e .

# 3. authenticate once (opens a browser; port 1455 must be free)
python codex_auth.py

# 4. run the example
python example.py
```

The first run of the bridge installs its npm dependencies into
`codex_oauth_bridge/node_modules/` automatically. After auth, your token lives in
`~/.config/codex_oauth_tokens.json` (chmod `0600`) and is refreshed transparently.

---

## Minimal usage

```python
from codex_llm_proxy import (
    ChatGPTCodexOpenAI,
    get_shared_subscription_client,
    extract_output_text,
)

bridge = get_shared_subscription_client()          # starts the Node bridge
client = ChatGPTCodexOpenAI(get_token_fn=bridge.get_api_key)

resp = client.responses.create(
    model="gpt-5.4",
    instructions="You are a helpful assistant.",
    input=[{"role": "user", "content": "Hello!"}],
)
print(extract_output_text(resp))
```

`client.responses.create(...)` mirrors the OpenAI Responses API, so you get
parallel tool calls, `resp_...` IDs, and streaming-backed speed. Multi-turn
conversations work via `previous_response_id` — the proxy reconstructs full
history locally because the chatgpt.com endpoint doesn't accept that field
server-side.

---

## CLI reference

```bash
python codex_auth.py            # full OAuth login flow
python codex_auth.py --check    # verify the stored token is valid
python codex_auth.py --revoke   # delete ~/.config/codex_oauth_tokens.json
```

---

## Configuration (env vars)

All optional — see `.env.example`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `CODEX_MODEL` | `gpt-5.4` | Model used by `example.py`. |
| `CODEX_OAUTH_BRIDGE_PORT` | `7777` | Local port for the Node bridge. |
| `CODEX_OAUTH_DEBUG` | `0` | `1`/`true` to echo bridge stderr to your terminal. |
| `CODEX_HISTORY_MAX` | `1000` | Max turns kept for `previous_response_id` chains. |
| `CODEX_OAUTH_BRIDGE_DIR` | _(auto)_ | Override the bridge directory location. |

> The OAuth **callback** always uses port **1455** (hard-coded by the OAuth
> library). It only needs to be free during the login flow.

---

## How it works

1. **`CodexSubscriptionClient.start()`** spawns `codex_oauth_bridge/server.js`
   (Node) on `127.0.0.1:7777` and waits for `/health`.
2. **Auth** (`codex_auth.py` → `/start-auth`) drives the PKCE browser flow via
   `@mariozechner/pi-ai`. Tokens are written to `~/.config/codex_oauth_tokens.json`.
3. **`get_api_key()`** (`/get-api-key`) returns a valid access token, refreshing
   it on the bridge side when expired.
4. **`ChatGPTCodexOpenAI`** builds a fresh `openai.OpenAI` per call pointed at
   `chatgpt.com/backend-api/codex` with the required headers
   (`chatgpt-account-id`, `originator: pi`, `OpenAI-Beta: responses=experimental`),
   then streams and reassembles the response (working around an SDK assembly
   quirk where the `response.completed` event arrives with `output=None`).

---

## Troubleshooting

* **`No Codex OAuth credentials found`** → run `python codex_auth.py`.
* **`Node.js is required` / `npm is required`** → install Node.js >= 20.
* **`Port 1455 is required ... in use`** → free port 1455 and re-run auth.
* **`returned a Cloudflare challenge (HTML)`** → re-authenticate
  (`python codex_auth.py`); the subscription endpoint is gated behind Cloudflare
  and occasionally needs a fresh token.
* **Bridge logs** → set `CODEX_OAUTH_DEBUG=1` to see the Node bridge's output.

---

## Note on the source

This is a self-contained extraction of a Codex-subscription proxy from a larger
private codebase. The Python client was lightly cleaned for release (debug-only
instrumentation tied to the original repo was removed); all functional behavior is
preserved.

## License

MIT.
