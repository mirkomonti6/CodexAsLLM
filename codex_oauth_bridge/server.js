/**
 * Codex OAuth Bridge
 *
 * A minimal HTTP service (127.0.0.1 only) that manages the ChatGPT/Codex
 * subscription OAuth lifecycle using @mariozechner/pi-ai.
 *
 * Endpoints:
 *   GET  /health        → { status, has_credentials, expires? }
 *   POST /start-auth    → { authorize_url } (waits for PKCE setup, then responds)
 *   GET  /auth-status   → { status: "pending"|"complete"|"failed", url?, error? }
 *   POST /exchange-code → { code } — manual code injection (fallback for onPrompt)
 *   GET  /get-api-key   → { api_key, expires } | HTTP 401
 *
 * Environment:
 *   CODEX_OAUTH_BRIDGE_PORT  (default 7777)
 *
 * Token storage: ~/.config/codex_oauth_tokens.json (chmod 0o600)
 */

// Polyfill globalThis.crypto for Node.js < 20 (pi-ai uses Web Crypto API)
import { webcrypto } from "node:crypto";
if (typeof globalThis.crypto === "undefined") {
  globalThis.crypto = webcrypto;
}

import { loginOpenAICodex, refreshOpenAICodexToken, getOAuthApiKey } from
  "@mariozechner/pi-ai/oauth";
import { createServer } from "node:http";
import { readFileSync, writeFileSync, chmodSync, existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { setDefaultResultOrder } from "node:dns";

// Cloudflare answers AAAA-first for auth.openai.com; on networks where IPv6
// egress is broken, Node's undici waits the full connect timeout (~10s) on
// IPv6 instead of racing IPv4. Force IPv4-first so the OAuth flow reaches
// OpenAI immediately.
setDefaultResultOrder("ipv4first");

function log(...args) {
  console.log(new Date().toISOString(), "[bridge]", ...args);
}

function logError(...args) {
  console.error(new Date().toISOString(), "[bridge]", ...args);
}

function formatError(err) {
  if (!err) return { message: "unknown error" };
  const out = {
    name: err.name,
    message: err.message,
  };
  if (err.code) out.code = err.code;
  if (err.cause) {
    out.cause =
      typeof err.cause === "object"
        ? { name: err.cause.name, message: err.cause.message, code: err.cause.code }
        : String(err.cause);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Token persistence
// ---------------------------------------------------------------------------
const TOKEN_PATH = join(homedir(), ".config", "codex_oauth_tokens.json");

function loadTokens() {
  try {
    return JSON.parse(readFileSync(TOKEN_PATH, "utf8"));
  } catch {
    return {};
  }
}

function saveTokens(credentialsMap) {
  const dir = dirname(TOKEN_PATH);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(TOKEN_PATH, JSON.stringify(credentialsMap, null, 2), "utf8");
  try { chmodSync(TOKEN_PATH, 0o600); } catch {}
}

// ---------------------------------------------------------------------------
// Auth state machine
// ---------------------------------------------------------------------------
// status: "idle" | "pending" | "complete" | "failed"
const authState = {
  status: "idle",
  url: null,
  error: null,
  resolveCode: null,   // resolved by /exchange-code → feeds onPrompt/onManualCodeInput
  resolveUrl: null,    // resolved when onAuth fires → unblocks /start-auth response
};

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------
function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => (data += chunk));
    req.on("end", () => {
      try { resolve(JSON.parse(data || "{}")); }
      catch { resolve({}); }
    });
    req.on("error", reject);
  });
}

function send(res, statusCode, body) {
  const json = JSON.stringify(body);
  res.writeHead(statusCode, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(json),
  });
  res.end(json);
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------
async function handleHealth(res) {
  const tokens = loadTokens();
  const creds = tokens["openai-codex"];
  send(res, 200, {
    status: "ok",
    has_credentials: !!creds,
    ...(creds ? { expires: creds.expires } : {}),
  });
}

async function handleStartAuth(res) {
  if (authState.status === "pending") {
    log("start-auth: already pending", authState.url ? "(has URL)" : "(no URL yet)");
    // Already in progress — return current URL if available
    send(res, 200, {
      status: "already_pending",
      ...(authState.url ? { authorize_url: authState.url } : {}),
    });
    return;
  }

  authState.status = "pending";
  authState.url = null;
  authState.error = null;
  authState.resolveCode = null;
  authState.resolveUrl = null;

  // Promise that resolves when onAuth fires (URL is ready)
  const urlPromise = new Promise((resolve) => {
    authState.resolveUrl = resolve;
  });

  // Promise for manual code paste (used as onManualCodeInput — races browser callback)
  const manualCodePromise = () =>
    new Promise((resolve) => {
      authState.resolveCode = (input) => resolve(input);
    });

  // Start the OAuth flow in background — do NOT await here
  loginOpenAICodex({
    onAuth({ url }) {
      authState.url = url;
      if (authState.resolveUrl) {
        authState.resolveUrl(url);
        authState.resolveUrl = null;
      }
    },
    onPrompt({ message }) {
      // Called as last-resort fallback. Return a promise that /exchange-code resolves.
      log("onPrompt:", message);
      return new Promise((resolve) => {
        authState.resolveCode = resolve;
      });
    },
    onManualCodeInput: manualCodePromise,
    onProgress(msg) {
      log("onProgress:", msg);
    },
  })
    .then((creds) => {
      const tokens = loadTokens();
      tokens["openai-codex"] = creds;
      saveTokens(tokens);
      authState.status = "complete";
      log("OAuth complete. Token expires:", new Date(creds.expires).toISOString());
    })
    .catch((err) => {
      const details = formatError(err);
      logError("OAuth error:", JSON.stringify(details));
      authState.status = "failed";
      authState.error =
        err?.code === "EADDRINUSE"
          ? "port_1455_in_use"
          : details.message || "fetch failed";
      // Unblock /start-auth if still waiting
      if (authState.resolveUrl) {
        authState.resolveUrl(null);
        authState.resolveUrl = null;
      }
    });

  // Wait for onAuth (URL is ready) with a 20-second timeout
  const url = await Promise.race([
    urlPromise,
    new Promise((resolve) => setTimeout(() => resolve(null), 20000)),
  ]);

  if (!url) {
    const errMsg = authState.error || "OAuth flow did not provide an authorize_url within 20s";
    authState.status = "failed";
    authState.error = errMsg;
    if (errMsg === "port_1455_in_use") {
      send(res, 500, {
        error: "port_1455_in_use",
        message:
          "Port 1455 is required for the OAuth callback (hard-coded by the library). " +
          "Ensure no other process is using port 1455 and try again.",
      });
    } else {
      send(res, 500, { error: errMsg });
    }
    return;
  }

  log("start-auth: authorize_url ready");
  send(res, 200, { authorize_url: url, status: "started" });
}

async function handleAuthStatus(res) {
  send(res, 200, {
    status: authState.status,
    ...(authState.url ? { url: authState.url } : {}),
    ...(authState.error ? { error: authState.error } : {}),
  });
}

async function handleExchangeCode(req, res) {
  const body = await readBody(req);
  const code = body.code;
  if (!code) {
    send(res, 400, { error: "missing field: code" });
    return;
  }
  if (authState.resolveCode) {
    authState.resolveCode(code);
    authState.resolveCode = null;
    log("exchange-code: accepted");
    send(res, 200, { status: "accepted" });
  } else {
    log("exchange-code: no_pending_prompt");
    send(res, 200, { status: "no_pending_prompt", note: "No active prompt waiting for a code." });
  }
}

async function handleGetApiKey(res) {
  const tokens = loadTokens();

  if (!tokens["openai-codex"]) {
    send(res, 401, { error: "not_authenticated" });
    return;
  }

  try {
    // getOAuthApiKey handles auto-refresh and returns { newCredentials, apiKey }
    const result = await getOAuthApiKey("openai-codex", tokens);

    if (!result) {
      send(res, 401, { error: "not_authenticated" });
      return;
    }

    const { newCredentials, apiKey } = result;

    // Persist if credentials were refreshed
    if (newCredentials !== tokens["openai-codex"]) {
      tokens["openai-codex"] = newCredentials;
      saveTokens(tokens);
      log("Token refreshed. New expiry:", new Date(newCredentials.expires).toISOString());
    }

    send(res, 200, { api_key: apiKey, expires: newCredentials.expires });
  } catch (err) {
    logError("Error getting/refreshing token:", err.message);
    send(res, 500, { error: err.message });
  }
}

// ---------------------------------------------------------------------------
// Router — all async handlers must be awaited so errors are caught
// ---------------------------------------------------------------------------
async function router(req, res) {
  const { method, url } = req;
  const pathname = (url || "").split("?")[0];
  log(`${method} ${pathname}`);
  try {
    if (method === "GET"  && url === "/health")       return await handleHealth(res);
    if (method === "POST" && url === "/start-auth")   return await handleStartAuth(res);
    if (method === "GET"  && url === "/auth-status")  return await handleAuthStatus(res);
    if (method === "POST" && url === "/exchange-code") return await handleExchangeCode(req, res);
    if (method === "GET"  && url === "/get-api-key")  return await handleGetApiKey(res);
    send(res, 404, { error: "not_found" });
  } catch (err) {
    logError("Unhandled error in router:", err);
    try { send(res, 500, { error: err.message }); } catch {}
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.CODEX_OAUTH_BRIDGE_PORT || "7777", 10);

const server = createServer(router);
server.listen(PORT, "127.0.0.1", () => {
  log(`Codex OAuth bridge listening on 127.0.0.1:${PORT}`);
  log(`Token file: ${TOKEN_PATH}`);
});

server.on("error", (err) => {
  logError("Server error:", err.message);
  process.exit(1);
});

process.on("SIGTERM", () => { server.close(); process.exit(0); });
process.on("SIGINT",  () => { server.close(); process.exit(0); });
