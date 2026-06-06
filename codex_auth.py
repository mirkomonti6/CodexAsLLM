#!/usr/bin/env python3
"""
codex_auth.py — First-time and re-authentication helper for the
Codex subscription OAuth flow.

Usage:
    python codex_auth.py           # full OAuth flow (opens a browser)
    python codex_auth.py --check   # verify existing token
    python codex_auth.py --revoke  # delete stored tokens

After successful auth, the token is stored in:
    ~/.config/codex_oauth_tokens.json

Then connect to the Codex LLM from Python:
    python example.py
"""

import argparse
import sys
import threading
import urllib.parse
import webbrowser

from codex_llm_proxy.subscription_client import (
    CodexSubscriptionAuthError,
    CodexSubscriptionClient,
    CodexSubscriptionNotAuthenticatedError,
)


def cmd_auth(client: CodexSubscriptionClient) -> None:
    print("Starting Codex OAuth flow...")
    print("Note: port 1455 must be free (used for the OAuth callback).\n")

    try:
        url = client.start_auth()
    except CodexSubscriptionAuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Authorization URL:\n  {url}\n")

    opened = webbrowser.open(url)
    if opened:
        print("Browser opened. Complete the login in your browser.")
    else:
        print("Could not open a browser automatically.")
        print("Please open the URL above manually.")

    # Background thread: allow manual redirect URL paste as fallback
    def _manual_input():
        try:
            paste = input(
                "\nIf the browser flow does not complete automatically,\n"
                "paste the full redirect URL here (or press Enter to skip): "
            ).strip()
            if paste:
                # The bridge handles the code exchange via the callback server;
                # this is a best-effort hint only.
                try:
                    code = paste
                    if paste.startswith("http://") or paste.startswith("https://"):
                        parsed = urllib.parse.urlparse(paste)
                        qs = urllib.parse.parse_qs(parsed.query)
                        code = (qs.get("code", [""]) or [""])[0] or paste
                    client._post("/exchange-code", {"code": code})
                except Exception:
                    pass
        except (EOFError, KeyboardInterrupt):
            pass

    t = threading.Thread(target=_manual_input, daemon=True)
    t.start()

    print("\nWaiting for authentication (up to 5 minutes)...")
    try:
        client.wait_for_auth(timeout=300.0)
    except CodexSubscriptionAuthError as exc:
        print(f"\nAuthentication failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verify we can retrieve a key
    try:
        api_key = client.get_api_key()
    except Exception as exc:
        print(
            f"\nAuthentication completed but could not retrieve key: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    masked = api_key[:8] + "..." if len(api_key) > 8 else "***"
    print("\nAuthenticated successfully!")
    print(f"  Token (masked): {masked}")
    print(f"  Stored at:      {CodexSubscriptionClient.TOKEN_PATH}")
    print()
    print("Now try the example:")
    print("  python example.py")


def cmd_check(client: CodexSubscriptionClient) -> None:
    try:
        api_key = client.get_api_key()
    except CodexSubscriptionNotAuthenticatedError:
        print("No credentials found. Run without --check to authenticate.")
        sys.exit(1)
    except Exception as exc:
        print(f"Error checking credentials: {exc}", file=sys.stderr)
        sys.exit(1)

    masked = api_key[:8] + "..." if len(api_key) > 8 else "***"
    print(f"Token valid. (masked: {masked})")
    print(f"File: {CodexSubscriptionClient.TOKEN_PATH}")


def cmd_revoke() -> None:
    token_path = CodexSubscriptionClient.TOKEN_PATH
    if token_path.exists():
        token_path.unlink()
        print(f"Removed: {token_path}")
    else:
        print("No token file found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Codex subscription OAuth authentication helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Verify existing token")
    group.add_argument("--revoke", action="store_true", help="Delete stored token")
    args = parser.parse_args()

    if args.revoke:
        cmd_revoke()
        return

    # All other commands need the bridge running
    client = CodexSubscriptionClient()
    try:
        client.start()
    except Exception as exc:
        print(f"Failed to start OAuth bridge: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.check:
            cmd_check(client)
        else:
            cmd_auth(client)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
