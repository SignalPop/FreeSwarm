"""Account management for the control plane and message board.

    python -m app.usercli add <username>       # prompts for a password, twice
    python -m app.usercli list
    python -m app.usercli delete <username>
    python -m app.usercli token <username>     # mint a long-lived token for an agent
    python -m app.usercli rotate-key           # invalidate every outstanding token

Creating the first account switches authentication on: `auth.auth_enabled()` is presence-
based, and `Settings.validate()` refuses a non-loopback bind until at least one exists.

Run from `ui/backend` so `app` is importable.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import time

from . import auth


def _add(args: argparse.Namespace) -> int:
    password = args.password
    if not password:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Confirm: "):
            print("passwords do not match", file=sys.stderr)
            return 1
    try:
        auth.create_user(args.username, password)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"created {args.username.strip().lower()}")
    print("Authentication is now REQUIRED on /api/* and /mb/*.")
    return 0


def _list(_: argparse.Namespace) -> int:
    users = auth._load_users()  # noqa: SLF001 - this module is the store's admin surface
    if not users:
        print("no accounts (authentication is disabled; loopback-only access)")
        return 0
    for name, record in sorted(users.items()):
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(record.get("created_at", 0)))
        print(f"{name:24s} created {created}")
    return 0


def _delete(args: argparse.Namespace) -> int:
    if auth.delete_user(args.username):
        print(f"deleted {args.username}")
        remaining = auth.user_count()
        if remaining == 0:
            print("WARNING: no accounts remain -- authentication is now disabled.")
        return 0
    print(f"no such user: {args.username}", file=sys.stderr)
    return 1


def _token(args: argparse.Namespace) -> int:
    """Mint an access token directly, for a headless agent that cannot do a login round-trip.

    Deliberately prints the raw token: it is a bearer credential, so treat it like a
    password. It expires like any other access token; use a short TTL for anything you
    paste into a config file.
    """
    username = args.username.strip().lower()
    if username not in auth._load_users():  # noqa: SLF001
        print(f"no such user: {username}", file=sys.stderr)
        return 1
    token = auth._encode(username, args.ttl, "access")  # noqa: SLF001
    print(token)
    print(f"\n# expires in {args.ttl}s. Use as:", file=sys.stderr)
    print(f"#   OPENAI_API_KEY={token[:16]}...", file=sys.stderr)
    return 0


def _rotate(_: argparse.Namespace) -> int:
    if auth.SECRET_FILE.is_file():
        auth.SECRET_FILE.unlink()
    auth._signing_key()  # noqa: SLF001 - regenerates and persists
    print("signing key rotated; every outstanding token is now invalid")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.usercli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="create an account")
    p_add.add_argument("username")
    p_add.add_argument("--password", help="non-interactive (avoid: lands in shell history)")
    p_add.set_defaults(func=_add)

    sub.add_parser("list", help="list accounts").set_defaults(func=_list)

    p_del = sub.add_parser("delete", help="remove an account")
    p_del.add_argument("username")
    p_del.set_defaults(func=_delete)

    p_tok = sub.add_parser("token", help="mint an access token for an agent")
    p_tok.add_argument("username")
    p_tok.add_argument("--ttl", type=int, default=30 * 24 * 3600, help="seconds (default 30d)")
    p_tok.set_defaults(func=_token)

    sub.add_parser("rotate-key", help="invalidate all tokens").set_defaults(func=_rotate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
