from __future__ import annotations

import getpass
import sys
from collections.abc import Callable, Sequence

from ..db import SessionLocal
from ..services.auth import create_user

PasswordReader = Callable[[str], str]


def main(
    argv: Sequence[str] | None = None,
    *,
    password_reader: PasswordReader = getpass.getpass,
    session_factory=SessionLocal,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m app.cli.create_user <email>", file=sys.stderr)
        return 2

    email = args[0]
    password = password_reader("Initial password: ")
    confirmation = password_reader("Confirm password: ")
    if password != confirmation:
        print("Passwords do not match.", file=sys.stderr)
        return 2

    with session_factory() as session:
        try:
            user = create_user(session, email, password)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    print(f"Created user: {user.email}")
    print("The user must change this password on first login.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
