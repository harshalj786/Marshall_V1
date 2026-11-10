"""Set Marshall's admin password. Run once, by the owner.

    python scripts/set_admin_password.py
    python scripts/set_admin_password.py --path db/admin/password.hash

Asks twice, hidden, and stores an Argon2id hash — never the password. Running it again
replaces the password. Delete the file to go back to guest-only sessions.
"""
import argparse
import getpass
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from admin_session import ADMIN_HASH_PATH, set_admin_password


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/set_admin_password.py",
                                     description="Set Marshall's admin password.")
    parser.add_argument("--path", default=ADMIN_HASH_PATH,
                        help=f"where the hash is stored (default: {ADMIN_HASH_PATH})")
    args = parser.parse_args(argv)

    first = getpass.getpass("new admin password: ")
    if not first:
        print("password cannot be empty")
        return 1
    if getpass.getpass("again: ") != first:
        print("passwords do not match")
        return 1
    print(f"admin password set: {set_admin_password(first, args.path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
