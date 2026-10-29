"""DEPRECATED (Phase 2G): RadGenome-ChestCT is no longer the active dataset.
See docs/dataset_migration_iu_xray.md. Kept for history.

Check Hugging Face auth without printing the token.

Usage (in the project venv):
    python scripts/hf_auth_check.py
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from huggingface_hub import whoami
    except ImportError:
        print("ERROR: huggingface_hub not installed in this environment.")
        return 2

    try:
        info = whoami()
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the user
        print(f"NOT AUTHENTICATED: {type(exc).__name__}: {exc}")
        print("Run: huggingface-cli login   (paste your token at the prompt, never in chat)")
        return 1

    name = info.get("name") or info.get("fullname") or "<unknown>"
    orgs = [o.get("name") for o in info.get("orgs", []) if isinstance(o, dict)]
    print(f"AUTHENTICATED as: {name}")
    if orgs:
        print(f"Orgs: {', '.join(o for o in orgs if o)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
