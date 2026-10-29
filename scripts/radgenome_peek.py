"""DEPRECATED (Phase 2G): project migrated to IU Chest X-Ray.
See docs/dataset_migration_iu_xray.md. Kept for history.

Pull a tiny sample of RadGenome-ChestCT grounded-report metadata.

- Streams one record (no CT volumes, no bulk download).
- Prints the keys and validates them against the expected schema.
- Stops immediately on access errors; does NOT bypass gating.

Usage (from the project root, in the venv):
    python scripts/radgenome_peek.py
"""
from __future__ import annotations

import pathlib
import sys

# Import our schema first (local `datasets/` package wins here — that's fine).
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from datasets.radgenome_schema import validate_record  # noqa: E402

# Now resolve the HF `datasets` library. Our local `datasets/` package shadows
# it, so drop any cached entry and the project root from sys.path, then import.
for _m in [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]:
    if _m != "datasets.radgenome_schema":
        del sys.modules[_m]
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != _ROOT]

try:
    from datasets import load_dataset  # type: ignore  # HF library
except ImportError:
    print("ERROR: Hugging Face `datasets` not installed. pip install datasets")
    sys.exit(2)

DATASET_ID = "RadGenome/RadGenome-ChestCT"
CONFIG = "grounded reports"  # verified via HF ValueError listing available configs


def main() -> int:
    try:
        ds = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True)
        record = next(iter(ds))
    except Exception as exc:  # noqa: BLE001 - surface the real cause verbatim
        name = type(exc).__name__
        msg = str(exc)
        print(f"ACCESS DENIED or LOAD FAILED: {name}: {msg}")
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "access" in msg.lower():
            print("Dataset is gated. Request access on the HF dataset page, then "
                  "run `huggingface-cli login` locally.")
        return 1

    print(f"OK — pulled 1 streaming record from {DATASET_ID}")
    print(f"Keys: {sorted(record.keys())}")
    check = validate_record(record)
    print(f"Expected present: {check.present}")
    if check.missing:
        print(f"Expected MISSING (schema drift — update code before relying on these): {check.missing}")
    if check.extra:
        print(f"Extra columns (informational): {check.extra}")
    return 0 if check.ok else 3


if __name__ == "__main__":
    sys.exit(main())
