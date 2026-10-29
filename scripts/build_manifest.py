"""DEPRECATED (Phase 2G): project migrated to IU Chest X-Ray.
See docs/dataset_migration_iu_xray.md. Kept for history.

Build a small, reproducible manifest from RadGenome-ChestCT grounded-reports metadata.

Only streams metadata rows (no CT volumes, masks, or model weights). Sentences
are NOT written to the manifest by default — only aggregate counts.

Usage (project venv):
    python scripts/build_manifest.py                       # 100 records, default path
    python scripts/build_manifest.py --limit 250
    python scripts/build_manifest.py --out artifacts/manifests/custom.json --force
    python scripts/build_manifest.py --preview             # opt-in truncated sentence samples

Import-collision note:
    Our local `datasets/` package shadows the Hugging Face `datasets` library.
    This script imports local helpers first, then pops the project root from
    sys.path before importing HF `datasets`. Keep this pattern in any script
    that needs the HF library.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Local helpers first — our `datasets/` package wins here, which is what we want.
from datasets.manifest import (  # noqa: E402
    DEFAULT_SAMPLE_LIMIT,
    aggregate_records,
    build_manifest,
    clamp_limit,
    write_manifest,
)
from datasets.radgenome_schema import validate_record  # noqa: E402

# Now flip to the HF library: drop cached shadow modules and the project root.
for _m in [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]:
    if _m not in ("datasets.manifest", "datasets.radgenome_schema"):
        del sys.modules[_m]
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != _ROOT]

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    print("ERROR: Hugging Face `datasets` not installed. pip install datasets")
    sys.exit(2)

DATASET_ID = "RadGenome/RadGenome-ChestCT"
CONFIG = "grounded reports"
DEFAULT_OUT = _ROOT / "artifacts" / "manifests" / "radgenome_grounded_reports.json"


def _stream_records(limit: int, preview: bool):
    ds = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True)
    collected: list[dict] = []
    previews: list[str] = []
    for i, record in enumerate(ds):
        if i == 0:
            chk = validate_record(record)
            if chk.missing:
                print(f"WARNING: expected columns missing from live schema: {chk.missing}")
        if i >= limit:
            break
        collected.append(record)
        if preview and len(previews) < 5:
            sent = record.get("Sentence") or record.get("sentence")
            if isinstance(sent, str) and sent.strip():
                previews.append(sent.strip())
    return collected, previews


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_LIMIT,
                   help=f"records to sample (default {DEFAULT_SAMPLE_LIMIT})")
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                   help="manifest output path")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    p.add_argument("--preview", action="store_true",
                   help="include up to 5 truncated sentence previews (opt-in)")
    p.add_argument("--allow-large", action="store_true",
                   help="override the safety cap on --limit")
    args = p.parse_args()

    try:
        limit = clamp_limit(args.limit, allow_override=args.allow_large)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        records, previews = _stream_records(limit, args.preview)
    except Exception as exc:  # noqa: BLE001
        print(f"ACCESS DENIED or LOAD FAILED: {type(exc).__name__}: {exc}")
        return 1

    agg = aggregate_records(records)
    manifest = build_manifest(
        agg,
        dataset_id=DATASET_ID,
        config=CONFIG,
        sample_limit=limit,
        include_sentence_preview=args.preview,
        preview_examples=previews,
    )

    try:
        path = write_manifest(manifest, args.out, force=args.force)
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"OK — wrote manifest: {path}")
    a = manifest["aggregate"]
    print(f"  sampled_rows        : {a['sampled_rows']}")
    print(f"  unique_volumes      : {a['unique_volumes']}")
    print(f"  unique_anatomies    : {a['unique_anatomies']}")
    print(f"  non_empty_sentences : {a['non_empty_sentences']}")
    print(f"  missing_by_column   : {a['missing_by_column']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
