"""DEPRECATED (Phase 2G): project migrated to IU Chest X-Ray.
See docs/dataset_migration_iu_xray.md. Kept for history and existing tests.

Expected schema for RadGenome-ChestCT grounded-report metadata.

These names are an *initial expectation* taken from the public dataset
description. Always validate against a real record before relying on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

EXPECTED_COLUMNS: tuple[str, ...] = ("VolumeName", "Anatomy", "Sentence")


@dataclass(frozen=True)
class SchemaCheck:
    present: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


def validate_columns(
    actual: Iterable[str], expected: Iterable[str] = EXPECTED_COLUMNS
) -> SchemaCheck:
    actual_set = {c for c in actual}
    expected_set = {c for c in expected}
    # Case-insensitive match so "Volumename" vs "VolumeName" doesn't false-fail.
    actual_lower = {c.lower(): c for c in actual_set}
    present, missing = [], []
    for col in expected_set:
        if col.lower() in actual_lower:
            present.append(actual_lower[col.lower()])
        else:
            missing.append(col)
    extra = sorted(actual_set - set(present))
    return SchemaCheck(tuple(sorted(present)), tuple(sorted(missing)), tuple(extra))


def validate_record(record: dict[str, Any]) -> SchemaCheck:
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}")
    return validate_columns(record.keys())
