"""DEPRECATED (Phase 2G): project migrated to IU Chest X-Ray.
See docs/dataset_migration_iu_xray.md. Kept for history and existing tests.

Pure helpers for building a RadGenome grounded-reports manifest.

Kept free of Hugging Face imports so that tests can run offline and so our
local `datasets/` package never has to touch the HF `datasets` library here
(the collision is handled only in the CLI script).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_VERSION = "0.1.0"
DEFAULT_SAMPLE_LIMIT = 100
MAX_SAMPLE_LIMIT = 2000  # safety cap; raise only on explicit user approval


@dataclass
class VolumeSummary:
    rows: int = 0
    anatomies: set[str] = field(default_factory=set)
    non_empty_sentences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "unique_anatomies": sorted(self.anatomies),
            "non_empty_sentences": self.non_empty_sentences,
        }


@dataclass
class Aggregate:
    sampled_rows: int = 0
    unique_volumes: int = 0
    unique_anatomies: int = 0
    non_empty_sentences: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    per_volume: dict[str, dict[str, Any]] = field(default_factory=dict)


def _norm(record: Mapping[str, Any]) -> dict[str, Any]:
    """Lowercase-key lookup so 'Volumename' / 'VolumeName' both work."""
    return {k.lower(): v for k, v in record.items()}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> Aggregate:
    """Compute the manifest aggregate. Does NOT retain sentence text."""
    missing: dict[str, int] = {"Volumename": 0, "Anatomy": 0, "Sentence": 0}
    per_vol: dict[str, VolumeSummary] = defaultdict(VolumeSummary)
    unique_anatomies: set[str] = set()
    total = 0
    non_empty_sent = 0

    for raw in records:
        total += 1
        rec = _norm(raw)
        vol = rec.get("volumename")
        anat = rec.get("anatomy")
        sent = rec.get("sentence")

        if _is_empty(vol):
            missing["Volumename"] += 1
            vol_key = "<missing>"
        else:
            vol_key = str(vol)
        if _is_empty(anat):
            missing["Anatomy"] += 1
        if _is_empty(sent):
            missing["Sentence"] += 1
        else:
            non_empty_sent += 1

        summary = per_vol[vol_key]
        summary.rows += 1
        if not _is_empty(anat):
            summary.anatomies.add(str(anat))
            unique_anatomies.add(str(anat))
        if not _is_empty(sent):
            summary.non_empty_sentences += 1

    return Aggregate(
        sampled_rows=total,
        unique_volumes=len([k for k in per_vol if k != "<missing>"]),
        unique_anatomies=len(unique_anatomies),
        non_empty_sentences=non_empty_sent,
        missing_by_column=missing,
        per_volume={k: v.to_dict() for k, v in per_vol.items()},
    )


def build_manifest(
    agg: Aggregate,
    *,
    dataset_id: str,
    config: str,
    sample_limit: int,
    include_sentence_preview: bool = False,
    preview_examples: list[str] | None = None,
) -> dict[str, Any]:
    """Shape the final manifest dict. Sentences stay out unless opted in."""
    manifest: dict[str, Any] = {
        "meta": {
            "dataset_id": dataset_id,
            "config": config,
            "sample_limit": sample_limit,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script_version": SCRIPT_VERSION,
            "disclaimer": (
                "Research prototype metadata only. Not a hallucination-labeled dataset. "
                "Grounded sentences are source captions, not verified hallucination labels."
            ),
        },
        "aggregate": {
            "sampled_rows": agg.sampled_rows,
            "unique_volumes": agg.unique_volumes,
            "unique_anatomies": agg.unique_anatomies,
            "non_empty_sentences": agg.non_empty_sentences,
            "missing_by_column": agg.missing_by_column,
        },
        "per_volume": agg.per_volume,
    }
    if include_sentence_preview and preview_examples:
        manifest["sentence_preview"] = [s[:80] + ("…" if len(s) > 80 else "") for s in preview_examples[:5]]
    return manifest


def write_manifest(manifest: dict[str, Any], out_path: Path, *, force: bool = False) -> Path:
    """Write manifest JSON. Refuses to overwrite unless force=True."""
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Manifest already exists at {out_path}. Pass --force to overwrite."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def clamp_limit(requested: int, allow_override: bool = False) -> int:
    if requested < 1:
        raise ValueError("sample limit must be >= 1")
    if requested > MAX_SAMPLE_LIMIT and not allow_override:
        raise ValueError(
            f"sample limit {requested} exceeds safety cap {MAX_SAMPLE_LIMIT}. "
            "Pass --allow-large to override."
        )
    return requested
