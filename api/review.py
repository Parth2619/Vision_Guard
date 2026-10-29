"""Expert-review persistence (Phase 8). Tiny SQLite table, stdlib only."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["confirm", "reject", "insufficient"]

DB_PATH = Path(os.environ.get("VISIONGUARD_REVIEW_DB", "reviews.db"))


def _conn(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS review ("
        " audit_id TEXT, claim_id TEXT, decision TEXT, note TEXT, at TEXT,"
        " PRIMARY KEY (audit_id, claim_id))"
    )
    return conn


class ReviewIn(BaseModel):
    audit_id: str = Field(min_length=1, max_length=64)
    claim_id: str = Field(min_length=1, max_length=64)
    decision: Decision
    note: str = Field(default="", max_length=2000)


class ReviewRow(BaseModel):
    audit_id: str
    claim_id: str
    decision: Decision
    note: str
    at: str


def record_review(row: ReviewIn, path: Path | None = None) -> ReviewRow:
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn(path) as c:
        c.execute(
            "INSERT INTO review VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(audit_id, claim_id) DO UPDATE SET "
            "decision=excluded.decision, note=excluded.note, at=excluded.at",
            (row.audit_id, row.claim_id, row.decision, row.note, at),
        )
    return ReviewRow(**row.model_dump(), at=at)


def list_reviews(audit_id: str, path: Path | None = None) -> list[ReviewRow]:
    with _conn(path) as c:
        rows = c.execute(
            "SELECT audit_id, claim_id, decision, note, at FROM review "
            "WHERE audit_id = ? ORDER BY at",
            (audit_id,),
        ).fetchall()
    return [ReviewRow(audit_id=a, claim_id=c, decision=d, note=n, at=t)
            for a, c, d, n, t in rows]
