"""Typed schemas for the audit pipeline. Provenance tiers T0–T3 are enforced here."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---- Enums -----------------------------------------------------------------

class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class ClaimType(str, Enum):
    OBJECT = "object"      # presence/absence of a finding
    ATTRIBUTE = "attribute"  # qualifier: size, severity, laterality
    COUNT = "count"        # numeric count of a finding
    SPATIAL = "spatial"    # positional relation


class Status(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INDETERMINATE = "indeterminate"


class Provenance(str, Enum):
    T0_REPORT = "T0_report_text"
    T1_EXTRACTED = "T1_extracted_claim"
    T2_WEAK_LABEL = "T2_weak_label"
    T3_VALIDATED = "T3_validated_reference"
    SYNTHETIC = "synthetic"
    IMAGE_MODEL = "image_model"  # pretrained vision classifier output (treated as independent when confident)


# ---- Core objects ----------------------------------------------------------

class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"c_{uuid4().hex[:8]}")
    sentence: str
    span: tuple[int, int] | None = None
    finding: str
    anatomy: str | None = None
    polarity: Polarity = Polarity.POSITIVE
    claim_type: ClaimType = ClaimType.OBJECT
    count: int | None = None  # for COUNT claims
    attribute: str | None = None  # for ATTRIBUTE claims (e.g. "small", "severe")
    spatial_relation: str | None = None  # for SPATIAL claims (e.g. "upper lobe")
    section: str | None = None
    extraction_method: str = "rule_based_v1"
    uncertainty_note: str | None = None


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int


class Evidence(BaseModel):
    source: str
    provenance: Provenance
    claim_id: str
    detail: str
    agrees: bool | None = None  # True = supports, False = contradicts, None = ambiguous
    confidence: float | None = None  # model probability or similar; None when not applicable
    bbox: BoundingBox | None = None  # present for detector evidence
    synthetic: bool = False
    limitations: str | None = None


class ClaimVerdict(BaseModel):
    claim_id: str
    status: Status
    explanation: str
    evidence_refs: list[str] = Field(default_factory=list)
    calibrated_confidence: float | None = None
    calibrated_probs: dict[str, float] | None = None
    risk_level: Literal["low", "medium", "high"] | None = None


class CriticFinding(BaseModel):
    claim_id: str | None
    issue: str
    severity: Literal["info", "warn", "block"] = "warn"


class TraceEntry(BaseModel):
    node: str
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    note: str | None = None


OverallStatus = Literal["clean", "contradicted", "needs_review", "insufficient_evidence"]


class AuditReport(BaseModel):
    audit_id: str = Field(default_factory=lambda: f"a_{uuid4().hex[:12]}")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    input_type: Literal["text", "synthetic_demo", "vlm_generated"]
    mode: Literal["normal", "synthetic_demo", "vlm_from_image"]
    generated_report_text: str | None = None  # populated when mode == vlm_from_image
    claims: list[Claim]
    evidence: list[Evidence]
    verdicts: list[ClaimVerdict]
    critic_findings: list[CriticFinding]
    recheck_count: int = 0
    needs_human_review: bool = False
    overall_hallucination_status: OverallStatus = "insufficient_evidence"
    trace: list[TraceEntry]
    aggregate: dict[str, int]
    verification_method: str
    disclaimer: str = (
        "VisionGuard Medical is a research prototype. Not clinically validated. "
        "Not for diagnostic or treatment decisions."
    )

    @field_validator("aggregate")
    @classmethod
    def _aggregate_keys(cls, v: dict[str, int]) -> dict[str, int]:
        allowed = {s.value for s in Status}
        extra = set(v) - allowed
        if extra:
            raise ValueError(f"aggregate contains non-status keys: {extra}")
        return v


class AuditRequest(BaseModel):
    mode: Literal["normal", "synthetic_demo", "vlm_from_image"] = "normal"
    report_text: str = Field(default="", max_length=20_000)

    @model_validator(mode="after")
    def _not_blank_in_normal(self):
        if self.mode == "normal" and not self.report_text.strip():
            raise ValueError("report_text must not be empty in normal mode")
        return self
