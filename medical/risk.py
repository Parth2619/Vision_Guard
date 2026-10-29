"""Phase 8 — clinical-urgency table used by the risk-aware decision gate.

Static mapping based on common radiology triage categories. Not a medical
recommendation; it only drives how aggressively the pipeline escalates to
human review.
"""
from __future__ import annotations

from .claims import Claim

_HIGH = {"pneumothorax", "mass", "pneumonia", "fracture"}
_LOW = {"atelectasis", "emphysema"}


def claim_risk(claim: Claim) -> str:
    f = claim.finding.lower()
    if any(k in f for k in _HIGH):
        return "high"
    if any(k in f for k in _LOW):
        return "low"
    return "medium"
