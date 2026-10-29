"""Transparent non-agentic baseline: extract claims and mark every one INDETERMINATE
when there is no independent evidence. Used for comparison in evaluation."""
from __future__ import annotations

from medical import ClaimVerdict, Status, extract


def baseline_audit(report_text: str) -> list[ClaimVerdict]:
    verdicts = []
    for claim in extract(report_text):
        verdicts.append(ClaimVerdict(
            claim_id=claim.claim_id,
            status=Status.INDETERMINATE,
            explanation="Baseline has no independent evidence source.",
        ))
    return verdicts
