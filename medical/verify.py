"""Conservative verification rules. Missing evidence is INDETERMINATE, never CONTRADICTED."""
from __future__ import annotations

from .claims import Claim, ClaimVerdict, CriticFinding, Evidence, Polarity, Provenance, Status

# Provenance tiers strong enough to flip a claim away from INDETERMINATE on their own.
# IMAGE_MODEL evidence is only emitted when the model is confident (see evidence.py),
# so it enters this set and is reported as a weak but independent signal.
_INDEPENDENT = {Provenance.T3_VALIDATED, Provenance.SYNTHETIC, Provenance.IMAGE_MODEL}


def _evidence_for(claim_id: str, evidence: list[Evidence]) -> list[Evidence]:
    return [e for e in evidence if e.claim_id == claim_id]


def verify_claim(claim: Claim, evidence: list[Evidence]) -> ClaimVerdict:
    related = _evidence_for(claim.claim_id, evidence)
    independent = [e for e in related if e.provenance in _INDEPENDENT and e.agrees is not None]

    # Align polarity: a negated claim is "supported" by evidence that disagrees with the finding.
    expects_present = claim.polarity == Polarity.POSITIVE

    if claim.polarity == Polarity.UNCERTAIN:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            status=Status.INDETERMINATE,
            explanation="Claim is hedged/uncertain in the source report; marked indeterminate.",
            evidence_refs=[e.source for e in related],
        )

    if not independent:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            status=Status.INDETERMINATE,
            explanation=(
                "No independent (T3 or synthetic) evidence available. Source-report agreement "
                "alone does not establish clinical truth."
            ),
            evidence_refs=[e.source for e in related],
        )

    supports = sum(1 for e in independent if e.agrees == expects_present)
    conflicts = sum(1 for e in independent if e.agrees != expects_present)

    if supports and conflicts:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            status=Status.INDETERMINATE,
            explanation="Conflicting independent evidence; requires human review.",
            evidence_refs=[e.source for e in independent],
        )
    if supports:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            status=Status.SUPPORTED,
            explanation=f"{supports} independent evidence item(s) agree with the claim.",
            evidence_refs=[e.source for e in independent],
        )
    return ClaimVerdict(
        claim_id=claim.claim_id,
        status=Status.CONTRADICTED,
        explanation=f"{conflicts} independent evidence item(s) contradict the claim.",
        evidence_refs=[e.source for e in independent],
    )


def critique(claims: list[Claim], verdicts: list[ClaimVerdict],
             evidence: list[Evidence]) -> list[CriticFinding]:
    findings: list[CriticFinding] = []
    for v in verdicts:
        rel = _evidence_for(v.claim_id, evidence)
        if v.status == Status.SUPPORTED and not any(
            e.provenance in _INDEPENDENT for e in rel
        ):
            findings.append(CriticFinding(
                claim_id=v.claim_id,
                issue="SUPPORTED without any independent evidence item.",
                severity="block",
            ))
        if v.status == Status.CONTRADICTED and not any(
            e.provenance in _INDEPENDENT and e.agrees is False for e in rel
        ):
            findings.append(CriticFinding(
                claim_id=v.claim_id,
                issue="CONTRADICTED without a disagreeing independent evidence item.",
                severity="block",
            ))
    for c in claims:
        if not _evidence_for(c.claim_id, evidence):
            findings.append(CriticFinding(
                claim_id=c.claim_id,
                issue="No evidence collected for claim.",
                severity="warn",
            ))
    return findings


def aggregate(verdicts: list[ClaimVerdict]) -> dict[str, int]:
    out = {s.value: 0 for s in Status}
    for v in verdicts:
        out[v.status.value] += 1
    return out
