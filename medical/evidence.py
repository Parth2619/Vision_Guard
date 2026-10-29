"""Evidence providers. Each provider emits Evidence items with explicit provenance.

shortcut: no image-model provider is implemented; the pipeline must therefore
return INDETERMINATE for anything that genuinely needs pixel evidence. Add an
ImageEvidenceProvider class with the same `.collect(claims) -> list[Evidence]`
signature when a real image model is wired up.
"""
from __future__ import annotations

from .claims import BoundingBox, Claim, Evidence, Provenance

# Image evidence is treated as "independent" by the verifier only when the model
# is far enough from 0.5. Default thresholds are intentionally conservative.
CONFIDENCE_PRESENT = 0.6
CONFIDENCE_ABSENT = 0.4

# Tiny deterministic synthetic knowledge base keyed on (finding, anatomy_or_None).
# Each entry: agrees=True means "the fixture confirms the finding", False contradicts.
SYNTHETIC_KB: dict[tuple[str, str | None], tuple[bool, str]] = {
    ("pleural effusion", "right lung"): (True, "Synthetic fixture: right-sided pleural effusion present."),
    ("cardiomegaly", None): (False, "Synthetic fixture: heart size within normal limits."),
    ("pneumothorax", None): (False, "Synthetic fixture: no pneumothorax observed."),
    ("fracture", "ribs"): (True, "Synthetic fixture: nondisplaced right 6th rib fracture."),
}


def report_text_evidence(claim: Claim, full_text: str) -> list[Evidence]:
    """T0 evidence — the source report text itself. Agreement with T0 is NOT clinical truth."""
    return [
        Evidence(
            source="original_report",
            provenance=Provenance.T0_REPORT,
            claim_id=claim.claim_id,
            detail=f"Claim derived from report text: '{claim.sentence}'",
            agrees=True,  # trivially agrees with itself; must not be read as clinical support
            limitations="Agreement with source report is not independent clinical verification.",
        )
    ]


def synthetic_evidence(claim: Claim) -> list[Evidence]:
    key = (claim.finding.lower(), (claim.anatomy or "").lower() or None)
    hit = SYNTHETIC_KB.get(key) or SYNTHETIC_KB.get((claim.finding.lower(), None))
    if hit is None:
        return []
    agrees, detail = hit
    return [
        Evidence(
            source="synthetic_fixture",
            provenance=Provenance.SYNTHETIC,
            claim_id=claim.claim_id,
            detail=detail,
            agrees=agrees,
            synthetic=True,
            limitations="Synthetic demonstration data, not real patient evidence.",
        )
    ]


def image_evidence(claim: Claim, probs: dict[str, float],
                   finding_to_label: dict[str, str]) -> list[Evidence]:
    """Convert classifier probabilities for one claim into T2 image-model evidence.

    Only emits an evidence item when the model is confident; mid-range probabilities
    produce no evidence and the claim stays indeterminate.
    """
    label = finding_to_label.get(claim.finding.lower())
    if label is None or label not in probs:
        return []
    p = probs[label]
    if p >= CONFIDENCE_PRESENT:
        agrees = True
    elif p <= CONFIDENCE_ABSENT:
        agrees = False
    else:
        return []
    return [Evidence(
        source=f"xray_classifier[{label}]",
        provenance=Provenance.IMAGE_MODEL,
        claim_id=claim.claim_id,
        detail=f"Classifier probability for '{label}' = {p:.2f}",
        agrees=agrees,
        confidence=p,
        limitations=("Weak label from a pretrained general chest-X-ray model. "
                     "Not clinical ground truth."),
    )]


def detection_evidence(claim: Claim, detections) -> list[Evidence]:
    """Emit evidence for a claim from Grounding-DINO detections.

    A detection whose label contains the claim's finding (or anatomy for spatial
    claims) counts as an agree/disagree signal depending on the claim's polarity.
    Only one evidence item per claim (the highest-scoring matching detection).
    """
    needle = claim.finding.lower()
    matched = [d for d in detections if needle in d.label.lower()]
    if not matched:
        return []
    best = max(matched, key=lambda d: d.score)
    present = claim.polarity.value != "negative"
    return [Evidence(
        source=f"grounding_dino[{best.label}]",
        provenance=Provenance.IMAGE_MODEL,
        claim_id=claim.claim_id,
        detail=f"Detected '{best.label}' with score {best.score:.2f}.",
        agrees=present,
        confidence=best.score,
        bbox=BoundingBox(x1=best.x1, y1=best.y1, x2=best.x2, y2=best.y2,
                         image_width=best.image_width, image_height=best.image_height),
        limitations=("Experimental zero-shot detection. Not a calibrated disease "
                     "probability. A bounding box does not prove a medical condition."),
    )]


def collect(claims: list[Claim], full_text: str, *, synthetic: bool,
            image_probs: dict[str, float] | None = None,
            finding_to_label: dict[str, str] | None = None,
            detections: list | None = None) -> list[Evidence]:
    out: list[Evidence] = []
    for c in claims:
        out.extend(report_text_evidence(c, full_text))
        if synthetic:
            out.extend(synthetic_evidence(c))
        if image_probs is not None and finding_to_label is not None:
            out.extend(image_evidence(c, image_probs, finding_to_label))
        if detections is not None:
            out.extend(detection_evidence(c, detections))
    return out
