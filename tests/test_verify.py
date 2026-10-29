from medical import (
    Claim,
    ClaimVerdict,
    Evidence,
    Polarity,
    Provenance,
    Status,
    aggregate,
    critique,
    verify_claim,
)


def _claim(**kw):
    return Claim(sentence="x", finding="pneumonia", **kw)


def test_missing_evidence_is_indeterminate_never_contradicted():
    c = _claim()
    v = verify_claim(c, evidence=[])
    assert v.status == Status.INDETERMINATE


def test_report_text_only_is_still_indeterminate():
    c = _claim()
    ev = [Evidence(source="original_report", provenance=Provenance.T0_REPORT,
                   claim_id=c.claim_id, detail="x", agrees=True)]
    assert verify_claim(c, ev).status == Status.INDETERMINATE


def test_synthetic_support_flips_to_supported():
    c = _claim()
    ev = [Evidence(source="syn", provenance=Provenance.SYNTHETIC,
                   claim_id=c.claim_id, detail="x", agrees=True, synthetic=True)]
    assert verify_claim(c, ev).status == Status.SUPPORTED


def test_synthetic_disagreement_contradicts_positive_claim():
    c = _claim()
    ev = [Evidence(source="syn", provenance=Provenance.SYNTHETIC,
                   claim_id=c.claim_id, detail="x", agrees=False, synthetic=True)]
    assert verify_claim(c, ev).status == Status.CONTRADICTED


def test_negated_claim_supported_by_disagreeing_evidence():
    c = _claim(polarity=Polarity.NEGATIVE)
    # "No pneumonia" is supported when evidence says pneumonia is absent (agrees=False).
    ev = [Evidence(source="syn", provenance=Provenance.SYNTHETIC,
                   claim_id=c.claim_id, detail="absent", agrees=False, synthetic=True)]
    assert verify_claim(c, ev).status == Status.SUPPORTED


def test_uncertain_claim_is_indeterminate_regardless_of_evidence():
    c = _claim(polarity=Polarity.UNCERTAIN)
    ev = [Evidence(source="syn", provenance=Provenance.SYNTHETIC,
                   claim_id=c.claim_id, detail="x", agrees=True, synthetic=True)]
    assert verify_claim(c, ev).status == Status.INDETERMINATE


def test_conflicting_independent_evidence_goes_indeterminate():
    c = _claim()
    ev = [
        Evidence(source="a", provenance=Provenance.SYNTHETIC, claim_id=c.claim_id,
                 detail="x", agrees=True, synthetic=True),
        Evidence(source="b", provenance=Provenance.SYNTHETIC, claim_id=c.claim_id,
                 detail="x", agrees=False, synthetic=True),
    ]
    assert verify_claim(c, ev).status == Status.INDETERMINATE


def test_critic_flags_supported_without_independent_evidence():
    c = _claim()
    v = ClaimVerdict(claim_id=c.claim_id, status=Status.SUPPORTED, explanation="x")
    ev = [Evidence(source="r", provenance=Provenance.T0_REPORT,
                   claim_id=c.claim_id, detail="x", agrees=True)]
    findings = critique([c], [v], ev)
    assert any(f.severity == "block" for f in findings)


def test_aggregate_counts_all_statuses():
    v1 = ClaimVerdict(claim_id="1", status=Status.SUPPORTED, explanation="")
    v2 = ClaimVerdict(claim_id="2", status=Status.INDETERMINATE, explanation="")
    out = aggregate([v1, v2])
    assert out == {"supported": 1, "contradicted": 0, "indeterminate": 1}
