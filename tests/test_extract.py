from medical import Polarity, extract


def test_extract_sections_and_polarity():
    text = ("FINDINGS: There is a right-sided pleural effusion. "
            "No pneumothorax is seen. Possible atelectasis.\n"
            "IMPRESSION: Cardiomegaly.")
    claims = extract(text)
    findings = {c.finding: c for c in claims}
    assert "pleural effusion" in findings
    assert findings["pleural effusion"].polarity == Polarity.POSITIVE
    assert findings["pneumothorax"].polarity == Polarity.NEGATIVE
    assert findings["atelectasis"].polarity == Polarity.UNCERTAIN
    assert findings["cardiomegaly"].section == "IMPRESSION"


def test_sentences_without_findings_are_skipped():
    assert extract("Patient is cooperative. No concerns stated by the technician.") == []


def test_negation_preserves_finding():
    [c] = extract("No pleural effusion identified.")
    assert c.finding == "pleural effusion"
    assert c.polarity == Polarity.NEGATIVE
