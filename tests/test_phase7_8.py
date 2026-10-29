"""Phase 7 calibration + Phase 8 risk gate / review persistence."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.review import DB_PATH, record_review, ReviewIn, list_reviews
from graph.build import run_audit
from medical import Claim, Evidence, Polarity, Provenance
from medical.risk import claim_risk
from models.calibration import FEATURE_NAMES, calibrate, features_for, get_model
from models.xray import set_backend as set_xray

client = TestClient(app)


# ---- Phase 7: calibration ---------------------------------------------------

def test_feature_vector_shape_matches_feature_names():
    c = Claim(sentence="x", finding="pleural effusion")
    ev = [Evidence(source="xray_classifier[Effusion]", provenance=Provenance.IMAGE_MODEL,
                   claim_id=c.claim_id, detail="x", agrees=True, confidence=0.9)]
    feats = features_for(c, ev)
    assert len(feats) == len(FEATURE_NAMES) == 8


def test_calibrate_probs_sum_to_one_and_pick_a_status():
    status, probs = calibrate([0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert status in {"supported", "contradicted", "indeterminate"}
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert status == max(probs, key=probs.get)


def test_model_training_is_deterministic():
    get_model.cache_clear()
    m1 = get_model()
    p1 = calibrate([0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])[1]
    get_model.cache_clear()
    m2 = get_model()
    p2 = calibrate([0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])[1]
    assert m1 is not m2  # fresh instance
    assert p1 == p2  # same seed -> same training -> same output


def test_calibration_prefers_supported_when_independent_supports():
    # Strong "has_independent_support" signal, no conflict, not uncertain.
    _, probs = calibrate([0.95, 0.45, 0.9, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert probs["supported"] > probs["contradicted"]


def test_calibration_prefers_contradicted_when_independent_conflicts():
    _, probs = calibrate([0.1, 0.4, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert probs["contradicted"] > probs["supported"]


class _Stub:
    def probabilities(self, image_bytes): return {"Effusion": 0.9}


def test_run_audit_populates_calibration_fields():
    set_xray(_Stub())
    try:
        r = run_audit("FINDINGS: Pleural effusion.", mode="normal", image_bytes=b"\x89PNG\r\n\x1a\n")
    finally:
        set_xray(None)
    v = r.verdicts[0]
    assert v.calibrated_confidence is not None
    assert v.calibrated_probs is not None
    assert set(v.calibrated_probs) == {"supported", "contradicted", "indeterminate"}
    assert v.risk_level in {"low", "medium", "high"}
    nodes = [t.node for t in r.trace]
    assert "fusion_calibration" in nodes


# ---- Phase 8: risk gate -----------------------------------------------------

def test_claim_risk_table():
    assert claim_risk(Claim(sentence="x", finding="pneumothorax")) == "high"
    assert claim_risk(Claim(sentence="x", finding="atelectasis")) == "low"
    assert claim_risk(Claim(sentence="x", finding="opacity")) == "medium"


def test_high_risk_low_confidence_escalates_to_review():
    # No image, no synthetic -> all indeterminate -> needs_human_review=True.
    r = run_audit("FINDINGS: Pneumothorax present.", mode="normal")
    assert r.needs_human_review is True


# ---- Phase 8: review persistence --------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "reviews.db"
    monkeypatch.setattr("api.review.DB_PATH", db)
    yield db


def test_review_post_and_get_roundtrip(tmp_db):
    r = client.post("/review", json={
        "audit_id": "a_test", "claim_id": "c_1",
        "decision": "confirm", "note": "obvious",
    })
    assert r.status_code == 200
    r = client.get("/review", params={"audit_id": "a_test"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1 and rows[0]["decision"] == "confirm"


def test_review_upsert_updates_existing_row(tmp_db):
    client.post("/review", json={"audit_id": "a_u", "claim_id": "c_u",
                                 "decision": "confirm", "note": ""})
    client.post("/review", json={"audit_id": "a_u", "claim_id": "c_u",
                                 "decision": "reject", "note": "changed mind"})
    rows = client.get("/review", params={"audit_id": "a_u"}).json()
    assert len(rows) == 1 and rows[0]["decision"] == "reject"


def test_review_rejects_invalid_decision(tmp_db):
    r = client.post("/review", json={"audit_id": "a", "claim_id": "c",
                                     "decision": "maybe", "note": ""})
    assert r.status_code == 422
