"""Phase 4 (image evidence) + Phase 5 (typed claims) tests. All offline, mocked."""
from __future__ import annotations

import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from api.main import app
from graph.build import run_audit
from medical import Claim, ClaimType, Polarity, Provenance, Status, extract
from medical.evidence import image_evidence
from models.xray import FINDING_TO_LABEL, set_backend

client = TestClient(app)


# ---- Phase 5: typed claim extraction -----------------------------------------

def test_extract_classifies_object_claim():
    [c] = extract("FINDINGS: There is pleural effusion.")
    assert c.claim_type == ClaimType.OBJECT


def test_extract_classifies_attribute_claim():
    [c] = extract("FINDINGS: Small pleural effusion.")
    assert c.claim_type == ClaimType.ATTRIBUTE
    assert c.attribute == "small"


def test_extract_classifies_count_claim():
    [c] = extract("FINDINGS: Three nodules in the right lung.")
    assert c.claim_type == ClaimType.COUNT
    assert c.count == 3


def test_extract_classifies_spatial_claim():
    [c] = extract("FINDINGS: Opacity in the upper lobe.")
    assert c.claim_type == ClaimType.SPATIAL
    assert c.spatial_relation == "upper lobe"


# ---- Phase 4: image evidence provider ----------------------------------------

def _claim(finding="pleural effusion", polarity=Polarity.POSITIVE):
    return Claim(sentence="x", finding=finding, polarity=polarity)


def test_image_evidence_confident_present():
    c = _claim()
    ev = image_evidence(c, {"Effusion": 0.92}, FINDING_TO_LABEL)
    assert len(ev) == 1 and ev[0].agrees is True
    assert ev[0].provenance == Provenance.IMAGE_MODEL
    assert ev[0].confidence == pytest.approx(0.92)


def test_image_evidence_confident_absent():
    ev = image_evidence(_claim(), {"Effusion": 0.05}, FINDING_TO_LABEL)
    assert len(ev) == 1 and ev[0].agrees is False


def test_image_evidence_midrange_emits_nothing():
    assert image_evidence(_claim(), {"Effusion": 0.5}, FINDING_TO_LABEL) == []


def test_image_evidence_unknown_finding_emits_nothing():
    ev = image_evidence(_claim(finding="widget"), {"Effusion": 0.9}, FINDING_TO_LABEL)
    assert ev == []


# ---- Backend stub + end-to-end image audit -----------------------------------

class _StubBackend:
    """Deterministic classifier stub so tests never touch the real weights."""
    def __init__(self, probs):
        self.probs = probs

    def probabilities(self, image_bytes: bytes) -> dict[str, float]:
        assert image_bytes  # must receive non-empty bytes
        return self.probs


def _tiny_png() -> bytes:
    # Minimal 1x1 grayscale PNG so we never embed a real X-ray in the repo.
    sig = b"\x89PNG\r\n\x1a\n"
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\x00"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.fixture
def stub_effusion_confident():
    set_backend(_StubBackend({"Effusion": 0.95}))
    yield
    set_backend(None)


def test_run_audit_with_image_supports_present_effusion(stub_effusion_confident):
    report = run_audit("FINDINGS: Pleural effusion.", mode="normal", image_bytes=_tiny_png())
    v = next(v for v in report.verdicts
             if next(c for c in report.claims if c.claim_id == v.claim_id).finding == "pleural effusion")
    assert v.status == Status.SUPPORTED
    img_ev = [e for e in report.evidence if e.provenance == Provenance.IMAGE_MODEL]
    assert img_ev and img_ev[0].confidence == pytest.approx(0.95)


def test_api_multipart_upload_round_trip(stub_effusion_confident):
    r = client.post(
        "/audit",
        data={"report_text": "FINDINGS: Pleural effusion.", "mode": "normal"},
        files={"image": ("x.png", _tiny_png(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["aggregate"]["supported"] >= 1


def test_api_rejects_non_image_upload():
    r = client.post(
        "/audit",
        data={"report_text": "FINDINGS: Pleural effusion.", "mode": "normal"},
        files={"image": ("x.png", b"not an image", "image/png")},
    )
    assert r.status_code == 415


def test_api_rejects_oversize_image():
    r = client.post(
        "/audit",
        data={"report_text": "FINDINGS: Pleural effusion.", "mode": "normal"},
        files={"image": ("big.png", b"\x89PNG\r\n\x1a\n" + b"0" * (11 * 1024 * 1024), "image/png")},
    )
    assert r.status_code == 413


def test_api_without_image_still_runs():
    r = client.post("/audit", data={"report_text": "", "mode": "synthetic_demo"})
    assert r.status_code == 200
    assert r.json()["mode"] == "synthetic_demo"


def test_api_empty_report_in_normal_mode_returns_json_422():
    """Regression: ValidationError.ctx carried a raw ValueError that broke JSON encoding."""
    r = client.post("/audit", data={"report_text": "", "mode": "normal"})
    assert r.status_code == 422
    body = r.json()  # must be valid JSON, not a 500
    assert any("empty" in e["msg"].lower() for e in body["detail"])


def test_image_classifier_backend_not_loaded_in_tests():
    """Belt-and-braces: no test above should have triggered a real weight download."""
    import sys
    assert "torchxrayvision.models" not in [m for m in sys.modules if "DenseNet" in str(sys.modules.get(m))]
