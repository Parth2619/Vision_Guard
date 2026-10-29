"""Phase 6 detector + bbox tests. All offline, backend stubbed."""
from __future__ import annotations

import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from api.main import app
from graph.build import run_audit
from medical import Claim, Polarity, Provenance, Status
from medical.evidence import detection_evidence
from models.detector import Detection, set_backend as set_detector
from models.xray import set_backend as set_xray

client = TestClient(app)


def _tiny_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\x00"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


class _StubDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image_bytes, prompts):
        assert image_bytes
        assert prompts  # graph must pass something
        return self.detections


class _StubXRay:
    def probabilities(self, image_bytes):
        return {"Effusion": 0.5}  # mid-range → no classifier evidence


@pytest.fixture
def stub_backends(request):
    set_detector(_StubDetector(request.param))
    set_xray(_StubXRay())
    yield
    set_detector(None)
    set_xray(None)


# ---- Unit: detection_evidence ------------------------------------------------

def test_detection_evidence_matches_finding_and_carries_bbox():
    c = Claim(sentence="x", finding="pleural effusion")
    det = [Detection(label="pleural effusion", score=0.82,
                     x1=10, y1=20, x2=100, y2=200, image_width=224, image_height=224)]
    [ev] = detection_evidence(c, det)
    assert ev.provenance == Provenance.IMAGE_MODEL
    assert ev.agrees is True
    assert ev.bbox and (ev.bbox.x1, ev.bbox.y1, ev.bbox.x2, ev.bbox.y2) == (10, 20, 100, 200)
    assert ev.confidence == pytest.approx(0.82)


def test_detection_evidence_contradicts_negated_claim():
    c = Claim(sentence="x", finding="pneumothorax", polarity=Polarity.NEGATIVE)
    det = [Detection(label="pneumothorax", score=0.9,
                     x1=0, y1=0, x2=1, y2=1, image_width=10, image_height=10)]
    [ev] = detection_evidence(c, det)
    assert ev.agrees is False  # "no pneumothorax" + detector found it → disagrees


def test_detection_evidence_ignores_unrelated_labels():
    c = Claim(sentence="x", finding="pleural effusion")
    det = [Detection(label="cardiomegaly", score=0.9,
                     x1=0, y1=0, x2=1, y2=1, image_width=10, image_height=10)]
    assert detection_evidence(c, det) == []


def test_detection_evidence_picks_highest_scoring_match():
    c = Claim(sentence="x", finding="nodule")
    det = [
        Detection(label="nodule", score=0.4, x1=0, y1=0, x2=1, y2=1, image_width=10, image_height=10),
        Detection(label="nodule", score=0.9, x1=5, y1=5, x2=9, y2=9, image_width=10, image_height=10),
    ]
    [ev] = detection_evidence(c, det)
    assert ev.confidence == pytest.approx(0.9)
    assert ev.bbox.x1 == 5


# ---- Integration: graph run with stubbed detector ----------------------------

@pytest.mark.parametrize("stub_backends", [[
    Detection(label="pleural effusion", score=0.88,
              x1=30, y1=40, x2=200, y2=180, image_width=224, image_height=224)
]], indirect=True)
def test_graph_includes_detector_evidence(stub_backends):
    report = run_audit("FINDINGS: Pleural effusion.", mode="normal", image_bytes=_tiny_png())
    det_ev = [e for e in report.evidence if e.source.startswith("grounding_dino")]
    assert det_ev and det_ev[0].bbox is not None
    verdict = next(v for v in report.verdicts
                   if next(c for c in report.claims if c.claim_id == v.claim_id).finding == "pleural effusion")
    assert verdict.status == Status.SUPPORTED  # detector agrees → promotes to supported


@pytest.mark.parametrize("stub_backends", [[]], indirect=True)
def test_graph_no_detections_keeps_claim_indeterminate(stub_backends):
    report = run_audit("FINDINGS: Pleural effusion.", mode="normal", image_bytes=_tiny_png())
    # Classifier stub returns 0.5 (mid-range → no classifier evidence); detector returns nothing.
    assert all(v.status == Status.INDETERMINATE for v in report.verdicts)


# ---- Integration: API multipart with stubbed detector ------------------------

@pytest.mark.parametrize("stub_backends", [[
    Detection(label="cardiomegaly", score=0.75,
              x1=50, y1=60, x2=150, y2=160, image_width=224, image_height=224)
]], indirect=True)
def test_api_returns_bbox_in_response(stub_backends):
    r = client.post(
        "/audit",
        data={"report_text": "FINDINGS: Cardiomegaly.", "mode": "normal"},
        files={"image": ("x.png", _tiny_png(), "image/png")},
    )
    assert r.status_code == 200
    bboxes = [e["bbox"] for e in r.json()["evidence"] if e.get("bbox")]
    assert bboxes and bboxes[0]["x1"] == 50
