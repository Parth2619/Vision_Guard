"""Phase 10 — VLM under audit."""
from __future__ import annotations

import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from api.main import app
from graph.build import run_audit
from models.vlm import PrecomputedVLMBackend, StubVLMBackend, get_backend, set_backend
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


def test_stub_vlm_returns_report():
    r = StubVLMBackend().generate_report(b"xx")
    assert "FINDINGS" in r and "IMPRESSION" in r


def test_stub_vlm_rejects_empty_bytes():
    with pytest.raises(ValueError):
        StubVLMBackend().generate_report(b"")


def test_default_backend_auto_selects_precomputed_when_file_exists():
    import pathlib
    from models.vlm import PRECOMPUTED_PATH
    set_backend(None)
    expected = PrecomputedVLMBackend if pathlib.Path(PRECOMPUTED_PATH).exists() else StubVLMBackend
    assert isinstance(get_backend(), expected)


def test_run_audit_vlm_mode_requires_image():
    with pytest.raises(ValueError):
        run_audit("", mode="vlm_from_image", image_bytes=None)


def test_run_audit_vlm_mode_populates_generated_report():
    class _NoopXRay:
        def probabilities(self, b): return {}
    set_xray(_NoopXRay())
    try:
        r = run_audit("", mode="vlm_from_image", image_bytes=_tiny_png())
    finally:
        set_xray(None)
    assert r.input_type == "vlm_generated"
    assert r.generated_report_text and "FINDINGS" in r.generated_report_text
    assert r.claims  # the generated report yields claims


def test_api_vlm_mode_without_image_is_422():
    r = client.post("/audit", data={"report_text": "", "mode": "vlm_from_image"})
    assert r.status_code == 422


def test_api_vlm_mode_with_image_round_trip():
    class _NoopXRay:
        def probabilities(self, b): return {}
    set_xray(_NoopXRay())
    try:
        r = client.post(
            "/audit",
            data={"report_text": "", "mode": "vlm_from_image"},
            files={"image": ("x.png", _tiny_png(), "image/png")},
        )
    finally:
        set_xray(None)
    assert r.status_code == 200
    body = r.json()
    assert body["input_type"] == "vlm_generated"
    assert body["generated_report_text"]


def test_precomputed_backend_matches_by_sha256(tmp_path):
    import hashlib, json
    png = _tiny_png()
    digest = hashlib.sha256(png).hexdigest()
    f = tmp_path / "vlm.json"
    f.write_text(json.dumps({digest: {"report": "FINDINGS: Pleural effusion.",
                                      "image": "x.png"}}))
    backend = PrecomputedVLMBackend(path=f)
    assert backend.generate_report(png) == "FINDINGS: Pleural effusion."


def test_precomputed_backend_falls_back_when_hash_missing(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("{}")
    backend = PrecomputedVLMBackend(path=f)
    # Unknown image → falls back to the stub.
    assert backend.generate_report(_tiny_png()) == StubVLMBackend.REPORT


def test_custom_backend_overrides_default():
    class _MyVLM:
        def generate_report(self, b): return "FINDINGS: Pleural effusion."
    set_backend(_MyVLM())
    try:
        class _NoopXRay:
            def probabilities(self, b): return {}
        set_xray(_NoopXRay())
        try:
            r = run_audit("", mode="vlm_from_image", image_bytes=_tiny_png())
        finally:
            set_xray(None)
    finally:
        set_backend(None)
    assert r.generated_report_text == "FINDINGS: Pleural effusion."
