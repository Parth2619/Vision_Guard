"""Phase 10 — Vision-Language Model under audit (stage 1 of the project diagram).

Produces the "report under audit" from an uploaded image. The pipeline then
audits the VLM's own output end-to-end.

Backends shipped here:
  * StubVLMBackend         — deterministic canned report, no deps, no GPU.
  * PrecomputedVLMBackend  — reads a JSON of {sha256_of_image: {report, ...}}
                              produced by notebooks/vlm_generate_reports.ipynb
                              on Kaggle (Qwen2-VL-7B on a free T4 GPU).
                              Falls through to the stub when the image's hash
                              is not in the precomputed set.

Default lookup uses PrecomputedVLMBackend if the precomputed file exists,
otherwise StubVLMBackend. Override anytime with `set_backend(...)`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

PRECOMPUTED_PATH = Path("datasets/iu_xray/vlm_generated_reports.json")


class VLMBackend(Protocol):
    def generate_report(self, image_bytes: bytes) -> str: ...


class StubVLMBackend:
    """Deterministic canned report. Clearly labeled as synthetic in the trace."""

    REPORT = (
        "FINDINGS: There is a right-sided pleural effusion. The cardiac silhouette "
        "shows cardiomegaly. No pneumothorax is seen. Possible atelectasis in the "
        "left lung.\n"
        "IMPRESSION: Right pleural effusion, likely small. No acute fracture of the ribs."
    )

    def generate_report(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")
        return self.REPORT


class PrecomputedVLMBackend:
    """Serves pre-generated Qwen2-VL reports from a JSON file keyed by SHA-256."""

    def __init__(self, path: Path = PRECOMPUTED_PATH, fallback: VLMBackend | None = None):
        self.path = path
        self.fallback = fallback or StubVLMBackend()
        self._cache: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._cache is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._cache = {}
                return self._cache
            # Accept both {sha: "text"} and {sha: {"report": "text"}} shapes.
            self._cache = {k: (v["report"] if isinstance(v, dict) else v)
                           for k, v in raw.items()}
        return self._cache

    def generate_report(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")
        digest = hashlib.sha256(image_bytes).hexdigest()
        hit = self._load().get(digest)
        if hit is not None:
            return hit
        return self.fallback.generate_report(image_bytes)


_backend: VLMBackend | None = None


def set_backend(backend: VLMBackend | None) -> None:
    global _backend
    _backend = backend


def get_backend() -> VLMBackend:
    global _backend
    if _backend is None:
        _backend = (PrecomputedVLMBackend() if PRECOMPUTED_PATH.exists()
                    else StubVLMBackend())
    return _backend
