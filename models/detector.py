"""Zero-shot object detector wrapper. Default: Grounding-DINO (tiny) via HF transformers.

Lazy weight download (~700 MB) on first use into the HF cache. CPU inference is
slow (~5–15 s per image) but functional for the research prototype.

shortcut: single-model, no NMS tuning beyond default thresholds; upgrade by
constructing multiple detectors or raising the box/text thresholds here.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25


@dataclass(frozen=True)
class Detection:
    label: str      # the prompt phrase the box matched
    score: float    # model confidence (NOT a disease probability)
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int


class DetectorBackend(Protocol):
    def detect(self, image_bytes: bytes, prompts: list[str]) -> list[Detection]: ...


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model, torch


class GroundingDinoBackend:
    def detect(self, image_bytes: bytes, prompts: list[str]) -> list[Detection]:
        if not prompts:
            return []
        from PIL import Image

        processor, model, torch = _load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size
        # Grounding-DINO expects a single string of ". "-separated phrases ending with ".".
        text = ". ".join(p.strip().lower() for p in prompts) + "."
        inputs = processor(images=img, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # transformers renamed `box_threshold` → `threshold` in recent releases.
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                target_sizes=[(H, W)],
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                target_sizes=[(H, W)],
            )[0]
        out: list[Detection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            out.append(Detection(label=str(label), score=float(score),
                                 x1=x1, y1=y1, x2=x2, y2=y2,
                                 image_width=W, image_height=H))
        return out


_backend: DetectorBackend | None = None


def set_backend(backend: DetectorBackend | None) -> None:
    global _backend
    _backend = backend


def get_backend() -> DetectorBackend:
    global _backend
    if _backend is None:
        _backend = GroundingDinoBackend()
    return _backend
