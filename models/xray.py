"""Pretrained chest X-ray classifier wrapper.

Uses torchxrayvision's `densenet121-res224-all` ensemble (trained on NIH, CheXpert,
MIMIC-CXR, PadChest, OpenI). Weights (~250 MB) download lazily on first use into
`~/.cache/torch/hub/`. CPU inference is a second or two per image.

shortcut: single-model, no ensemble averaging across backbones; upgrade by
constructing multiple models and averaging their outputs here.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Protocol

# Map our extractor's finding vocabulary → torchxrayvision pathology label names.
FINDING_TO_LABEL: dict[str, str] = {
    "pneumonia": "Pneumonia",
    "effusion": "Effusion",
    "pleural effusion": "Effusion",
    "cardiomegaly": "Cardiomegaly",
    "edema": "Edema",
    "consolidation": "Consolidation",
    "nodule": "Nodule",
    "mass": "Mass",
    "opacity": "Lung Opacity",
    "atelectasis": "Atelectasis",
    "pneumothorax": "Pneumothorax",
    "fracture": "Fracture",
    "infiltrate": "Infiltration",
    "emphysema": "Emphysema",
}


class ClassifierBackend(Protocol):
    def probabilities(self, image_bytes: bytes) -> dict[str, float]: ...


@lru_cache(maxsize=1)
def _load_model():
    import torch  # local import so tests can skip loading
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model, xrv, torch


class TorchXRayVisionBackend:
    """Default backend. Downloads weights on first call."""

    def probabilities(self, image_bytes: bytes) -> dict[str, float]:
        from PIL import Image
        import numpy as np

        model, xrv, torch = _load_model()
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.array(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255)  # scale to [-1024, 1024] range xrv expects
        arr = arr[None, ...]
        arr = xrv.datasets.XRayCenterCrop()(arr)
        arr = xrv.datasets.XRayResizer(224)(arr)
        with torch.no_grad():
            tensor = torch.from_numpy(arr).unsqueeze(0)
            out = model(tensor)[0]
        return {name: float(p) for name, p in zip(model.pathologies, out) if name}


_backend: ClassifierBackend | None = None


def set_backend(backend: ClassifierBackend | None) -> None:
    """Install a backend; `None` restores the default lazy loader."""
    global _backend
    _backend = backend


def get_backend() -> ClassifierBackend:
    global _backend
    if _backend is None:
        _backend = TorchXRayVisionBackend()
    return _backend
