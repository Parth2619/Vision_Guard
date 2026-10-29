"""Phase 7 — fusion + calibration MLP (stage 5 of the project diagram).

Features (8) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(3) → softmax.
Trained once at first use on deterministic synthetic feature vectors whose
labels are derived from the rule-based verifier plus small noise. This gives
soft probabilities in place of the rule-based categorical output; it is
calibrated against the rules, NOT against any medical ground truth.

shortcut: training data is self-generated from the rule logic; upgrade to a
hand-labeled set from IU X-Ray (Phase 7 option A) by replacing _synthetic_dataset.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from medical.claims import Claim, Evidence

STATUS_INDEX = {"supported": 0, "contradicted": 1, "indeterminate": 2}
INDEX_STATUS = {v: k for k, v in STATUS_INDEX.items()}

FEATURE_NAMES = (
    "classifier_prob", "classifier_abs_from_midpoint",
    "detector_score", "detector_present",
    "polarity_is_negative", "polarity_is_uncertain",
    "has_independent_support", "has_independent_conflict",
)


class CalibrationMLP(nn.Module):
    def __init__(self, n_in: int = 8, n_out: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def features_for(claim: "Claim", evidence_for_claim: list["Evidence"]) -> list[float]:
    """Pure function: 8 features from a claim + its evidence slice. Deterministic."""
    from medical.claims import Polarity, Provenance

    classifier = [e for e in evidence_for_claim
                  if e.provenance == Provenance.IMAGE_MODEL and e.source.startswith("xray_classifier")]
    detector = [e for e in evidence_for_claim
                if e.provenance == Provenance.IMAGE_MODEL and e.source.startswith("grounding_dino")]
    independent = [e for e in evidence_for_claim
                   if e.provenance in (Provenance.SYNTHETIC, Provenance.IMAGE_MODEL, Provenance.T3_VALIDATED)]
    expects_present = claim.polarity != Polarity.NEGATIVE

    cls_p = classifier[0].confidence if classifier and classifier[0].confidence is not None else 0.5
    det_p = max((d.confidence or 0.0) for d in detector) if detector else 0.0
    return [
        float(cls_p),
        float(abs(cls_p - 0.5)),
        float(det_p),
        float(bool(detector)),
        float(claim.polarity == Polarity.NEGATIVE),
        float(claim.polarity == Polarity.UNCERTAIN),
        float(any(e.agrees == expects_present for e in independent)),
        float(any(e.agrees == (not expects_present) for e in independent)),
    ]


def _synthetic_dataset(n: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 8), generator=g)
    # Binarize the "has" flags; keep probs continuous.
    for col in (3, 4, 5, 6, 7):
        x[:, col] = (x[:, col] > 0.5).float()
    y = torch.full((n,), STATUS_INDEX["indeterminate"], dtype=torch.long)
    for i in range(n):
        has_sup, has_conf = x[i, 6].item(), x[i, 7].item()
        uncertain = x[i, 5].item()
        if uncertain:
            y[i] = STATUS_INDEX["indeterminate"]
        elif has_sup and not has_conf:
            y[i] = STATUS_INDEX["supported"]
        elif has_conf and not has_sup:
            y[i] = STATUS_INDEX["contradicted"]
    # Mild label noise so the softmax isn't a step function on the boundary.
    flip = (torch.rand(n, generator=g) < 0.05)
    y[flip] = torch.randint(0, 3, (int(flip.sum()),), generator=g)
    return x, y


_IU_WEIGHTS = __import__("pathlib").Path(__file__).with_name("calibration_iu.pt")


@lru_cache(maxsize=1)
def get_model() -> CalibrationMLP:
    model = CalibrationMLP()
    if _IU_WEIGHTS.exists():
        # Trained on real IU Chest X-Ray reports; see scripts/train_calibration_from_iu.py
        model.load_state_dict(torch.load(_IU_WEIGHTS, map_location="cpu", weights_only=True))
        model.eval()
        return model
    # Fallback: synthetic dataset so the project still runs if IU data is absent.
    torch.manual_seed(0)
    x, y = _synthetic_dataset(2000)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def calibrate(features: list[float]) -> tuple[str, dict[str, float]]:
    """Return (predicted_status, {status: probability})."""
    model = get_model()
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([features], dtype=torch.float32)), dim=1)[0]
    probs_d = {INDEX_STATUS[i]: float(probs[i]) for i in range(3)}
    status = max(probs_d, key=probs_d.get)
    return status, probs_d
