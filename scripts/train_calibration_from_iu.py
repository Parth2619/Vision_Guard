"""Phase 7 option B — train the calibration MLP on REAL IU Chest X-Ray reports.

Reads `datasets/iu_xray/indiana_reports.csv`, runs the existing rule-based
extractor on each report's FINDINGS + IMPRESSION, weak-labels each claim by
MeSH / Problems match, then trains the 128→64→3 MLP and saves the weights
to `models/calibration_iu.pt`. Runtime loads those weights if present.

Honest limitation: labels come from the IU report's own MeSH/Problems columns
(a weak proxy for truth), and image features are neutral during training —
image-grounded calibration is a later upgrade requiring per-image inference.

Usage (from the project venv):
    python scripts/train_calibration_from_iu.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from medical import Polarity, extract  # noqa: E402
from models.calibration import (  # noqa: E402
    CalibrationMLP, FEATURE_NAMES, STATUS_INDEX,
)

REPORTS_CSV = _ROOT / "datasets" / "iu_xray" / "indiana_reports.csv"
WEIGHTS_OUT = _ROOT / "models" / "calibration_iu.pt"


def _normalize(s: str | float) -> list[str]:
    if not isinstance(s, str):
        return []
    return [t.strip().lower() for t in re.split(r"[;/,]", s) if t.strip()]


def _weak_label(claim, mesh_terms: list[str], problems: list[str]) -> str:
    present_in_truth = any(claim.finding.lower() in t for t in mesh_terms + problems)
    if claim.polarity == Polarity.UNCERTAIN:
        return "indeterminate"
    if claim.polarity == Polarity.NEGATIVE:
        return "supported" if not present_in_truth else "contradicted"
    return "supported" if present_in_truth else "contradicted"


def _features(claim, label: str) -> list[float]:
    """8-dim vector matching models/calibration.FEATURE_NAMES. Image features are
    neutral during training (image-grounded calibration is a later upgrade)."""
    expects_present = claim.polarity != Polarity.NEGATIVE
    has_support = label == "supported"
    has_conflict = label == "contradicted"
    return [
        0.5, 0.0,                                  # classifier (neutral during training)
        0.0, 0.0,                                  # detector  (neutral during training)
        float(claim.polarity == Polarity.NEGATIVE),
        float(claim.polarity == Polarity.UNCERTAIN),
        float(has_support if expects_present else has_support),
        float(has_conflict if expects_present else has_conflict),
    ]


def build_training_set() -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    df = pd.read_csv(REPORTS_CSV)
    xs, ys = [], []
    counts = {"supported": 0, "contradicted": 0, "indeterminate": 0}
    for _, row in df.iterrows():
        mesh = _normalize(row.get("MeSH"))
        problems = _normalize(row.get("Problems"))
        text_parts = []
        for sec in ("findings", "impression"):
            v = row.get(sec)
            if isinstance(v, str) and v.strip():
                text_parts.append(f"{sec.upper()}: {v}")
        if not text_parts:
            continue
        for claim in extract("\n".join(text_parts)):
            label = _weak_label(claim, mesh, problems)
            xs.append(_features(claim, label))
            ys.append(STATUS_INDEX[label])
            counts[label] += 1
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long), counts)


def main() -> int:
    if not REPORTS_CSV.exists():
        print(f"ERROR: {REPORTS_CSV} not found. Run the Kaggle download first.")
        return 2
    x, y, counts = build_training_set()
    if len(x) == 0:
        print("ERROR: produced zero training rows. Check extractor vocab vs IU text.")
        return 2
    print(f"training rows: {len(x)}  label counts: {counts}")

    torch.manual_seed(0)
    model = CalibrationMLP(n_in=len(FEATURE_NAMES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(100):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"final loss={loss.item():.4f}  train accuracy={acc:.3f}")

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"saved weights → {WEIGHTS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
