"""Image-grounded calibration training (Phase 7 refinement).

Uses the real pretrained classifier (torchxrayvision) AND Grounding-DINO on
the ~50 PNGs under datasets/iu_xray/images/ to generate REAL image features.
Combines those rows with the 6514 report-only rows from
train_calibration_from_iu.py (neutral image features) and retrains the MLP.

Honest labels still weak (MeSH / Problems match on each report).

    python scripts/train_calibration_image_grounded.py
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medical import Polarity, extract  # noqa: E402
from models.calibration import (  # noqa: E402
    CalibrationMLP, FEATURE_NAMES, STATUS_INDEX,
)
from models.detector import get_backend as get_detector  # noqa: E402
from models.xray import FINDING_TO_LABEL, get_backend as get_xray  # noqa: E402

REPORTS_CSV = ROOT / "datasets" / "iu_xray" / "indiana_reports.csv"
PROJ_CSV = ROOT / "datasets" / "iu_xray" / "indiana_projections.csv"
IMG_DIR = ROOT / "datasets" / "iu_xray" / "images"
WEIGHTS_OUT = ROOT / "models" / "calibration_iu.pt"


def _normalize(s) -> list[str]:
    if not isinstance(s, str):
        return []
    return [t.strip().lower() for t in re.split(r"[;/,]", s) if t.strip()]


def _weak_label(claim, truth_terms: list[str]) -> str:
    present = any(claim.finding.lower() in t for t in truth_terms)
    if claim.polarity == Polarity.UNCERTAIN:
        return "indeterminate"
    if claim.polarity == Polarity.NEGATIVE:
        return "supported" if not present else "contradicted"
    return "supported" if present else "contradicted"


def _features(claim, probs: dict[str, float] | None,
              dets: list | None, label: str) -> list[float]:
    classifier_p = 0.5
    if probs:
        lab = FINDING_TO_LABEL.get(claim.finding.lower())
        if lab and lab in probs:
            classifier_p = float(probs[lab])
    needle = claim.finding.lower()
    matched = [d for d in (dets or []) if needle in d.label.lower()]
    det_score = max((d.score for d in matched), default=0.0)
    expects_present = claim.polarity != Polarity.NEGATIVE
    has_support = label == "supported"
    has_conflict = label == "contradicted"
    return [
        classifier_p,
        abs(classifier_p - 0.5),
        det_score,
        float(bool(matched)),
        float(claim.polarity == Polarity.NEGATIVE),
        float(claim.polarity == Polarity.UNCERTAIN),
        float(has_support if expects_present else has_support),
        float(has_conflict if expects_present else has_conflict),
    ]


def build_rows():
    reports = pd.read_csv(REPORTS_CSV)
    proj = pd.read_csv(PROJ_CSV)
    # map uid -> first frontal filename (if any image was downloaded)
    frontal = proj[proj.projection == "Frontal"].drop_duplicates("uid")
    uid_to_file = dict(zip(frontal["uid"], frontal["filename"]))

    xs, ys, counts = [], [], {"supported": 0, "contradicted": 0, "indeterminate": 0}
    image_rows_added = 0
    xray_backend = get_xray()
    detector_backend = get_detector()

    for i, row in enumerate(reports.itertuples(), 1):
        truth = _normalize(row.MeSH) + _normalize(row.Problems)
        text_parts = [f"{sec.upper()}: {getattr(row, sec)}"
                      for sec in ("findings", "impression")
                      if isinstance(getattr(row, sec, None), str)]
        if not text_parts:
            continue
        claims = extract("\n".join(text_parts))
        if not claims:
            continue

        probs, dets = None, None
        img_file = uid_to_file.get(row.uid)
        img_path = IMG_DIR / img_file if img_file else None
        if img_path and img_path.exists():
            img_bytes = img_path.read_bytes()
            t0 = time.perf_counter()
            try:
                probs = xray_backend.probabilities(img_bytes)
                prompts = sorted({c.finding for c in claims})
                dets = detector_backend.detect(img_bytes, prompts)
                image_rows_added += 1
                print(f"  img [{image_rows_added}] uid={row.uid} "
                      f"claims={len(claims)} dt={time.perf_counter()-t0:.1f}s")
            except Exception as exc:  # noqa: BLE001
                print(f"  img uid={row.uid} FAILED: {type(exc).__name__}: {exc}")
                probs, dets = None, None

        for claim in claims:
            label = _weak_label(claim, truth)
            xs.append(_features(claim, probs, dets, label))
            ys.append(STATUS_INDEX[label])
            counts[label] += 1

    print(f"\ntotal rows: {len(xs)}  image-grounded rows: {image_rows_added}  "
          f"label counts: {counts}")
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def main() -> int:
    if not REPORTS_CSV.exists():
        print("Run scripts/download_iu_xray.py (or the Kaggle fetch) first.")
        return 2
    x, y = build_rows()
    torch.manual_seed(0)
    model = CalibrationMLP(n_in=len(FEATURE_NAMES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"final loss={loss.item():.4f}  train accuracy={acc:.3f}")
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"saved → {WEIGHTS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
