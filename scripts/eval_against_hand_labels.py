"""Measure the pipeline's agreement with hand labels collected in the Streamlit tab.

Reads  datasets/iu_xray/hand_labels.json  and  labeling_queue.json
Runs the full pipeline on each (image, report-fragment) and prints a confusion
matrix + per-status precision/recall/F1 against the hand decisions.

    python scripts/eval_against_hand_labels.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph.build import run_audit  # noqa: E402
from medical.evaluation import _compute_metrics, CaseResult  # noqa: E402

QUEUE = ROOT / "datasets" / "iu_xray" / "labeling_queue.json"
LABELS = ROOT / "datasets" / "iu_xray" / "hand_labels.json"
IMG_DIR = ROOT / "datasets" / "iu_xray" / "images"


def main() -> int:
    if not LABELS.exists():
        print("No hand labels yet. Open the Label Samples tab and click some.")
        return 2
    queue = {it["claim_id"]: it for it in json.loads(QUEUE.read_text(encoding="utf-8"))}
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    results = []
    for claim_id, lab in labels.items():
        if lab.get("decision") == "skipped" or claim_id not in queue:
            continue
        it = queue[claim_id]
        img_path = IMG_DIR / it["image"]
        if not img_path.exists():
            continue
        # Audit a one-sentence "report" so the extractor emits exactly this claim.
        report = run_audit(f"FINDINGS: {it['sentence']}", mode="normal",
                           image_bytes=img_path.read_bytes())
        actual = next((v.status.value for v in report.verdicts
                       if any(c.finding == it["finding"] and c.claim_id == v.claim_id
                              for c in report.claims)), "indeterminate")
        results.append(CaseResult(
            name=claim_id, overall_status=report.overall_hallucination_status,
            per_claim=[{"finding": it["finding"], "expected": lab["decision"],
                        "actual": actual, "matched": actual == lab["decision"]}],
            recheck_count=report.recheck_count, latency_ms=0.0,
            needs_human_review=report.needs_human_review,
        ))
    if not results:
        print("No labeled items with available images.")
        return 2
    m = _compute_metrics(results)
    print(json.dumps({"n_claims": m.n_claims, "accuracy": m.accuracy,
                      "precision": m.precision, "recall": m.recall, "f1": m.f1,
                      "confusion": m.confusion}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
