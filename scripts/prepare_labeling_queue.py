"""Build a hand-labeling queue of (claim, image) pairs from the downloaded IU data.

Output: datasets/iu_xray/labeling_queue.json  — read by the Streamlit tab.

    python scripts/prepare_labeling_queue.py            # 50 items
    python scripts/prepare_labeling_queue.py --n 100
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medical import extract  # noqa: E402

REPORTS = ROOT / "datasets" / "iu_xray" / "indiana_reports.csv"
PROJ = ROOT / "datasets" / "iu_xray" / "indiana_projections.csv"
IMG_DIR = ROOT / "datasets" / "iu_xray" / "images"
QUEUE_OUT = ROOT / "datasets" / "iu_xray" / "labeling_queue.json"


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    reports = pd.read_csv(REPORTS)
    frontal = pd.read_csv(PROJ)
    frontal = frontal[frontal.projection == "Frontal"].drop_duplicates("uid")
    uid_to_file = dict(zip(frontal.uid, frontal.filename))

    items = []
    for _, row in reports.iterrows():
        if len(items) >= args.n:
            break
        img = uid_to_file.get(row["uid"])
        if not img or not (IMG_DIR / img).exists():
            continue
        text_parts = [f"{s.upper()}: {row[s]}" for s in ("findings", "impression")
                      if isinstance(row.get(s), str)]
        if not text_parts:
            continue
        for claim in extract("\n".join(text_parts))[:1]:  # one claim per image, biggest signal
            items.append({
                "uid": int(row["uid"]),
                "image": img,
                "claim_id": f"iu_{row['uid']}_{len(items)}",
                "sentence": claim.sentence,
                "finding": claim.finding,
                "polarity": claim.polarity.value,
            })
    QUEUE_OUT.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} items → {QUEUE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
