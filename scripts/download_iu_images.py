"""Download ~N frontal IU X-Ray PNGs via the Kaggle CLI (one file at a time).

Picks the first N unique reports that have a frontal projection, downloads each
PNG into datasets/iu_xray/images/, and unwraps the Kaggle single-file zips.

Usage:
    python scripts/download_iu_images.py                 # default N=50
    python scripts/download_iu_images.py --n 100
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJ_CSV = ROOT / "datasets" / "iu_xray" / "indiana_projections.csv"
IMG_DIR = ROOT / "datasets" / "iu_xray" / "images"
DATASET = "raddar/chest-xrays-indiana-university"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    proj = pd.read_csv(PROJ_CSV)
    pick = (proj[proj["projection"] == "Frontal"]
            .drop_duplicates("uid").head(args.n))
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(pick.itertuples(), 1):
        fname = row.filename
        dst = IMG_DIR / fname
        if dst.exists():
            print(f"  [{i}/{len(pick)}] skip (exists): {fname}")
            continue
        src_in_dataset = f"images/images_normalized/{fname}"
        print(f"  [{i}/{len(pick)}] fetch: {fname}")
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", DATASET, "-f", src_in_dataset, "-p", str(IMG_DIR), "--force"],
            check=True, capture_output=True,
        )
        # Kaggle wraps single-file downloads in <filename>.zip; unwrap it.
        zpath = IMG_DIR / f"{fname}.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(IMG_DIR)
            zpath.unlink()
    print(f"\n{len(list(IMG_DIR.glob('*.png')))} PNGs in {IMG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
