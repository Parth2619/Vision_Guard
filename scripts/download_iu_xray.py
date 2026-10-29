"""One-shot IU X-Ray downloader (Stages A + B of docs/dataset_migration_iu_xray.md).

Stage A: NLMCXR_reports.tgz  — ~tens of MB, extracted to datasets/iu_xray/reports/
Stage B: 10 PNGs             — pulled individually via Open-i's image URL pattern

Both targets are git-ignored by the existing .gitignore. Nothing is committed.

Usage (from the project venv, Windows PowerShell):
    python scripts/download_iu_xray.py             # Stage A + 10 PNGs
    python scripts/download_iu_xray.py --n-images 5
    python scripts/download_iu_xray.py --skip-images  # Stage A only
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import httpx  # noqa: E402

from datasets.iu_xray import image_url, load_reports  # noqa: E402

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
OUT_DIR = _ROOT / "datasets" / "iu_xray"
REPORTS_DIR = OUT_DIR / "reports"
IMAGES_DIR = OUT_DIR / "images"


def _download(url: str, dst: pathlib.Path, timeout: float = 300.0) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        total = 0
        with dst.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                total += len(chunk)
        return total


def stage_a() -> None:
    tar_path = OUT_DIR / "NLMCXR_reports.tgz"
    if REPORTS_DIR.exists() and any(REPORTS_DIR.glob("*.xml")):
        print(f"Stage A: already extracted at {REPORTS_DIR} — skipping download")
        return
    print(f"Stage A: downloading {REPORTS_URL} → {tar_path}")
    size = _download(REPORTS_URL, tar_path)
    print(f"  downloaded {size/1_000_000:.1f} MB")
    print(f"  extracting → {REPORTS_DIR}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(REPORTS_DIR, filter="data")
    tar_path.unlink()  # don't keep the archive after extraction
    xmls = list(REPORTS_DIR.rglob("*.xml"))
    print(f"  Stage A done — {len(xmls)} XML files")


def stage_b(n: int) -> None:
    reports = load_reports(REPORTS_DIR)
    if not reports:
        raise SystemExit("Stage B: no reports found; run Stage A first.")
    # Flatten unique parent image ids across the first N reports with at least one image.
    picked: list[str] = []
    for r in reports:
        if len(picked) >= n:
            break
        for pid in r.parent_image_ids:
            if pid not in picked:
                picked.append(pid)
            if len(picked) >= n:
                break
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Stage B: fetching {len(picked)} PNG(s) → {IMAGES_DIR}")
    ok, failed = 0, 0
    for pid in picked:
        dst = IMAGES_DIR / f"{pid}.png"
        if dst.exists():
            print(f"  skip (exists): {pid}")
            ok += 1
            continue
        url = image_url(pid)
        try:
            size = _download(url, dst)
            print(f"  ok ({size/1024:.0f} KB): {pid}")
            ok += 1
        except httpx.HTTPError as exc:
            print(f"  FAILED: {pid} — {type(exc).__name__}: {exc}")
            failed += 1
    print(f"  Stage B done — ok={ok} failed={failed}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--skip-images", action="store_true")
    args = p.parse_args()
    stage_a()
    if not args.skip_images:
        stage_b(args.n_images)
    print(f"\nAll outputs under: {OUT_DIR}")
    print("These paths are git-ignored. Do not commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
