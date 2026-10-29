> **DEPRECATED (Phase 2G)**: project migrated to IU Chest X-Ray. See [dataset_migration_iu_xray.md](dataset_migration_iu_xray.md). This file is kept for history.

# Sample Download Plan — RadGenome-ChestCT

**Date:** 2026-10-10 (updated Phase 2F)  **Status:** Plan only. **No CT volumes or masks downloaded.** User approval required before any download.

Follows the Phase 2D recommendation ([docs/dataset_evidence_audit.md](dataset_evidence_audit.md)) to acquire a small CT-and-mask sample before implementing visual verification. Phase 2F added documentation findings from the HF repo `README.md`, confirmed license/gating terms, and documented Kaggle feasibility without downloading data.

## 0. Verified from the official HF `README.md` (fetched Phase 2F)

Only `README.md` (5.5 KB, text metadata — not CT data) was fetched. Verified:

- **License**: `cc-by-4.0` (declared in YAML front matter).
- **Access is gated** via HF's `extra_gated_prompt` with Name / Institution / Email + a checkbox acknowledging the CT-RATE-copied terms. **Streaming of the metadata parquet views still worked unauthenticated in Phase 2B/2C**, but the gate is likely enforced on LFS downloads (CT volumes, mask archives). **UNVERIFIED whether unauthenticated LFS fetches succeed** — do not test this before step 1 of the sample.
- **Permitted use**: academic / research / educational only. **Commercial use forbidden** without prior permission.
- **Redistribution forbidden**. Do not re-host CT volumes, masks, or derivatives (including on Kaggle datasets, GitHub, or in `git` for this project). Our `.gitignore` already blocks `*.nii.gz`, `datasets/**`, `artifacts/**`.
- **Attribution**: cite [RadGenome-Chest CT paper (arXiv:2404.16754)](https://arxiv.org/pdf/2404.16754) and [CT-CLIP (arXiv:2403.17834)](https://arxiv.org/abs/2403.17834).
- **Scale (stated by the paper)**: 25,692 non-contrast 3D chest CT volumes from 20,000 patients; 197 organ-level segmentation categories; 665 K grounded report sentences; 1.3 M grounded VQA pairs.
- **IMPORTANT**: *"All grounded reports and VQA pairs in the validation set have gone through manual verification to ensure dataset quality."* → validation `Sentence`s are human-adjudicated against the corresponding anatomical region. For VisionGuard this upgrades them from "source captions" to "human-approved region-level descriptions" — a usable but still textual grounding. It does **not** make them pixel-level supported/contradicted labels.
- **No separate `LICENSE` file** in the repo. Terms live entirely in the README front matter.
- **`case-level vqa` typo confirmed**: the README itself declares the validation path as `calidation_case_disorders.csv`, which does not exist. The config is broken upstream, not just mis-loaded.

The README does **not** document the internal filename convention inside `valid_anatomy_mask.tar.gz` or the mapping from the 197 anatomy categories to mask label ids. That mapping remains **UNVERIFIED** until the first extraction.

Paper itself (arXiv PDF) was **not** fetched in this phase to avoid pulling a large external PDF. The abstract (quoted above from the README) was sufficient to answer this phase's questions.

## 1. Verified repository paths and sizes

All sizes retrieved via `HfApi.get_paths_info` against `RadGenome/RadGenome-ChestCT` on 2026-10-10. All listed files are LFS-backed in the HF repo.

| Path | Size | Verified? | Notes |
|---|---:|---|---|
| `dataset/report_hierarchy_tree.zip` | 102,260,274 B (≈ **97.5 MiB**) | ✅ HF `get_paths_info` | Anatomy hierarchy, used for region normalization. |
| `dataset/radgenome_files/validation_region_report.csv` | 9,616,073 B (≈ **9.17 MiB**) | ✅ | Validation split of the "grounded reports" CSV. |
| `dataset/valid_anatomy_mask.tar.gz` | 10,504,007,274 B (≈ **9.78 GiB**) | ✅ | **Monolithic tarball** for the entire validation anatomy-mask set. |
| `dataset/valid_region_mask.tar.gz` | 2,165,981,473 B (≈ **2.02 GiB**) | ✅ | Monolithic tarball for validation region masks. |
| `dataset/valid_preprocessed/valid_1/valid_1a/valid_1_a_1.nii.gz` | 26,882,653 B (≈ 25.6 MiB) | ✅ | Volume candidate A. |
| `dataset/valid_preprocessed/valid_10/valid_10a/valid_10_a_1.nii.gz` | 27,627,753 B (≈ 26.3 MiB) | ✅ | Volume candidate B. |
| `dataset/valid_preprocessed/valid_100/valid_100a/valid_100_a_1.nii.gz` | 65,882,828 B (≈ 62.8 MiB) | ✅ | Volume candidate C. |
| `dataset/train_anatomy_mask_{aa..ae}` | ~16.1 GB × 4 + 13.9 GB ≈ **~77.9 GB total** | ✅ | **Split-archive parts** (`split -b`). All parts must be downloaded and concatenated before extraction. Avoid for a sample. |

> UNVERIFIED: whether the HF LFS "size" field reflects the exact byte count returned over HTTP, or the pre-dedup logical size. Treat the numbers as accurate to within a few percent for budgeting.

## 2. Candidate volumes and mask-to-volume mapping

**Three validation candidates**, chosen for size and for the fact that validation masks ship as a single tarball (train masks require ~78 GB of split parts, which is disqualifying for a sample):

| Volume (verified path) | Volume size |
|---|---:|
| `dataset/valid_preprocessed/valid_1/valid_1a/valid_1_a_1.nii.gz` | 25.6 MiB |
| `dataset/valid_preprocessed/valid_10/valid_10a/valid_10_a_1.nii.gz` | 26.3 MiB |
| `dataset/valid_preprocessed/valid_100/valid_100a/valid_100_a_1.nii.gz` | 62.8 MiB |

**How the volume → mask mapping is established:**

- Volumes live under `dataset/valid_preprocessed/valid_<N>/valid_<N>a/valid_<N>_a_1.nii.gz` — **verified** by file listing.
- Anatomy masks live inside `dataset/valid_anatomy_mask.tar.gz` — **verified** that the tarball exists; the mapping between a given volume filename and the mask file(s) inside the tarball is **UNVERIFIED without extracting**. Standard practice in CT-RATE-derived datasets is a mirrored layout (`valid_<N>/valid_<N>a/valid_<N>_a_1.nii.gz` → one or more mask `.nii.gz` files under the same path, or a single multi-label mask with the same name). This expectation must be validated against the actual archive contents on first extraction.
- Grounded-report metadata linking an anatomy label to a volume is established via the `Volumename` column of `validation_region_report.csv` (same column we verified in `grounded reports` live-streaming — **verified**).

## 3. Can individual files be fetched independently?

| Asset class | Independent file access? |
|---|---|
| `.nii.gz` volume files under `*_preprocessed/` | **Yes.** Each volume is its own LFS file at a known HF path. |
| `report_hierarchy_tree.zip`, hierarchy & template files | **Yes.** |
| `radgenome_files/*.csv` | **Yes.** |
| Validation anatomy / region masks | **No.** Only the full `valid_anatomy_mask.tar.gz` (9.78 GiB) and `valid_region_mask.tar.gz` (2.02 GiB) are exposed. There is **no way to fetch a single volume's mask without downloading the entire tarball** (short of HTTP range requests into the tar stream, which is fragile and not supported by the HF client). |
| Train anatomy masks | **No, and worse.** Only split-archive parts (five files, ~78 GB total) are exposed. All must be downloaded and concatenated before `tar` can read them. |

## 4. Proposed minimum viable sample

Objective: demonstrate (a) volume load, (b) mask load, (c) volume/mask alignment check, (d) link anatomy region → grounded report row.

### Recommended minimum
| File | Verified size | Purpose |
|---|---:|---|
| `dataset/radgenome_files/validation_region_report.csv` | 9.17 MiB | Row-level grounded-report metadata for linkage. |
| `dataset/report_hierarchy_tree.zip` | 97.5 MiB | Anatomy label normalization. |
| `dataset/valid_preprocessed/valid_1/valid_1a/valid_1_a_1.nii.gz` | 25.6 MiB | Volume A — load + geometry check. |
| `dataset/valid_preprocessed/valid_10/valid_10a/valid_10_a_1.nii.gz` | 26.3 MiB | Volume B — second case for alignment sanity. |
| `dataset/valid_preprocessed/valid_100/valid_100a/valid_100_a_1.nii.gz` | 62.8 MiB | Volume C — larger case, catches edge cases. |
| `dataset/valid_anatomy_mask.tar.gz` | **9.78 GiB** | **Only path to any validation anatomy mask.** Downloaded once, we extract **only** the three mask entries matching the three volumes and discard the rest. |

**Total download: ≈ 9.98 GiB** (dominated by the mask tarball).
**Peak disk use during extraction: ≈ 10 GiB** (the tarball itself; extracted mask files are small). The tarball can be deleted after extracting the three mask paths.

### Why not smaller
- The validation anatomy-mask tarball is the smallest mask artifact in the repo. Train masks are ~78 GB of split parts. **There is no "small masks" asset.**
- Dropping masks and using only the volumes + grounded sentences gets us a text-only pipeline, not visual verification — which is exactly what the Phase 2D audit called insufficient.

### Optional add-on (skip for Phase 2F)
- `dataset/valid_region_mask.tar.gz` (2.02 GiB) — region-level masks. Defer until anatomy masks are proven workable.

## 5. License and access considerations (verified Phase 2F)

See §0 — license is CC-BY-4.0, access gated, redistribution forbidden, research-only. Before downloading, complete the HF gate form on the dataset page (do that locally in a browser; never paste tokens in chat). After acceptance, authenticate with `huggingface-cli login`.

**PHI**: CT-RATE is de-identified upstream, but VisionGuard still treats this as medical data — kept local, excluded from git.

## 5b. Kaggle feasibility

Legitimate path on Kaggle, respecting the no-redistribution clause:

- **Do not** create a Kaggle Dataset that mirrors the HF files. That would violate §5 (redistribution).
- **Do** run a Kaggle notebook that authenticates to HF with a user secret (`HF_TOKEN` set via Kaggle Secrets, never in the notebook body) and downloads directly from the official HF repo during the session.
- The notebook author must have personally been granted access via the HF gate form.
- Outputs that could re-expose source data (full tarball extractions, raw NIfTI volumes attached as a Kaggle dataset output) **must not** be published.

**UNVERIFIED here**: current Kaggle Python-kernel disk and RAM quotas. Kaggle's quotas have changed over time and are not fetched in this phase. Widely-documented recent limits (as of our model's knowledge, not re-verified now) were roughly 20 GB on `/kaggle/working` and larger ephemeral scratch on `/tmp` or `/kaggle/temp`, which would comfortably fit a 9.78 GiB tarball + extraction. **Confirm current quotas by starting a Kaggle notebook and inspecting `df -h` and `/proc/meminfo` before relying on them.** Do not pre-allocate work there based on assumed quotas.

Pros vs. local Windows: Kaggle gives throwaway disk and faster bandwidth to HF; nothing persists after the session unless explicitly saved. For a one-off "extract three masks from the tarball" job, this is attractive. Cons: must bring the HF token in via Kaggle Secrets; results need to be exfiltrated as a very small, non-redistributive artifact (e.g. hashes, shapes, affines — not the images themselves).

## 6. Remaining uncertainties and risks

| # | Item | Impact | Mitigation |
|---|---|---|---|
| 1 | Mask filename convention inside the tarball not verified | A download may land, but we may not know which entry maps to which volume | Validate on first extraction with a dry `tar -tzf` listing; abort/replan if mapping is non-obvious. |
| 2 | Whether volume `.nii.gz` LFS downloads are gated | Could block the plan after allocating disk | Attempt one small volume first (`valid_1_a_1.nii.gz`, 25.6 MiB). If 401/403, log in via HF CLI and retry. |
| 3 | License / attribution requirements not confirmed in this phase | Compliance risk | Read `README.md` + `LICENSE` of the HF repo before download; record the citation in `docs/`. |
| 4 | Volume ↔ mask spatial alignment (orientation, spacing, affine) not guaranteed to be identical even if filenames match | Alignment check in Phase 2F could reveal mismatch, needing resampling | Design Phase 2F to compute shapes/affines explicitly and report mismatches rather than assume. |
| 5 | 9.78 GiB tarball is a lot of bandwidth and disk relative to using 3 masks | Cost | Download once, extract only the three needed mask paths, delete the tarball immediately. Consider acquiring on a Kaggle environment with more disk and discarding the tarball at the end of the session. |

## 7. Smallest feasible experiment — staged plan

Phase 2F revises the approval ask into **two sequential downloads** rather than one. Smallest-first keeps the commitment reversible.

### Stage A — single-volume feasibility (approve first)
Download, in this order:

1. `dataset/radgenome_files/validation_region_report.csv` — 9.17 MiB (metadata only).
2. `dataset/report_hierarchy_tree.zip` — 97.5 MiB (metadata only).
3. `dataset/valid_preprocessed/valid_1/valid_1a/valid_1_a_1.nii.gz` — **25.6 MiB** (one CT volume).
4. `dataset/valid_anatomy_mask.tar.gz` — **9.78 GiB** (the only way to obtain any validation mask).

Then: extract **only** the mask entry matching `valid_1_a_1` from the tarball and delete the tarball. Run the alignment checks in §8. **Stop and report.**

Approximate working set: 9.9 GiB peak, dropping to ≈ 160 MiB after cleanup.

### Stage B — three-volume sample (approve only after Stage A succeeds)
Add volumes `valid_10` and `valid_100` (26.3 + 62.8 MiB). The 9.78 GiB tarball is already consumed and discarded — Stage B is text and `~90 MiB` of volume files only, no further mask download.

If Stage A fails (401/403 on LFS, mapping mismatch, alignment issue), Stage B is not attempted.

## 8. Alignment checks required for CT + mask pairs

For each (volume, mask) pair, before using them for anything:

1. **Shape match**: volume `.shape == mask.shape` on all three spatial axes (ignoring any trailing channel axis if the mask stores labels in a 4th dim). Mismatch → stop; no automatic resampling without a human review.
2. **Affine / orientation match**: NIfTI affines (`nib.load(...).affine`) within a small tolerance (per-element diff ≤ 1e-3). Catches silent LPS/RAS flips and spacing drift.
3. **Voxel spacing match**: `header.get_zooms()[:3]` within tolerance. A mismatch here while the shape matches is almost always a bug.
4. **Label-set sanity**: unique mask label ids are integers within `[0, 197]` (per README: 197 organ categories). Values outside this range → stop and audit.
5. **Non-empty mask**: the mask has at least one non-zero voxel for the volume (an all-zero mask is a signal that the mapping is wrong).
6. **Filename-to-row linkage**: the volume filename appears as a `Volumename` value in `validation_region_report.csv`. If it doesn't appear, the sample case is unlinkable for grounded-report evaluation — pick another.
7. **Anatomy-name → label-id mapping**: still **UNVERIFIED**; part of what Stage A exists to discover. Expected: either (a) a documentation file inside the tarball, or (b) the convention from the paper's cited universal segmentation model.

Checks are read-only: implement them as functions that return a report dict, not assertions, so a mismatch produces a diagnosable artifact rather than a crash.

## 9. Approval requested

Before Phase 2G downloads anything, confirm:

1. **Approve Stage A** — ~9.9 GiB download, one volume + full validation anatomy-mask tarball + two metadata files, with the tarball deleted after extracting one mask entry.
2. **Pick a run location** — local Windows (my disk has it, no new infra) or a Kaggle notebook with your HF token in Kaggle Secrets (throwaway disk; must not publish any outputs that redistribute source data).
3. **Confirm you have completed the HF gate form** on the dataset page under your HF account. If not, do that first.
4. **Hold Stage B** approval until Stage A reports successful alignment checks.

**No files will be downloaded until these are confirmed.**
