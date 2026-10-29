> **DEPRECATED (Phase 2G)**: project migrated to IU Chest X-Ray. See [dataset_migration_iu_xray.md](dataset_migration_iu_xray.md). This file is kept for history.

# Dataset Evidence Audit — RadGenome-ChestCT

**Date:** 2026-10-10  **Phase:** 2D  **Status:** Audit only, no downloads of CT volumes or masks.

## 1. Sources consulted

- Hugging Face dataset repository `RadGenome/RadGenome-ChestCT` (file listing via `HfApi.list_repo_files` — 25,723 files).
- The three dataset configs exposed by HF `datasets`: `grounded reports`, `grounded vqa`, `case-level vqa`.
- One streamed record from each config (metadata only).

Not consulted directly in this phase: the published paper (not fetched), the full `README.md` of the dataset repo (not fetched — only file list retrieved). Any claim derived from them below is marked **UNVERIFIED**.

## 2. Repository layout (verified by file listing)

```
RadGenome-ChestCT/
├── README.md                             # UNVERIFIED body
├── processed_code/                       # preprocess_ctrate_{train,valid}.py
└── dataset/
    ├── qa_templates/                     # 5 JSON prompt templates
    ├── radgenome_files/                  # 12 CSVs — the report/VQA metadata
    │   ├── train_region_report.csv       # (very likely the "grounded reports" source)
    │   ├── train_vqa_abnormality.csv
    │   ├── train_vqa_{location,presence,size}.csv
    │   ├── train_case_disorders.csv
    │   └── validation_* (mirror of the above)
    ├── report_hierarchy_tree.zip         # anatomy hierarchy (not inspected)
    ├── train_preprocessed/               # 24,129 .nii.gz CT volumes
    ├── valid_preprocessed/               #  1,564 .nii.gz CT volumes
    └── train_anatomy_mask_{aa,ab,ac,ad,ae}   # split archive parts of anatomy masks
```

The `train_anatomy_mask_a?` files are **split-archive parts** (classic `split -b`); they must be concatenated before extraction. **Not** downloaded in this phase.

## 3. What the metadata actually contains

### 3.1 `grounded reports` (sampled live)
| Field | Type | Example | Notes |
|---|---|---|---|
| `Volumename` | str | `train_16242_a_1.nii.gz` | Points at a file under `dataset/train_preprocessed/.../`. |
| `Anatomy` | str \| None | `lung` / `None` | Null in some rows (seen 4/100 missing in Phase 2C sample). |
| `Sentence` | str | *"The widths of the mediastinal main vascular structures are normal…"* | Free-text report sentence. |

**No mask path, no voxel coordinates, no slice indices, no bounding box, no mask label id.** Grounding is **implicit by anatomy name**, not explicit.

### 3.2 `grounded vqa` (sampled live)
| Field | Type | Example |
|---|---|---|
| `Volumename` | str | `train_16242_a_1.nii.gz` |
| `Anatomy` | str | `lung` |
| `Abnormality` | str | `millimetric nonspecific nodules, emphysematous changes, linear atelectasis` |

Same shape as grounded reports (volume + anatomy label), with a comma-joined abnormality list in place of a free-text sentence.

### 3.3 `case-level vqa`
The HF `datasets` builder **fails to load** this config: it tries to open `dataset/radgenome_files/calidation_case_disorders.csv` (note the typo — `calidation` instead of `validation`) and the file does not exist. This is an upstream bug in the loader script, not an access issue; it will prevent using this config until the maintainers fix it or we read the underlying CSV directly.

## 4. Volume / anatomy / mask identifiers — what IS known vs. UNKNOWN

| Question | Finding | Status |
|---|---|---|
| Field identifying a CT volume | `Volumename` (filename stem of `.nii.gz` under `*_preprocessed/`) | **Verified** |
| Field identifying an anatomical region | `Anatomy` (free-text label, e.g. `lung`) | **Verified** |
| Per-row segmentation-mask path | None in any sampled row | **Verified absent** |
| Per-row coordinates / slice refs | None in any sampled row | **Verified absent** |
| Where masks are stored | `dataset/train_anatomy_mask_a?` split-archive parts | **Verified by file list** |
| Mapping `Anatomy` string → mask label id | Not in the row metadata; presumed defined inside the mask archive or in the paper's convention (likely a TotalSegmentator-style labeling, since the dataset is derived from CT-RATE) | **UNVERIFIED** |
| Validation masks | No `valid_anatomy_mask_*` entry seen in file listing | **UNVERIFIED** (may live inside the train archives or elsewhere) |

## 5. "Linked" ≠ "verified"

A `grounded reports` row says: *"this sentence was emitted about the anatomy `lung` in volume `train_16242_a_1`"*. It does **not** say:

- that the sentence is clinically correct;
- that the sentence's claim is visible in the referenced anatomy;
- that any human adjudicated the claim against the pixels.

For VisionGuard's `supported` / `contradicted` / `indeterminate` labels, a `grounded reports` row is at best a **source caption**, not a label. Treating a grounded sentence as a positive "supported" example would re-use the model's own report-generation signal as truth — a circular evaluation.

## 6. Is the metadata alone enough for visual verification?

**No.** With only `(Volumename, Anatomy, Sentence)`:
- A text-only claim-verification agent could compare a generated report to the reference `Sentence` for the same `(Volumename, Anatomy)` pair — but that is textual entailment, not visual verification. It cannot distinguish a visually unsupported claim from one the reference simply omitted.
- A visual-evidence agent needs pixels. Minimum additional assets:
  1. The corresponding `.nii.gz` CT volume (large: tens to hundreds of MB per volume).
  2. The anatomy mask for that volume (requires reassembling the split archive parts).
  3. The `Anatomy` → mask-label-id mapping convention.
  4. (Nice-to-have) The `report_hierarchy_tree.zip` for anatomy normalization.

All four live inside the same HF repo and are **publicly listable without auth**. Downloading any CT volume or mask requires explicit opt-in — not performed in this phase.

## 7. Recommendation

**Obtain a small, authorized CT-and-mask sample before implementing visual verification.**

Rationale:
- Metadata alone cannot support the `supported` / `contradicted` label distinction without smuggling the generator's own captions in as ground truth.
- The split-archive mask layout and the undocumented anatomy→label mapping are both non-trivial — they need to be resolved on a handful of real volumes before any visual agent is designed, not after.
- A text-only path (claim extraction + textual entailment against grounded sentences) is still viable as a **separate, clearly-labeled baseline**, but it must not be presented as visual verification.

### Concrete minimum next-phase assets (user approval required before downloading)
- 3–5 `.nii.gz` volumes from `dataset/valid_preprocessed/` (smallest split, 1,564 volumes total).
- The matching anatomy masks for those 3–5 volumes (requires reassembling split archive parts — defer deciding on download size until the pack-size per volume is confirmed).
- `report_hierarchy_tree.zip` (small, safe to grab).
- The `validation_region_report.csv` rows that reference those volumes.

Total disk footprint: estimate under 2 GB; confirm before pulling.

## 8. Open items / blockers

- `case-level vqa` config is broken upstream (`calidation_case_disorders.csv` typo). File an issue on the HF repo or read the CSV directly.
- Validation anatomy masks: location not confirmed from the file list alone.
- Anatomy label vocabulary and `Anatomy → mask-channel` mapping: unverified. Needs the README body or an actual mask file.
- Nothing in the audited metadata is PHI, but once we start pulling volumes we are handling DICOM-derived data — handle locally only, keep out of git (already enforced by `.gitignore`).
