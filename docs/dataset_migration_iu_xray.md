# Dataset Migration — RadGenome-ChestCT → IU Chest X-Ray (Open-i)

**Date:** 2026-10-10  **Status:** Plan + migration decision. **No files downloaded.** User approval required before any download.

## 1. Why we're migrating

RadGenome-ChestCT is being set aside for VisionGuard Phase 2+ because:

- Access is gated by a manual HF form plus CT-RATE terms copied into the dataset README.
- The validation anatomy masks ship **only** as a single 9.78 GiB tarball — no per-volume selective access (see [docs/sample_download_plan.md](sample_download_plan.md)).
- Train masks ship as a 77.9 GB five-part split archive, disqualifying them for an iterative sample.
- The `case-level vqa` config is broken upstream (`calidation_case_disorders.csv` typo).

RadGenome Phases 2A–2F documentation is **kept** for history; code paths are **deprecated** (see §8).

## 2. Replacement — Indiana University Chest X-Ray (NLM Open-i)

- Official info: https://openi.nlm.nih.gov/faq
- Images archive: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
- Reports archive: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

Properties relevant to VisionGuard:

- 2D chest X-rays (PA + lateral per case). Much smaller per-case footprint than CT volumes.
- Reports are structured XML with named sections (Findings, Impression, Indication, Comparison) + MeSH tags.
- Each report lists `parentImage` ids that match the PNG basenames — explicit, documented, per-row linkage.
- Public, no gate form, no manual approval delay.

### UNVERIFIED here (confirm against the live FAQ before downloading)
WebFetch returned empty content for `openi.nlm.nih.gov/faq` and `openi.nlm.nih.gov/` from this environment (JS-rendered page, likely). The following items are from widely-published, longstanding documentation of this collection but were **not** re-verified live in this phase:

- Exact dataset size (commonly cited as **~3,955 reports and ~7,470 images** — Demner-Fushman et al., JAMIA 2016).
- The precise XML tag names (commonly cited as `AbstractText` elements with `Label="FINDINGS" | "IMPRESSION" | "INDICATION" | "COMPARISON"`, plus `MeSH/major` and `MeSH/automatic`, and `parentImage id="..."`).
- Whether the two archives are the only official distribution channel vs. an API also exposing per-image selective download.
- Current Open-i reuse policy text.

Treat §3–§5 below as **expected** until you confirm them against https://openi.nlm.nih.gov/faq.

## 3. Expected data structure (to verify on first inspection)

### 3.1 Report XML (one file per report, named e.g. `1.xml`, `2.xml`, …)
Expected elements:

| Expected element | Purpose |
|---|---|
| `/eCitation/uId` | report id |
| `/eCitation/MedlineCitation/...` | citation metadata |
| `/eCitation/MeSH/major` and `.../automatic` | manual + auto-generated anatomy/finding tags |
| `/eCitation/parentImage/@id` | PNG basename(s) linked to this report (0..N per report) |
| `/eCitation/.../AbstractText[@Label='COMPARISON']` | prior-study comparison |
| `/eCitation/.../AbstractText[@Label='INDICATION']` | clinical indication |
| `/eCitation/.../AbstractText[@Label='FINDINGS']` | free-text findings |
| `/eCitation/.../AbstractText[@Label='IMPRESSION']` | free-text impression (the summary) |

### 3.2 Image files (inside `NLMCXR_png.tgz`)
Expected: PNG files named `CXR<uid>_IM-<series>-<n>.png`, where `<uid>` matches `parentImage/@id` in the XML. **Verify filename convention on first extraction.**

### 3.3 Image ↔ report linkage
A report XML lists its images via `parentImage` elements. The reverse (image → report) is reconstructed by indexing all XMLs once.

## 4. Reuse / redistribution / attribution (UNVERIFIED — confirm against the FAQ)

The commonly-cited position (NOT re-verified from the live page in this phase):

- The IU chest X-ray collection is from the Indiana Network for Patient Care and was de-identified for public release.
- Open-i content follows the originating source's terms; the IU chest X-ray collection has been published by NLM for research/educational use.
- No standardized CC license tag is attached to the collection in the materials usually seen.
- Attribution: cite Demner-Fushman et al., *"Preparing a collection of radiology examinations for distribution and retrieval"*, JAMIA 2016, and the Open-i service.

**Rules VisionGuard applies regardless** of what the FAQ says:

- **Do not redistribute** the raw images or reports. Keep them local; `.gitignore` already blocks `datasets/**`, `artifacts/**`, and image/binary types.
- **Do not re-identify** or attempt to cross-link to any outside identity.
- **Do not post** images, report text, or MeSH tag combinations that could narrow identification in public issues/PRs/screenshots.
- Treat the collection as medical research data: a research-prototype disclaimer stays on the Streamlit UI (already in `frontend/app.py`).

If the live FAQ states anything stricter than the above, the live FAQ wins. Before any download, read https://openi.nlm.nih.gov/faq, and if it disagrees with §4, update this doc.

## 5. Smallest feasible acquisition — staged

### Stage A — reports only (approve first)
- `NLMCXR_reports.tgz` — **archive size UNVERIFIED here** (HTTP `HEAD` would confirm; not performed in this phase, since the task said no network transfer). Based on widely-cited characteristics (few-thousand short XML files), expected a few–tens of MB.
- Rationale: text-only sample is sufficient to validate XML parsing, linkage structure, section extraction, and MeSH-tag coverage. **No images needed** to prove the pipeline wiring and the Phase-3 claim-extraction baseline.

### Stage B — a small image slice (approve only after Stage A)
Needed because `NLMCXR_png.tgz` is a **monolithic tarball** on this URL — **selective download of individual images from this archive URL is NOT supported**. However:

- **Option B1 (preferred)**: use the Open-i search API (`https://openi.nlm.nih.gov/api/search?...`) to fetch individual image URLs for a small, explicit list of report ids identified in Stage A, then download those PNGs one by one. **UNVERIFIED here** that this API is currently exposed as documented; confirm in Stage A by hitting the search endpoint for one query and inspecting the JSON before fetching any image.
- **Option B2 (fallback)**: download the full `NLMCXR_png.tgz` once, extract **only** the PNGs for the sample case list, delete the tarball. Peak disk use ≈ archive size + extracted sample. Archive size **UNVERIFIED here**.

Stage B stays unapproved until (a) Stage A has mapped reports to parentImage ids, and (b) you pick B1 vs. B2.

### What we are explicitly NOT doing
- Not downloading `NLMCXR_png.tgz` or `NLMCXR_reports.tgz` in this phase.
- Not installing any ML package (`torch`, etc.) in this phase.
- Not implementing agents or training.
- Not mirroring either archive to Kaggle, GitHub, or any public bucket.

## 6. Architecture update — 2D images instead of 3D volumes

LangGraph pipeline is **unchanged in topology**:

1. Supervisor / Orchestrator
2. Input Validation
3. Atomic Claim Extraction
4. Visual Evidence
5. Claim Verification
6. Critic
7. Audit Report
8. Human-review escalation

What changes module-by-module:

| Module | 3D CT (deprecated) | 2D X-ray (now) |
|---|---|---|
| Loader | NIfTI `.nii.gz` + anatomy mask tarball | PNG per image + XML per report |
| Linkage | `Volumename` + `Anatomy` (implicit) | `parentImage/@id` in report XML (explicit) |
| Visual Evidence agent | 3D mask ROI → feature extractor | 2D image region (initially whole image; later: region proposals, Grad-CAM, or CheXpert-style keyword alignment) |
| Weak labels | 197 anatomy categories | MeSH `major` + `automatic` tags |
| Report sections | one `Sentence` row per anatomy | named `AbstractText[@Label=...]` sections per report |

Directories unchanged. The `datasets/` package now hosts IU X-Ray loaders instead of RadGenome ones; old files are kept, headered as deprecated.

## 7. Four-tier claim provenance (adopt now, enforce in later phases)

A single, project-wide separation of what text is what. **Never conflate these in code, logs, or UI**:

| Tier | What it is | Example source | Used for |
|---|---|---|---|
| T0 — **Original report text** | Human-authored sections from the IU XML (`FINDINGS`, `IMPRESSION`, …) | NLM archive | Baseline reference; textual-entailment target |
| T1 — **Generated claims** | Atomic claims extracted from a report under audit (either a model-generated report or a T0 report, depending on experiment) | Claim-extraction agent output | Pipeline input to verification |
| T2 — **Weak labels** | Signals present without human claim-level adjudication: MeSH tags, keyword matches, image-level classifier outputs | NLM MeSH tags; later CheXpert-labeler outputs | Noisy training / confidence prior; **not** ground truth |
| T3 — **Validated claim-level ground truth** | Per-claim `supported / contradicted / indeterminate` labels adjudicated by a human on this project | To be produced on a small evaluation set in a later phase | Evaluation only |

Rule: a VisionGuard metric that cites T2 as truth must say "weak-label estimate" in both code and UI. T3 is the only thing allowed to be called "ground truth".

## 8. Deprecation of RadGenome assets

Keep for history; redirect active work. Specifically:

- `scripts/hf_auth_check.py`, `scripts/radgenome_peek.py`, `scripts/build_manifest.py` — add a `DEPRECATED:` header comment pointing here.
- `datasets/radgenome_schema.py`, `datasets/manifest.py` — same header; keep tests passing.
- `docs/dataset_evidence_audit.md`, `docs/sample_download_plan.md` — add a top-of-file banner noting superseded by this document; keep as-is otherwise.
- Phase-1 README roadmap row for datasets is updated to point at IU X-Ray.
- No files are deleted.

## 9. Smallest-data acquisition plan (final, waiting for approval)

**Stage A (reports only)** — approval requested:
- `GET https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz` once, extract into `datasets/iu_xray/reports/`, confirm count + XML structure.
- Pull a listing of all `parentImage/@id` values, group by report, sample 5 report ids for Stage B.
- Produce a small manifest analogous to Phase 2C: counts, per-section non-empty rates, MeSH-tag frequency. No report text in the manifest.

**Stage B (small image slice)** — hold:
- Decide B1 (per-image via Open-i API, needs API probe first) vs. B2 (monolithic PNG tarball, extract 10 PNGs, delete archive).
- Preferred: B1 if the API probe succeeds.

**Then stop and report.** No agents, no training, no additional packages.

## 10. Remaining uncertainties and risks

| # | Item | Impact | Mitigation |
|---|---|---|---|
| 1 | Open-i FAQ did not render for WebFetch; §3/§4 are expected, not verified | May disagree with live policy | Confirm against https://openi.nlm.nih.gov/faq before Stage A |
| 2 | Report tarball size unverified (no HTTP HEAD performed) | Budget surprise | HEAD the URL as the first operation of Stage A; abort if > 500 MB (far above any public estimate) |
| 3 | Image tarball size unverified | Decides B1 vs. B2 | Same HEAD check before Stage B |
| 4 | Per-image Open-i API endpoint not probed | B1 may need a workaround | Probe once in Stage A with a single search query, inspect JSON |
| 5 | De-identification already applied upstream; no PHI expected but not re-verified | PHI leak risk if assumption is wrong | On first XML read, scan for obvious PHI patterns (names, dates matching DoB formats) before storing anything |
| 6 | XML tag conventions may differ across the released version | Loader breakage | Validate the first XML's full tag tree before writing the loader; let the real schema drive the parser |
