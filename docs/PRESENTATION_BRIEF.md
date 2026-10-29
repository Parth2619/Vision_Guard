# VisionGuard Medical — Presentation Brief

**For teammates building the slide deck.** Everything your presentation needs is here, grouped by what naturally becomes one or two slides. Headings → slide titles. Bullets → slide bullets. Longer "Say this" blocks → speaker notes.

---

## 1. One-line pitch (title slide)

> **VisionGuard Medical** — an agentic AI pipeline that audits AI-generated chest X-ray reports for hallucinated claims.

Subtitle idea: *"Catching what the VLM made up."*

---

## 2. The problem (1 slide)

- Vision-Language Models (VLMs) — ChatGPT with vision, LLaVA, Qwen-VL, Gemini — can read a chest X-ray and generate a free-text radiology report.
- These reports often contain **hallucinated claims**: statements the image doesn't actually support.
- Example: a VLM says *"large mass in the right lower lobe"* when there is no mass. A doctor reading only the report would be misled.
- No system today independently checks each sentence of an AI-generated medical report against the pixels.

**Say this:** Hallucination is the #1 reason medical AI can't be trusted yet. VisionGuard addresses it by auditing per-claim rather than per-report.

---

## 3. Our solution in one slide

- Takes an image + a VLM-generated report.
- Breaks the report into individual claims.
- Checks each claim against **independent visual evidence** from pretrained vision models.
- Classifies each claim as **Supported / Contradicted / Indeterminate**.
- Escalates uncertain or high-risk claims for human review.
- Produces a structured audit report with evidence trail and confidence scores.

**Not clinically validated. Research prototype.** Say this out loud; it protects you from scope questions.

---

## 4. Architecture (2 slides)

### Slide A — the 8-stage pipeline (reference diagram)

Our architecture matches the **Human-in-the-Loop Claim-to-Evidence Auditing Framework for Medical VLMs** reference:

| # | Stage | What it does |
|---|---|---|
| 1 | **Input** | Receive chest X-ray + VLM-generated report. |
| 2 | **Claim Extraction** | Parse report into typed atomic claims (Object / Attribute / Count / Spatial). |
| 3 | **Visual Evidence** | Run pretrained vision models on the image. |
| 4 | **Claim Verification** | Decide supported/contradicted/indeterminate per claim. |
| 5 | **Fusion + Calibration** | Trained MLP (128 → 64 → 3) emits calibrated confidence. |
| 6 | **Risk-Aware Gate** | Escalate high-risk or low-confidence claims. |
| 7 | **Human Review** | Expert confirms, rejects, or flags insufficient. |
| 8 | **Audit Report** | Per-claim decision + evidence + confidence + overall status. |

### Slide B — implementation flow

```
   Image + Report
         ↓
   LangGraph StateGraph (7 agents)
         ↓
   validate → extract → evidence → verify → calibrate → critic → human_review
                                        │
                       (if critic blocks: bounded recheck once)
         ↓
   AuditReport JSON (claims · verdicts · evidence · trace · overall status)
```

**Say this:** Every stage is a real LangGraph node with shared state, conditional edges, and bounded retry.

---

## 5. Technology stack (1 slide)

| Layer | Tool |
|---|---|
| Agent orchestration | **LangGraph** (StateGraph, nodes, conditional edges) |
| Backend | **FastAPI** (serves `/audit`, `/review`, static frontend) |
| Frontend | Vanilla HTML/CSS/JS — reading-room console design |
| Schemas | **Pydantic v2** (Claim, Evidence, ClaimVerdict, AuditReport) |
| Chest X-ray classifier | **torchxrayvision DenseNet-121** (18 pathologies, pretrained) |
| Object detector | **Grounding-DINO tiny** via Hugging Face (zero-shot, pretrained) |
| VLM under audit | **Qwen2-VL-2B-Instruct** (pretrained, ran once on Kaggle T4) |
| Calibration MLP | **PyTorch** — our own trained network |
| Persistence | **SQLite** (expert-review decisions) |
| Testing | **pytest** — 92 tests, all offline, all passing |
| Language | **Python 3.12** on Windows |
| Dataset | **IU Chest X-Ray / NLM Open-i** via Kaggle mirror `raddar/chest-xrays-indiana-university` (3,851 reports, 7,466 image refs) |

---

## 6. What we built vs. what we reused (1 slide)

| | Trained by us? | Source |
|---|---|---|
| torchxrayvision classifier | ❌ pretrained | NIH + CheXpert + MIMIC-CXR + PadChest + Open-i (~400k images) |
| Grounding-DINO detector | ❌ pretrained | IDEA-Research |
| Qwen2-VL-2B VLM | ❌ pretrained | Alibaba Qwen |
| **Calibration MLP (128→64→3)** | ✅ **trained from scratch** | **by us, on 6,514 real IU report claims** |

**Say this:** The three big vision/language models are off-the-shelf pretrained models — using them as-is is the right choice. The one network we trained ourselves is the small fusion head that combines signals and produces calibrated confidence.

---

## 7. The agentic layer (1 slide)

Seven specialized agents orchestrated by LangGraph:

| Agent | Responsibility |
|---|---|
| `validate` | Checks input, routes to extraction or straight to human review. |
| `extract` | Parses the report into typed atomic claims. |
| `evidence` | Calls classifier + detector, assembles independent evidence per claim. |
| `verify` | Applies conservative rules: missing evidence → indeterminate, never contradicted. |
| `calibrate` | Runs the trained MLP, emits softmax probabilities + risk level. |
| `critic` | Reviews verdicts; if a verdict lacks independent evidence it blocks for recheck. |
| `human_review` | Risk-aware gate: escalates indeterminate or high-risk/low-confidence claims. |

**Agentic properties:** shared mutable state, conditional edges, bounded retry (`MAX_RECHECKS = 1`), execution trace visible in the UI, explicit human-in-the-loop handoff.

**Say this:** The architecture is agentic. The agents are deliberately deterministic (not LLM-driven) for reproducibility and offline operation; any agent can be swapped for an LLM-backed one via the existing protocols without changing the graph.

---

## 8. End-to-end request lifecycle (1 slide — walk through it)

User uploads `1000_IM-0003-1001.dcm.png`, picks VLM mode, clicks Run.

1. **FastAPI** receives multipart; validates image magic bytes + size.
2. **VLM step** — SHA-256 of image bytes → lookup in `vlm_generated_reports.json` → Qwen2-VL's cached report *"FINDINGS: There is a large mass in the right lower lobe…"*.
3. **LangGraph** invoked. Each node runs in order:
   - `validate` → ok.
   - `extract` → 1 claim: `finding="mass", polarity=positive, type=object`.
   - `evidence` → classifier returns 18 probabilities; detector returns bounding boxes; evidence list assembled.
   - `verify` → classifier confidence 0.18 for "Mass" → contradicts the positive claim → **CONTRADICTED**.
   - `calibrate` → MLP produces `{supported:0.05, contradicted:0.90, indeterminate:0.05}`.
   - `critic` → no blockers.
   - `human_review` → risk level `high` + calibrated confidence 0.90 → not escalated.
4. **AuditReport JSON** returned → frontend renders X-ray with red bounding box, verdict card with `contradicted · conf 0.90 · risk high`, agent trace strip, overall status "contradicted".

**Say this:** The VLM claimed a mass that isn't visibly there. Our pipeline caught it. That's the whole value prop in one example.

---

## 9. The frontend (1 slide)

- Replaced Streamlit with a static web app served by FastAPI.
- Reading-room console aesthetic: deep carbon background, hairline dividers, Inter Tight typography, status LEDs instead of pills.
- Three columns: inputs (left), radiograph viewer with bounding-box overlays (center, hero), per-claim verdicts (right).
- Empty state: pulsing ECG trace + instrument metadata.
- Expert Review tab for human adjudication; posts to SQLite via `/review`.
- About tab explains what the system is and what it isn't.
- Fully keyboard-accessible, respects `prefers-reduced-motion`.

---

## 10. Measured results (1 slide — most important)

**42 real IU Chest X-Ray claim/image pairs hand-labeled by the project author.**

| Metric | Value |
|---|---:|
| Accuracy (agreement with annotator) | **78.6 %** (33 / 42) |
| F1 — supported | 0.87 |
| F1 — indeterminate | 0.31 |
| F1 — contradicted | undefined (0 cases in queue) |

**Confusion matrix** (rows = annotator, cols = pipeline):

| | supported | contradicted | indeterminate |
|---|---:|---:|---:|
| supported | 31 | 0 | 0 |
| contradicted | 0 | 0 | 0 |
| indeterminate | 9 | 0 | 2 |

**Failure mode we're honest about:** when the annotator could not confidently decide ("indeterminate"), the pipeline over-claimed "supported" 9 of 11 times. The system has a measurable bias toward confidence on ambiguous images.

**Say this:** 78.6 % is the headline. The failure-mode slide tells the honest story: the system still over-claims on uncertain cases, which is the first thing to tighten in follow-up work.

---

## 11. What makes the project credible (1 slide)

- **92 automated tests**, all offline, all passing.
- **Three real pretrained vision models**, not toy wrappers.
- **One MLP trained from scratch on 6,514 real medical report claims.**
- **Real IU Chest X-Ray data** downloaded via Kaggle (official license, non-redistribution respected locally).
- **Real VLM output** generated on Kaggle T4 — Qwen2-VL-2B produced the reports we audit.
- **Measured results on hand-labeled real data**, not synthetic benchmarks.
- **Honest documentation**: `docs/PROJECT_REPORT.md` lists every limitation.

---

## 12. Honest limitations (1 slide — don't hide these)

1. Not clinically validated. Research prototype only.
2. Rule-based claim extractor (not an LLM) — fine for the 14-finding chest X-ray vocab, insufficient for general-domain audit.
3. No CLIP/SigLIP similarity verification.
4. Count and Spatial claim types are extracted but share the Object verifier (no dedicated verifier).
5. Calibration MLP trained on weak labels (MeSH tags), not on hand-adjudicated per-claim truth.
6. Hand labels came from a non-expert annotator (project author), not a radiologist.
7. 42 cases is small — 95 % CI roughly ±13 percentage points.
8. 0 contradicted cases in the eval queue → the pipeline's ability to catch deliberately false claims was not measured.
9. VLM output is **precomputed** on 48 IU X-rays; arbitrary uploads fall back to a stub.

**Say this:** Each of these has a planned fix in future work.

---

## 13. Future work (1 slide)

- **Spatial verifier** with anatomy segmentation (MedSAM / TotalSegmentator). Catches left/right confusion — a real hallucination class.
- **BiomedCLIP** for attribute verification (size, severity, laterality).
- **Count verifier** reusing existing detector output.
- **LLM-based claim parser** for general-domain inputs.
- **Radiologist-labeled held-out eval set** for real calibration.
- **Live VLM backend** (needs GPU) replacing the precomputed lookup.
- **Expected Calibration Error (ECE)** metric once more hand labels exist.

---

## 14. Repository layout (1 slide)

```
agents/            # reserved; agents live inline as LangGraph nodes
api/
  main.py          # FastAPI: /audit, /review, serves frontend
  review.py        # SQLite expert-review persistence
baselines/
  text_only.py     # non-agentic baseline (all indeterminate)
datasets/
  iu_xray.py       # XML parser for IU reports
  manifest.py      # (legacy) RadGenome aggregator — deprecated
frontend/web/      # vanilla HTML + CSS + JS, served by FastAPI
graph/
  build.py         # LangGraph StateGraph + all node definitions
medical/
  claims.py        # Pydantic schemas (Claim, Evidence, Verdict, AuditReport)
  extract.py       # rule-based typed claim extractor
  evidence.py      # provider functions (report-text, synthetic, image, detector)
  verify.py        # verification + critic rules
  risk.py          # finding → risk level map
  evaluation.py    # metrics harness
models/
  xray.py          # torchxrayvision wrapper
  detector.py      # Grounding-DINO wrapper
  vlm.py           # VLM backend protocol + stub + precomputed
  calibration.py   # trained MLP + feature extraction
  calibration_iu.pt# weights trained on 6,514 IU claims
scripts/           # one-shot utilities (training, download, eval)
tests/             # 92 pytest files
notebooks/
  vlm_generate_reports.ipynb   # Kaggle T4 notebook for Qwen2-VL
docs/              # PROJECT_REPORT.md, this brief, dataset docs
```

---

## 15. Demo flow (what to actually click during the presentation)

**Pre-demo setup**
```powershell
cd 'D:\STUDY\DL Lab\Project\VisionGuard-Medical'
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000`. Pre-upload one X-ray so weights are warm.

**During the demo (10 minutes)**

1. **Audit tab — Normal mode (3 min)**
   - Paste a short clean report ("No acute abnormality.")
   - Upload a known-clean X-ray.
   - Run audit → shows "Clean" overall, SUPPORTED verdict, no boxes.
   - *"The pipeline confirms a true report on a normal X-ray."*

2. **Audit tab — VLM mode (4 min)**
   - Switch to VLM mode.
   - Upload `datasets/iu_xray/images/1000_IM-0003-1001.dcm.png` (one of the 48 precomputed).
   - Run audit → shows Qwen2-VL's report claiming a large mass.
   - Point at the CONTRADICTED verdict and the red bounding box.
   - *"The VLM hallucinated. VisionGuard caught it."*
   - Expand a verdict card → show the evidence trail with calibrated confidence.
   - Point at the Agent trace strip at the bottom → *"Every stage of the LangGraph ran."*

3. **Review tab (1 min)**
   - Show flagged claims from the last audit.
   - Click Confirm/Reject on one → "Decisions persist to local SQLite."

4. **About tab (1 min)**
   - Read the Pipeline, Models, and Measured Result panels.
   - Close with "78.6 % agreement with the author on 42 hand-labeled cases."

5. **Numbers slide** (if time) — the measured-result table from §10 above.

---

## 16. Q&A cheat sheet (speaker notes)

| Question | Answer |
|---|---|
| Did you train a model from scratch? | Yes — the calibration MLP (128 → 64 → 3), on 6,514 real IU report claims. The vision models (classifier, detector, VLM) are pretrained — using them as-is is the correct engineering choice; retraining them would need orders of magnitude more compute. |
| Is this clinically usable? | No. Research prototype. Not validated. Repeated in docs and schema. |
| What's the dataset? | IU Chest X-Ray / NLM Open-i, 3,851 reports and 7,466 image references, downloaded via Kaggle. We used 50 PNGs locally and the full reports CSV. |
| Why no LLM for claim extraction? | Chest X-ray vocab is small (14 findings). Deterministic rules are faster, reproducible, offline, and achieve comparable extraction. The slot for an LLM backend exists in code. |
| Why precomputed VLM? | Live VLM inference needs a GPU. We generated 48 reports on a Kaggle T4 and the pipeline looks them up by image hash. Any system with GPU can run the VLM live via the same backend protocol. |
| Is this really agentic? | Yes. LangGraph StateGraph, 7 specialized agents, shared state, conditional edges, bounded retry, human-in-the-loop. Agents are deterministic by choice (not an LLM-reasoning cascade). |
| What's the accuracy? | 78.6 % agreement with the hand-labeling annotator on 42 claim/image pairs. F1 for Supported 0.87. The system over-claims on ambiguous cases — a documented failure mode. |
| What about ECE / Cohen's κ? | Code hooks exist; meaningful numbers require a larger hand-labeled set. Future work. |
| Can your pipeline catch a real hallucination? | Demonstrated qualitatively in the VLM demo (contradicts a VLM-claimed mass). Not yet measured systematically because the eval queue happened to have zero contradicted cases. |
| What's the biggest limitation? | Weak-label training of the calibration MLP + non-expert hand labels. Fixing both needs radiologist annotation time. |

---

## 17. Credits (last slide)

- **torchxrayvision** — chest X-ray classifier pretrained on NIH + CheXpert + MIMIC-CXR + PadChest + Open-i.
- **Grounding-DINO** — IDEA-Research zero-shot detector.
- **Qwen2-VL-2B-Instruct** — Alibaba vision-language model.
- **LangGraph** — agent orchestration framework.
- **FastAPI + Pydantic + PyTorch + scikit-learn**.
- **IU Chest X-Ray** — Demner-Fushman D. et al., *"Preparing a collection of radiology examinations for distribution and retrieval"*, JAMIA 2016. Accessed via Kaggle mirror `raddar/chest-xrays-indiana-university`.
- **Kaggle** — free T4 GPU for Qwen2-VL inference.

---

## Suggested slide order (for your teammates)

1. Title + one-line pitch
2. The problem (hallucination)
3. Our solution (one paragraph)
4. Architecture diagram (reference 8 stages)
5. Implementation flow (LangGraph nodes)
6. Tech stack
7. Built vs. reused
8. The agentic layer (7 agents)
9. End-to-end example (walk through one audit)
10. Frontend screenshot
11. Measured results (headline number + failure mode)
12. Honest limitations
13. Future work
14. Demo
15. Q&A / credits

Each slide = ~1 of the sections above. Teachers reward structure.

---

## File metadata

- Project: VisionGuard Medical
- Phase: 10 (all planned phases complete)
- Tests: 92 passing, 0 failing
- Last measured eval: 2026-10-24
- Primary developer: project author
- Not clinically validated. Research prototype only.
