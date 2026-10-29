# VisionGuard Medical — Context brief for Claude chat

**How to use:** On claude.ai, start a new chat and paste this whole document as your first message. Then ask any question about the project. Claude will have full context.

---

## Prompt to Claude (the chat you start)

> You are helping me prepare for and defend a university project called **VisionGuard Medical**. I will ask you questions about it on-the-go from my phone. Everything you need to know is in the context below. If a question goes beyond the context, say so honestly — never invent numbers, file paths, or claims the context doesn't support. Keep answers short, direct, and defensible on stage. If I ask "what should I say" or "give me a one-liner", produce a short confident sentence I can speak out loud.

---

## 1. One-sentence pitch

VisionGuard Medical is an agentic AI pipeline that audits AI-generated chest X-ray reports — it breaks each report into individual claims and checks every claim against the actual image using pretrained vision models, flagging anything the pixels don't support as a hallucination.

**Not clinically validated. Research prototype only.**

---

## 2. Why this project exists

Vision-Language Models (ChatGPT-vision, Qwen-VL, LLaVA, Gemini) can read a chest X-ray and write a radiology report. They often hallucinate — write confident sentences about findings that aren't in the image. VisionGuard audits each sentence of the AI-generated report against the pixels.

---

## 3. The architecture — 8 stages (matches the reference diagram)

| # | Stage | Our implementation |
|---|---|---|
| 1 | Input (image + VLM report) | FastAPI `/audit` multipart endpoint |
| 2 | Claim extraction | Rule-based keyword parser → typed claims (Object / Attribute / Count / Spatial) |
| 3 | Visual evidence | torchxrayvision classifier (18 chest pathologies) + Grounding-DINO zero-shot detector with bounding boxes |
| 4 | Claim verification | Conservative rules on provenance; missing evidence → indeterminate never contradicted |
| 5 | Fusion + calibration MLP (128→64→3) | **Trained from scratch by us on 6,514 real IU report claims** |
| 6 | Risk-aware decision gate | Risk table (pneumothorax/mass/pneumonia/fracture = high); escalates uncertain or high-risk/low-confidence |
| 7 | Human-in-the-loop review | Web UI with Confirm / Reject / Insufficient buttons; SQLite persistence |
| 8 | Final audit report | JSON: per-claim decision, bounding boxes, confidence, agent trace, overall status |

Orchestration: **LangGraph StateGraph** with 7 specialised agent nodes (validate → extract → evidence → verify → calibrate → critic → human_review), shared state, conditional edges, bounded retry (`MAX_RECHECKS = 1`), execution trace visible to the user.

---

## 4. Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 on Windows |
| Agent orchestration | LangGraph |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS (replaced Streamlit) |
| Schemas | Pydantic v2 |
| Chest X-ray classifier | torchxrayvision DenseNet-121 (pretrained) |
| Object detector | Grounding-DINO tiny via HF transformers (pretrained) |
| VLM | Qwen2-VL-2B-Instruct (pretrained, ran once on Kaggle T4) |
| Our trained model | CalibrationMLP in PyTorch |
| Review storage | SQLite (stdlib) |
| Testing | pytest — 92 tests, all offline, passing |
| Dataset | IU Chest X-Ray / NLM Open-i via Kaggle mirror `raddar/chest-xrays-indiana-university` |

---

## 5. Models — three pretrained, one trained from scratch

### Three used as-is (no training, no fine-tuning)

- **torchxrayvision DenseNet-121** — ~7M params. Pretrained on ~400,000 chest X-rays (NIH + CheXpert + MIMIC-CXR + PadChest + Open-i). Outputs 18 pathology probabilities. Role: classify *what* is in the image.
- **Grounding-DINO tiny** — ~170M params. Pretrained general-domain by IDEA-Research. Zero-shot detector with text prompts. Outputs bounding boxes + confidence scores. Role: locate *where* a finding is.
- **Qwen2-VL-2B-Instruct** — 2B params. Pretrained by Alibaba. Ran once on a free Kaggle T4 to generate reports for 48 IU X-rays; outputs saved to JSON keyed by image SHA-256. Role: generate the report under audit.

### One trained from scratch — **CalibrationMLP**

- Architecture: `Linear(8→128) → ReLU → Linear(128→64) → ReLU → Linear(64→3)`.
- Total parameters: ~9,603.
- Takes 8 features (classifier prob, classifier confidence margin, detector score, detector-present flag, polarity-is-negative, polarity-is-uncertain, has-independent-support, has-independent-conflict).
- Outputs 3 probabilities via softmax: `{supported, contradicted, indeterminate}`.
- Training data: 6,514 claims extracted from 3,851 real IU chest X-ray reports.
- Training labels: weak — derived automatically by matching each claim's finding against the report's MeSH + Problems columns.
- Optimizer: Adam, lr = 1e-2.
- Loss: cross-entropy.
- Epochs: 100.
- Hardware: CPU.
- Training time: ~2 seconds.
- Final train accuracy: 100 % (deterministic from weak labels; expected).
- Weights saved to `models/calibration_iu.pt` (~40 KB).

### Activation functions used in the MLP

- **ReLU** between hidden layers → introduces non-linearity so interactions between features can be learned.
- **Softmax** on the output → turns three raw scores into a probability distribution summing to 1.

### Pooling

No pooling added by us. Pooling happens **inside the pretrained models** (DenseNet uses global average pooling; Swin backbone in Grounding-DINO uses patch merging). Our MLP operates on 8 scalar features — nothing spatial to pool.

### Transfer learning

Technically we don't *fine-tune* any pretrained model. But our setup is a form of transfer learning in the **feature-extraction** sense: pretrained backbones produce signals that feed our trained MLP head. The heavy visual understanding is reused; our head only learns how to combine the signals.

---

## 6. Dataset — IU Chest X-Ray

- Source: Indiana University Chest X-Ray via Kaggle mirror `raddar/chest-xrays-indiana-university`.
- Scale: 3,851 reports + 7,466 image references.
- Downloaded via Kaggle API: `indiana_reports.csv`, `indiana_projections.csv`, and 50 PNG images locally.
- Qwen2-VL ran on 50 of those images on Kaggle; 48 reports produced (2 failed silently).
- 42 claim/image pairs hand-labelled by the project author via the Label Samples tab (Supported / Contradicted / Indeterminate / Skip buttons).

Note: the IU dataset labels are **report-level** (MeSH tags per case). VisionGuard works **per-claim**. The hand-labelling filled that gap: nobody has ever labelled IU at the sentence level, so we labelled 42 ourselves for the eval.

---

## 7. Measured results (hand-label eval, 42 cases)

| Metric | Value |
|---|---:|
| Accuracy (pipeline agrees with annotator) | **78.6 %** (33 / 42) |
| F1 — supported | **0.87** |
| F1 — indeterminate | 0.31 |
| F1 — contradicted | undefined (zero contradicted cases in queue) |
| Precision — supported | 0.775 |
| Recall — supported | 1.00 |
| Precision — indeterminate | 1.00 |
| Recall — indeterminate | 0.18 |

Confusion matrix (rows = annotator, cols = pipeline):

| | supported | contradicted | indeterminate |
|---|---:|---:|---:|
| supported | 31 | 0 | 0 |
| contradicted | 0 | 0 | 0 |
| indeterminate | 9 | 0 | 2 |

**Documented failure mode:** on 11 cases the annotator judged "indeterminate", the pipeline over-claimed "supported" **9 out of 11 times**. The system is over-confident on ambiguous images — this is the primary target for future work.

---

## 8. Honest limitations (don't hide these)

1. Not clinically validated. Research prototype only.
2. Rule-based keyword extractor (not an LLM) — fine for 14-finding chest-X-ray vocab, not for general domain.
3. No CLIP/SigLIP attribute verification.
4. Count and Spatial claims are extracted and typed but share the Object verifier (no dedicated verifier).
5. Calibration MLP trained on weak labels (MeSH matching), not on hand-adjudicated per-claim truth.
6. Hand labels came from a non-expert annotator (project author), not a radiologist.
7. 42 cases is small — accuracy carries ~±13 percentage-point confidence interval.
8. Zero "contradicted" cases in the eval queue — we don't yet have hard evidence the system catches deliberate hallucinations systematically.
9. VLM output is precomputed on 48 IU X-rays; arbitrary uploads fall back to a stub.

---

## 9. Common questions — ready answers

**Did you train anything?**
Yes. One network from scratch: the CalibrationMLP (128→64→3), 9,603 parameters, trained for 100 epochs with Adam + cross-entropy on 6,514 real IU chest X-ray report claims. Training took about 2 seconds on CPU. The three large models — classifier, detector, VLM — are pretrained; using them as-is is the correct engineering choice.

**Is this clinically usable?**
No. Research prototype. Not validated.

**Why no LLM for claim extraction?**
Chest X-ray vocabulary is small (14 findings). Deterministic rules are faster, reproducible, offline, and perform comparably. The ClaimExtractor protocol lets us swap in an LLM backend later without changing the graph.

**Why precomputed VLM?**
Live VLM inference needs a GPU with ≥6 GB VRAM. We generated 48 reports once on a free Kaggle T4. The pipeline looks them up by image SHA-256. The VLMBackend protocol lets any GPU-equipped host run the VLM live.

**Is it really agentic?**
Yes. LangGraph StateGraph, 7 specialised agents, shared state, conditional edges, bounded retry, execution trace, human-in-the-loop handoff. Individual agents are deterministic by design — for reproducibility, offline operation, and cost. Any agent can be swapped for an LLM-backed one via the existing protocols.

**What's the accuracy?**
78.6 % agreement with the hand-labelling annotator on 42 real cases. F1 supported 0.87. Failure mode: over-claims on ambiguous images (9 of 11).

**Why 100 epochs?**
The dataset is 6,514 rows and the model is tiny — 100 epochs with Adam at lr 0.01 converges to zero loss in ~2 seconds. More would just overfit the weak labels harder.

**Why ReLU + Softmax?**
ReLU between hidden layers introduces non-linearity (without it, three linear layers collapse to one). Softmax on the output converts raw scores into probabilities summing to 1 — the risk gate and UI both need calibrated probabilities, not raw logits.

**Did you do pooling?**
No pooling added by us. The pretrained models pool internally (DenseNet uses global average pooling, Swin uses patch merging). Our MLP input is 8 scalar features — nothing spatial to pool over.

**Why Grounding-DINO AND DenseNet?**
DenseNet answers *what* (probability a finding is present). Grounding-DINO answers *where* (a bounding box). Two different models with different failure modes provide independent cross-checks and give the UI the visual proof it needs to draw boxes.

**Why F1 instead of just accuracy?**
Accuracy can be misleading when classes are imbalanced. F1 (harmonic mean of precision and recall) only climbs when both are high; it exposes models that pretend to work by always predicting the majority class.

**What is F1 indeterminate = 0.31?**
On the "indeterminate" class specifically, the pipeline's F1 is only 0.31 — precision 1.00 (every "indeterminate" it emits is correct) but recall 0.18 (it only catches 2 of 11 cases a human would mark ambiguous). The system is miserly with indeterminate; it should abstain more often.

**Why did you have to hand-label 42 if the dataset is labelled?**
IU gives you report-level labels (MeSH tags per case). VisionGuard works per-claim — on individual sentences in the report. Nobody has ever labelled IU at that granularity. We hand-labelled 42 so we could measure whether the pipeline agrees with human judgment on the same per-claim question the pipeline is trying to answer.

**What's the biggest limitation?**
The calibration MLP was trained on weak labels (MeSH tag matching), and hand labels came from a non-expert annotator. Fixing both requires radiologist time, which is scheduled as future work.

**What is "transfer learning" in this project?**
Feature extraction form: pretrained vision models are frozen backbones that produce signals; our trained MLP sits on top as a lightweight fusion head. We don't fine-tune the big models.

**Why no fine-tuning?**
torchxrayvision was trained on ~400k X-rays — fine-tuning on IU's 7k would make it worse. Grounding-DINO fine-tuning needs bounding-box annotations we don't have. Qwen2-VL fine-tuning needs multiple GPUs for days.

**What's "overall hallucination status"?**
A top-line label on each audit: `clean` (no hallucinations detected), `contradicted` (at least one contradicted claim found), `needs_review` (indeterminate claims present), or `insufficient_evidence` (nothing conclusive).

**How does the risk gate work?**
A claim is escalated to human review if: its status is INDETERMINATE, or the finding is in the high-risk list (pneumothorax, mass, pneumonia, fracture) and calibrated confidence < 0.75, or the critic flagged a block-severity issue.

---

## 10. Demo flow (what will happen on stage)

1. **Normal mode**: paste a clean report + upload any IU X-ray → pipeline confirms it → overall status "clean".
2. **VLM mode**: upload `1000_IM-0003-1001.dcm.png` → Qwen2-VL's precomputed report claims a mass → pipeline marks it CONTRADICTED with ~0.9 confidence → red box drawn on X-ray.
3. **Review tab**: shows flagged claims; click Confirm/Reject → persisted to SQLite.
4. **About tab**: six panels explaining pipeline, models, measured result.

---

## 11. Project structure (file names you might reference)

- `api/main.py` — FastAPI backend, `/audit` endpoint
- `graph/build.py` — LangGraph StateGraph + all 7 agent nodes
- `medical/claims.py` — Pydantic schemas (Claim, Evidence, Verdict, AuditReport)
- `medical/extract.py` — rule-based claim extractor
- `medical/verify.py` — verification + critic rules
- `medical/evaluation.py` — metrics harness
- `models/calibration.py` — the trained MLP + feature extraction
- `models/calibration_iu.pt` — trained weights
- `models/xray.py`, `models/detector.py`, `models/vlm.py` — pretrained model wrappers
- `frontend/web/` — the HTML/CSS/JS UI
- `tests/` — 92 passing pytest files
- `notebooks/metrics_overview.ipynb` — all metrics + charts (open to show teacher)
- `scripts/train_calibration_from_iu.py` — the training script

---

## 12. Numbers to remember on stage

- **78.6 %** accuracy on 42 hand labels
- **0.87** F1 for supported
- **0.31** F1 for indeterminate (the weak spot)
- **9 / 11** ambiguous cases over-claimed as supported (failure mode)
- **MLP: 128 → 64 → 3** — 9,603 parameters
- **Trained on 6,514** real IU claims, **100 epochs**, cross-entropy, Adam, ~2 s on CPU
- **3 pretrained** (classifier, detector, VLM) + **1 trained from scratch** (MLP)
- **7 agents** in LangGraph; bounded retry `MAX_RECHECKS = 1`
- **92 tests passing**

---

## 13. Short truthful lines you can speak out loud

- *"VisionGuard audits AI-generated chest X-ray reports claim-by-claim, flagging unsupported sentences as potential hallucinations."*
- *"We use three pretrained vision models and trained one small MLP from scratch on 6,514 real IU chest X-ray claims."*
- *"On 42 hand-labelled claim/image pairs the pipeline agrees with the annotator 78.6 % of the time; we document an over-claiming failure mode on ambiguous images."*
- *"The orchestration is agentic — LangGraph with 7 specialised agents, shared state, conditional routing, bounded retry, and a human-in-the-loop gate."*
- *"Not clinically validated. Research prototype only."*

---

## End of brief

If I ask something you can't answer from this context, say *"not in the brief"* — don't guess. If I ask for a one-liner for stage, give me one short sentence I can speak confidently.
