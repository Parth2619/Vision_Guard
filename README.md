# VisionGuard Medical

Agentic AI auditor for AI-generated chest CT reports. Flags unsupported or hallucinated claims against the source scan.

> ⚠️ **Research prototype. Not clinically validated. Not for diagnostic use.**

## Status

**Phase 6 — Image + bounding-box verification.** Typed claim extraction (Object / Attribute / Count / Spatial), LangGraph workflow with bounded recheck, FastAPI `/audit` multipart endpoint, Streamlit workspace with image upload and bbox overlay. Visual evidence from a pretrained chest-X-ray classifier (`torchxrayvision`, ~250 MB on first use) **and** zero-shot Grounding-DINO detector (~700 MB on first use). Weights download lazily into the local HF/torch cache. Synthetic deterministic offline demo still works without any downloads.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

> Python 3.10–3.12 recommended. Some ML libs added in later phases may not yet publish wheels for 3.13+.

## Run

```powershell
# Backend (terminal 1)
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
# → http://127.0.0.1:8000/health   http://127.0.0.1:8000/docs   POST /audit

# Frontend (terminal 2)
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
# → http://localhost:8501  (set VISIONGUARD_API to point elsewhere if needed)
```

Try the synthetic demo from the Streamlit **Audit Workspace** tab, or:

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/audit -H "content-type: application/json" -d '{\"mode\":\"synthetic_demo\"}'
```

## Test

```powershell
pytest
```

## Architecture

- `medical/` — `Claim`, `Evidence`, `AuditReport` schemas; rule-based extractor; verification rules (missing evidence → `indeterminate`, never `contradicted`); synthetic fixture; evidence providers with explicit provenance tiers T0–T3.
- `graph/build.py` — LangGraph `StateGraph`: validate → extract → evidence → verify → critic → (conditional recheck once) → human_review. Bounded retry via `MAX_RECHECKS`.
- `api/main.py` — FastAPI with `/`, `/health`, `/audit`. CORS restricted to the local Streamlit origin.
- `frontend/app.py` — Streamlit tabs (Overview, Audit Workspace, About) that call the backend and render verdicts, evidence, critic findings, trace, and a JSON download.
- `baselines/text_only.py` — non-agentic comparator: everything `indeterminate` without independent evidence.

## Limitations

- Rule-based extractor only — misses atypical phrasing, abbreviations, and composite findings. LLM backend is pluggable via `medical/extract.py` `ClaimExtractor` protocol but not wired in.
- **Image classifier and Grounding-DINO detector outputs are experimental AI signals, not calibrated disease probabilities.** A bounding box does not prove a medical condition. Treat all image evidence as a weak signal; the verifier only promotes a claim to `supported` / `contradicted` when the model is confidently above/below the configured thresholds, otherwise `indeterminate`.
- Grounding-DINO is zero-shot and general-domain; it has not been trained on chest-X-ray phrases. Expect many misses and false positives for subtle findings.
- Synthetic KB is tiny and labeled synthetic — it is a demo of the pipeline wiring, not a benchmark.
- No persistence. Audits are not stored server-side.
- First real `/audit` with an image stalls ~30–90 s while torchxrayvision + Grounding-DINO weights download. Subsequent calls are seconds.

## Roadmap

| Phase | Scope |
|-------|-------|
| 1 | Repo foundation, API health, Streamlit stub, tests *(current)* |
| 2 | IU Chest X-Ray (Open-i) loader — small reports+images sample, domain types, claim schema. See [docs/dataset_migration_iu_xray.md](docs/dataset_migration_iu_xray.md) |
| 3 | Claim-extraction agent + tests |
| 4 | Visual-evidence agent (image/region features) |
| 5 | Verification + Critic agents, LangGraph wiring |
| 6 | Audit report + human-review escalation |
| 7 | Baselines + evaluation |

## Safety rules

- Audit labels: `supported`, `contradicted`, `indeterminate`.
- Missing evidence → `indeterminate`, never `contradicted`.
- Never commit datasets, PHI, checkpoints, or secrets.
- No external API calls with medical data without explicit opt-in.

See [CLAUDE.md](CLAUDE.md) for architecture and contributor rules.
