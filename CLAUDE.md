# VisionGuard Medical — Contributor & Agent Guide

Research prototype that audits AI-generated chest CT reports for unsupported or hallucinated claims. **Not clinically validated. Not for diagnostic use.**

## Architecture (target state)

Agentic pipeline orchestrated with **LangGraph**:

1. Supervisor / Orchestrator
2. Input Validation
3. Atomic Claim Extraction
4. Visual Evidence
5. Claim Verification
6. Critic
7. Audit Report
8. Human-review escalation

Audit labels: `supported`, `contradicted`, `indeterminate`. **Missing evidence is never `contradicted`** — it is `indeterminate`. The system must surface uncertainty.

Backend: FastAPI. Frontend: Streamlit. ML: PyTorch + scikit-learn. **Dataset: IU Chest X-Ray (NLM Open-i)** — 2D chest X-rays + XML reports; see [docs/dataset_migration_iu_xray.md](docs/dataset_migration_iu_xray.md). RadGenome-ChestCT was explored in Phases 2A–2F and deprecated.

## Repository layout

```
agents/     # LangGraph agents (added phase-by-phase)
graph/      # LangGraph wiring / state
medical/    # Domain logic: claims, ontologies, label rules
models/     # PyTorch models / wrappers
datasets/   # Loaders only — NEVER commit data
baselines/  # Baseline comparators
api/        # FastAPI app
frontend/   # Streamlit UI
tests/      # pytest
configs/    # YAML / env-driven config
notebooks/  # Exploratory
artifacts/  # Local run outputs — NEVER commit
```

## Development rules

- Python 3.10–3.12. Windows is the primary dev target; PowerShell commands below.
- Keep configuration (`.env`, `configs/`) separate from code. No hard-coded keys or paths.
- Phases are additive and testable. Do not scaffold empty agent files ahead of their phase.
- Prefer small, documentation-supported dependencies. Pin or bound every dependency.
- Handle errors explicitly. No bare `except:`. No silent fallbacks on medical logic.
- Never make external API calls with medical data unless the user has explicitly opted in.
- Never commit: datasets, CT scans, reports, PHI, checkpoints, `.env`, secrets.
- Do not download the full RadGenome-ChestCT dataset locally — use a small sample.

## Testing

- `pytest` must pass before any phase is declared complete.
- Every new behavior ships with at least one test.
- Do not claim tests passed without running them.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
uvicorn api.main:app --reload
streamlit run frontend/app.py
```
