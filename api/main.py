from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

_WEB_DIR = Path(__file__).resolve().parents[1] / "frontend" / "web"

from api.review import ReviewIn, ReviewRow, list_reviews, record_review
from graph.build import run_audit
from medical import AuditReport, AuditRequest

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
# PNG/JPEG magic bytes — never trust the filename alone.
_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")

app = FastAPI(
    title="VisionGuard Medical",
    description="Research prototype: agentic auditor for AI-generated chest X-ray reports. NOT clinically validated.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve the static frontend. index.html lives at /, assets under /static/*.
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "service": "visionguard-medical", "phase": 4}


@app.get("/")
def root():
    index = _WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "name": "VisionGuard Medical",
        "disclaimer": "Research prototype only. Not for clinical use.",
        "docs": "/docs",
    }


@app.get("/info")
def info():
    return {
        "name": "VisionGuard Medical",
        "disclaimer": "Research prototype only. Not for clinical use.",
        "docs": "/docs",
    }


def _validate_image(data: bytes) -> None:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image exceeds 10 MB limit")
    if not any(data.startswith(m) for m in _IMAGE_MAGIC):
        raise HTTPException(415, "image must be PNG or JPEG")


@app.post("/audit", response_model=AuditReport)
async def audit(
    report_text: str = Form(default=""),
    mode: str = Form(default="normal"),
    image: UploadFile | None = File(default=None),
) -> AuditReport:
    try:
        req = AuditRequest(report_text=report_text, mode=mode)
    except ValidationError as exc:
        # include_context=False drops the raw ValueError objects pydantic puts in ctx,
        # which are not JSON-serializable and crash FastAPI's response encoder.
        raise HTTPException(status_code=422, detail=exc.errors(include_context=False)) from exc

    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read()
        _validate_image(image_bytes)

    if req.mode == "vlm_from_image" and image_bytes is None:
        raise HTTPException(status_code=422, detail="vlm_from_image mode requires an image upload")

    try:
        return run_audit(req.report_text, mode=req.mode, image_bytes=image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/json", response_model=AuditReport)
def audit_json(req: AuditRequest) -> AuditReport:
    """JSON-only variant for programmatic use without image upload."""
    try:
        return run_audit(req.report_text, mode=req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/review", response_model=ReviewRow)
def post_review(row: ReviewIn) -> ReviewRow:
    return record_review(row)


@app.get("/review", response_model=list[ReviewRow])
def get_reviews(audit_id: str) -> list[ReviewRow]:
    return list_reviews(audit_id)
