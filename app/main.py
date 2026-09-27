"""Face Score Community: a free, self-hosted facial beauty scoring API and web page.

No accounts, no payments, no telemetry. Uploaded photos are scored in memory and
never written to disk.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.detection import FaceDetectionError
from app.predictor import BeautyPredictor

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

NOTICE = (
    "For entertainment and research only. Scores reflect a statistical model of "
    "average rater preferences, not a judgement of anyone's worth. Any SCUT-FBP5500-"
    "derived weights are limited to non-commercial use; see NOTICE.md."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = BeautyPredictor(os.getenv("MODEL_PATH") or None)
    yield


app = FastAPI(title="Face Score Community", version="0.1.0", lifespan=lifespan)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "detail": message})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info")
async def info() -> dict[str, str]:
    return {"mode": app.state.predictor.mode, "scale": "1-5", "notice": NOTICE}


@app.post("/api/score")
async def score(file: UploadFile = File(...)):
    """Score one single-face photo on a 1-5 scale."""
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not image_bytes:
        return _error(400, "image_unreadable", "The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return _error(400, "file_too_large", f"The image must be under {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")

    try:
        result = app.state.predictor.predict(image_bytes)
    except FaceDetectionError as exc:
        return _error(400, exc.code, str(exc))
    return {**result, "notice": NOTICE}


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")
