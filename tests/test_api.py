import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import predictor as predictor_module
from app.detection import FaceDetectionError
from app.main import app
from app.model import build_model


def png_bytes() -> bytes:
    pixels = np.random.default_rng(1).integers(0, 255, (320, 320, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_face(monkeypatch, symmetric_landmarks):
    monkeypatch.setattr(predictor_module, "detect", lambda _bytes: (symmetric_landmarks, 320, 320))


@pytest.fixture
def geometry_client(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with TestClient(app) as client:
        yield client


def test_health(geometry_client):
    assert geometry_client.get("/health").json() == {"status": "ok"}


def test_info_reports_geometry_mode_without_weights(geometry_client):
    body = geometry_client.get("/api/info").json()
    assert body["mode"] == "geometry" and body["scale"] == "1-5"


def test_score_in_geometry_mode(geometry_client, fake_face):
    response = geometry_client.post("/api/score", files={"file": ("a.png", png_bytes(), "image/png")})
    body = response.json()
    assert response.status_code == 200 and body["mode"] == "geometry"
    assert 1.0 <= body["score"] <= 5.0
    assert len(body["dimensions"]) == 6


def test_score_in_cnn_mode(tmp_path, monkeypatch, fake_face):
    weights = tmp_path / "w.pth"
    torch.save({"state_dict": build_model().state_dict()}, weights)
    monkeypatch.setenv("MODEL_PATH", str(weights))
    with TestClient(app) as client:
        assert client.get("/api/info").json()["mode"] == "cnn"
        body = client.post("/api/score", files={"file": ("a.png", png_bytes(), "image/png")}).json()
    assert body["mode"] == "cnn" and 1.0 <= body["score"] <= 5.0


def test_missing_weights_file_fails_at_startup(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "does/not/exist.pth")
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


def test_no_face_is_a_400_with_a_code(geometry_client, monkeypatch):
    def boom(_bytes):
        raise FaceDetectionError("no_face", "No face was detected in the image.")

    monkeypatch.setattr(predictor_module, "detect", boom)
    response = geometry_client.post("/api/score", files={"file": ("a.png", png_bytes(), "image/png")})
    assert response.status_code == 400 and response.json()["code"] == "no_face"


def test_empty_upload_is_rejected(geometry_client):
    response = geometry_client.post("/api/score", files={"file": ("a.png", b"", "image/png")})
    assert response.status_code == 400 and response.json()["code"] == "image_unreadable"


def test_oversized_upload_is_rejected(geometry_client, monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 10)
    response = geometry_client.post("/api/score", files={"file": ("a.png", b"x" * 50, "image/png")})
    assert response.status_code == 400 and response.json()["code"] == "file_too_large"
