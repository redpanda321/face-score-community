"""Scoring front door: a CNN when weights are configured, otherwise the geometric baseline."""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from app.detection import _decode, detect
from app.geometry import DIMENSION_LABELS, analyze_landmarks, band_key, predict_stars

logger = logging.getLogger(__name__)

# The SCUT-FBP5500 faces are tight frontal crops; give the detector's box a small margin.
CROP_MARGIN = 0.25


def crop_face(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Square RGB crop around the landmark bounding box, clamped to the image."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    centre_x, centre_y = float(xs.mean()), float(ys.mean())
    side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) * (1.0 + CROP_MARGIN)
    height, width = image_bgr.shape[:2]
    half = side / 2.0
    left, right = int(max(0, centre_x - half)), int(min(width, centre_x + half))
    top, bottom = int(max(0, centre_y - half)), int(min(height, centre_y + half))
    return cv2.cvtColor(image_bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)


class BeautyPredictor:
    """Scores one face photo. ``mode`` is ``"cnn"`` or ``"geometry"``."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._model = None
        if weights_path:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"MODEL_PATH does not exist: {weights_path}")
            from app.model import load_weights

            self._model = load_weights(weights_path)
            logger.info("Loaded CNN weights from %s", weights_path)
        else:
            logger.warning("No MODEL_PATH set: using the geometric baseline, not a trained CNN.")

    @property
    def mode(self) -> str:
        return "cnn" if self._model is not None else "geometry"

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Return ``{"score": 1-5, "mode": ..., ...}``; raises FaceDetectionError."""
        landmarks, _width, _height = detect(image_bytes)

        if self._model is not None:
            from app.model import predict_score

            face = crop_face(_decode(image_bytes), landmarks)
            return {"score": predict_score(self._model, face), "mode": "cnn"}

        _composite, dimensions = analyze_landmarks(landmarks)
        return {
            "score": predict_stars(landmarks),
            "mode": "geometry",
            "dimensions": [
                {
                    "key": d.key,
                    "label": DIMENSION_LABELS[d.key],
                    "score": d.score,
                    "band": band_key(d.score),
                }
                for d in dimensions
            ],
        }
