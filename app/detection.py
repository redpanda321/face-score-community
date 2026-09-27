"""Face landmark detection using MediaPipe Face Mesh.

Converts raw uploaded image bytes into a (468, 3) pixel-coordinate array,
raising FaceDetectionError with a machine-readable code when the image
cannot be scored.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from app.geometry import Landmarks

MIN_IMAGE_DIMENSION = 200
LANDMARK_COUNT = 468


class FaceDetectionError(Exception):
    """Raised when an image cannot be turned into scoreable landmarks."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _create_face_mesh():
    """Build a Face Mesh detector. Patched out in tests."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,  # detect 2 so we can reject multi-face images
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )


def _decode(image_bytes: bytes) -> np.ndarray:
    """Decode bytes to a BGR image, raising FaceDetectionError on failure."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FaceDetectionError(
            "image_unreadable", "The file could not be read as an image."
        )
    height, width = image.shape[:2]
    if min(height, width) < MIN_IMAGE_DIMENSION:
        raise FaceDetectionError(
            "image_too_small",
            f"The image must be at least {MIN_IMAGE_DIMENSION}px on its shorter side.",
        )
    return image


def detect_landmarks(image_bytes: bytes) -> Landmarks:
    """Detect exactly one face and return its landmarks in pixel coordinates."""
    landmarks, _width, _height = detect(image_bytes)
    return landmarks


def detect(image_bytes: bytes) -> tuple[Landmarks, int, int]:
    """Detect one face, returning its landmarks plus the image's pixel size.

    The size travels with the landmarks because the overlay normalises against
    it, and pixel coordinates alone cannot say what frame they belong to.
    """
    image = _decode(image_bytes)
    height, width = image.shape[:2]

    mesh = _create_face_mesh()
    try:
        result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        mesh.close()

    faces = getattr(result, "multi_face_landmarks", None)
    if not faces:
        raise FaceDetectionError(
            "no_face", "No face was detected in the image."
        )
    if len(faces) > 1:
        raise FaceDetectionError(
            "multiple_faces",
            "More than one face was detected. Please upload a photo with a single face.",
        )

    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    for index, landmark in enumerate(faces[0].landmark[:LANDMARK_COUNT]):
        points[index] = [landmark.x * width, landmark.y * height, landmark.z * width]
    return points, width, height
