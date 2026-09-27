"""Shared fixtures: synthetic landmark arrays with known geometry.

The fixtures are built against the REAL MediaPipe indices the scorer reads
(`MIDLINE_PAIRS`, `MIDLINE_POINTS`), not against consecutive index pairs.
A fixture that mirrors arbitrary consecutive indices does not exercise the
production constants at all, and lets a distorted face drag the midline with
it — which would make a wildly distorted face look perfectly symmetric.
"""

import numpy as np
import pytest

from app.geometry import MIDLINE_PAIRS, MIDLINE_POINTS

FACE_WIDTH = 400.0
FACE_HEIGHT = 600.0
MIDLINE_X = 200.0


@pytest.fixture
def symmetric_landmarks() -> np.ndarray:
    """A synthetic face mirrored about x=200 across every scored pair."""
    rng = np.random.default_rng(seed=42)
    points = np.zeros((468, 3), dtype=np.float32)

    # Plausible noise everywhere first, so the face has a realistic width.
    points[:, 0] = rng.uniform(MIDLINE_X - 180.0, MIDLINE_X + 180.0, size=468)
    points[:, 1] = rng.uniform(0.0, FACE_HEIGHT, size=468)

    # Exact mirror symmetry on the pairs the scorer actually reads.
    for left_idx, right_idx in MIDLINE_PAIRS:
        offset = rng.uniform(10.0, 180.0)
        y = rng.uniform(0.0, FACE_HEIGHT)
        points[left_idx] = [MIDLINE_X - offset, y, 0.0]
        points[right_idx] = [MIDLINE_X + offset, y, 0.0]

    # Midline landmarks define the midline, so pin them exactly onto it.
    for idx in MIDLINE_POINTS:
        points[idx, 0] = MIDLINE_X

    return points


@pytest.fixture
def asymmetric_landmarks(symmetric_landmarks: np.ndarray) -> np.ndarray:
    """The same face with every scored right-side point shoved 40px outward."""
    points = symmetric_landmarks.copy()
    for _left_idx, right_idx in MIDLINE_PAIRS:
        points[right_idx, 0] += 40.0
    return points
