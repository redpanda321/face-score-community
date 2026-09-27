"""Geometric baseline: facial-proportion measurements from MediaPipe Face Mesh landmarks,
mapped to a 1-5 score by a small model fitted on the Face Research Lab London Set (CC BY 4.0).

Used when no trained SCUT-FBP5500-style CNN weights are configured. Pure functions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Landmarks = np.ndarray  # shape (468, 3), float32, pixel coordinates

# Left/right landmark index pairs on the MediaPipe Face Mesh topology.
# Each tuple is (left index, right index) mirrored about the facial midline.
MIDLINE_PAIRS: list[tuple[int, int]] = [
    (33, 263),    # outer eye corners
    (133, 362),   # inner eye corners
    (61, 291),    # mouth corners
    (234, 454),   # cheek edges
    (127, 356),   # temples
    (93, 323),    # jaw sides
    (58, 288),    # lower jaw
    (172, 397),   # jaw curve
    (136, 365),   # jaw near chin
    (70, 300),    # eyebrow outer
    (63, 293),    # eyebrow mid
    (105, 334),   # eyebrow inner
    (98, 327),    # nose wings
    (205, 425),   # mid cheeks
]

# Landmarks lying on the facial midline, used to fit the midline itself.
MIDLINE_POINTS: list[int] = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 152]


def _normalize_deviation(deviation: float, tolerance: float) -> float:
    """Map a relative deviation to 0-100, where 0 deviation scores 100.

    A deviation equal to ``tolerance`` scores 0. Anything worse clamps at 0.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    ratio = float(deviation) / tolerance
    return float(np.clip((1.0 - ratio) * 100.0, 0.0, 100.0))


def _face_width(landmarks: Landmarks) -> float:
    """Horizontal extent of the face in pixels."""
    return float(landmarks[:, 0].max() - landmarks[:, 0].min())


def _midline_x(landmarks: Landmarks) -> float:
    """Average x of the landmarks that sit on the facial midline."""
    return float(np.mean(landmarks[MIDLINE_POINTS, 0]))


def align_landmarks(landmarks: Landmarks) -> Landmarks:
    """Rotate the face so the outer-eye line is horizontal (removes head roll).

    Without this a tilted head reads as an asymmetric face. Scorers stay pure
    and assume their input is already aligned; ``analyze_landmarks`` does it.
    """
    left, right = landmarks[33, :2], landmarks[263, :2]
    dx, dy = float(right[0] - left[0]), float(right[1] - left[1])
    if dx == 0.0 and dy == 0.0:
        return landmarks
    angle = -np.arctan2(dy, dx)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    centre = (left + right) / 2.0
    aligned = landmarks.copy()
    shifted = landmarks[:, :2] - centre
    aligned[:, 0] = shifted[:, 0] * cos_a - shifted[:, 1] * sin_a + centre[0]
    aligned[:, 1] = shifted[:, 0] * sin_a + shifted[:, 1] * cos_a + centre[1]
    return aligned


# Deviation at which each dimension scores 0. Tuned so real faces spread across
# the full range instead of piling up at one end (see docs/face-score-calibration).
TOLERANCES: dict[str, float] = {
    "symmetry": 0.2762,
    "thirds": 0.1147,
    "fifths": 0.4286,
    "golden_ratio": 0.0792,
    "contour": 0.1143,
    "placement": 0.0695,
}


def _dev_symmetry(landmarks: Landmarks) -> float | None:
    midline = _midline_x(landmarks)
    width = _face_width(landmarks)
    if width <= 0.0:
        return None

    deviations = []
    for left_idx, right_idx in MIDLINE_PAIRS:
        left_distance = abs(landmarks[left_idx, 0] - midline)
        right_distance = abs(landmarks[right_idx, 0] - midline)
        vertical_gap = abs(landmarks[left_idx, 1] - landmarks[right_idx, 1])
        deviations.append((abs(left_distance - right_distance) + vertical_gap) / width)
    return float(np.mean(deviations))


def score_symmetry(landmarks: Landmarks) -> float:
    """Score left-right symmetry: how evenly paired points straddle the midline."""
    deviation = _dev_symmetry(landmarks)
    if deviation is None:
        return 0.0
    return _normalize_deviation(deviation, tolerance=TOLERANCES["symmetry"])


GOLDEN_RATIO = 1.618

# Vertical thirds ("three courts"): hairline, brow, nose base, chin.
THIRDS_POINTS = {"hairline": 10, "brow": 9, "nose_base": 2, "chin": 152}

# Horizontal fifths ("five eyes"): face edge, outer/inner eye corners x2, face edge.
FIFTHS_POINTS = [234, 33, 133, 362, 263, 454]

JAW_CHAIN = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]


def _distance(landmarks: Landmarks, a: int, b: int) -> float:
    """Euclidean distance between two landmarks, in pixels."""
    return float(np.linalg.norm(landmarks[a, :2] - landmarks[b, :2]))


def _feat_thirds(landmarks: Landmarks) -> np.ndarray | None:
    y = {name: float(landmarks[idx, 1]) for name, idx in THIRDS_POINTS.items()}
    segments = [
        y["brow"] - y["hairline"],
        y["nose_base"] - y["brow"],
        y["chin"] - y["nose_base"],
    ]
    total = sum(segments)
    if total <= 0.0 or min(segments) <= 0.0:
        return None
    return np.array(segments) / total


def _feat_fifths(landmarks: Landmarks) -> np.ndarray | None:
    xs = sorted(float(landmarks[idx, 0]) for idx in FIFTHS_POINTS)
    segments = [b - a for a, b in zip(xs, xs[1:])]
    segments = [s for s in segments if s > 0.0]
    if len(segments) < 5:
        return None
    return np.array(segments) / sum(segments)


def _feat_golden_ratio(landmarks: Landmarks) -> np.ndarray | None:
    face_height = _distance(landmarks, 10, 152)
    face_width = _face_width(landmarks)
    mouth_width = _distance(landmarks, 61, 291)
    nose_width = _distance(landmarks, 98, 327)
    if min(face_width, nose_width) <= 0.0:
        return None
    return np.array([face_height / face_width, mouth_width / nose_width])


def _feat_contour(landmarks: Landmarks) -> np.ndarray | None:
    """Jaw taper (jaw width / cheek width) and chin height (lip to chin / face height)."""
    cheek = _distance(landmarks, 234, 454)
    jaw = _distance(landmarks, 172, 397)
    height = _distance(landmarks, 10, 152)
    if min(cheek, height) <= 0.0:
        return None
    chin_height = abs(float(landmarks[152, 1]) - float(landmarks[17, 1])) / height
    return np.array([jaw / cheek, chin_height])


# Proportion-space prototypes: the median of each ratio over a reference set of
# faces. These are instrument-corrected targets (MediaPipe's "hairline" point sits
# on the upper forehead, so literal equal thirds is unreachable) and encode the
# averageness principle in ratios only, never a mean landmark shape.
IDEAL: dict[str, np.ndarray] = {
    "thirds": np.array([0.2030, 0.4022, 0.3957]),
    "fifths": np.array([0.1751, 0.2042, 0.2477, 0.2023, 0.1654]),
    "golden_ratio": np.array([1.1644, 1.6687]),
    "contour": np.array([0.8015, 0.1918]),
}

# Ridge regression of mean human attractiveness rating on the six geometric
# deviations, fitted on the Face Research Lab London Set (DeBruine & Jones, CC BY 4.0,
# doi:10.6084/m9.figshare.5047666.v5; 102 faces, 2513 raters). Nested leave-one-out
# Pearson r = 0.46 (the previous hand-weighted score scored r = 0.31 on the same faces).
# Small, mostly-White, studio-lit sample: treat it as a modest statistical fit, not ground truth.
# Signs are learned, not assumed: e.g. "placement" deviation correlates POSITIVELY with rating.
DEVIATION_ORDER = ("symmetry", "thirds", "fifths", "golden_ratio", "contour", "placement")
RATING_MODEL = {
    "intercept": 3.01890561238423,
    "coef": [-0.0007981747647600841, 0.03653576082769867, -0.044133811728517054,
             -0.1289299792588408, -0.16384196394556533, 0.1993588425535537],
    "mu": [0.025900468886169058, 0.05365430137670168, 0.05857010313540847,
           0.05726108098212977, 0.0688754418385205, 0.05470348363342109],
    "sd": [0.016899499812820077, 0.029530712880089672, 0.026757196896715078,
           0.018830594593920825, 0.040888310769621204, 0.008930624310766948],
    "pred_mean": 3.01890561238423,
    "pred_sd": 0.3539038896710826,
}
STAR_SPREAD = 0.4


def predict_stars(landmarks: Landmarks) -> float:
    """Predict a 1-5 score: fitted rating, z-scored, then squashed through tanh.

    Typical faces land near 3; outliers approach but never leave 1 and 5.
    A degenerate measurement contributes nothing (treated as the training mean).
    """
    aligned = align_landmarks(landmarks)
    rating = RATING_MODEL["intercept"]
    for key, coef, mu, sd in zip(DEVIATION_ORDER, RATING_MODEL["coef"], RATING_MODEL["mu"], RATING_MODEL["sd"]):
        deviation = _DEVIATIONS[key](aligned)
        if deviation is not None:
            rating += coef * (deviation - mu) / sd
    z = (rating - RATING_MODEL["pred_mean"]) / RATING_MODEL["pred_sd"]
    return round(float(3.0 + 2.0 * np.tanh(STAR_SPREAD * z)), 1)


def _relative_deviation(features: np.ndarray | None, key: str) -> float | None:
    if features is None:
        return None
    ideal = IDEAL[key]
    return float(np.mean(np.abs(features - ideal) / ideal))


def _dev_thirds(landmarks: Landmarks) -> float | None:
    return _relative_deviation(_feat_thirds(landmarks), "thirds")


def _dev_fifths(landmarks: Landmarks) -> float | None:
    return _relative_deviation(_feat_fifths(landmarks), "fifths")


def _dev_golden_ratio(landmarks: Landmarks) -> float | None:
    return _relative_deviation(_feat_golden_ratio(landmarks), "golden_ratio")


def _dev_contour(landmarks: Landmarks) -> float | None:
    return _relative_deviation(_feat_contour(landmarks), "contour")


def _dev_placement(landmarks: Landmarks) -> float | None:
    width = _face_width(landmarks)
    height = _distance(landmarks, 10, 152)
    if width <= 0.0 or height <= 0.0:
        return None

    # Inter-eye gap should equal one eye width.
    eye_width = _distance(landmarks, 33, 133)
    eye_gap = _distance(landmarks, 133, 362)
    gap_deviation = abs(eye_gap - eye_width) / width if eye_width > 0.0 else 1.0

    # Mouth centre sits about two-thirds of the way down the face.
    mouth_position = (float(landmarks[13, 1]) - float(landmarks[10, 1])) / height
    mouth_deviation = abs(mouth_position - 0.66)
    return (gap_deviation + mouth_deviation) / 2.0


_DEVIATIONS = {
    "symmetry": _dev_symmetry,
    "thirds": _dev_thirds,
    "fifths": _dev_fifths,
    "golden_ratio": _dev_golden_ratio,
    "contour": _dev_contour,
    "placement": _dev_placement,
}


def _score_from(dev_fn, key: str, landmarks: Landmarks) -> float:
    deviation = dev_fn(landmarks)
    if deviation is None:
        return 0.0
    return _normalize_deviation(deviation, tolerance=TOLERANCES[key])


def score_thirds(landmarks: Landmarks) -> float:
    """Score how evenly the face divides into vertical thirds."""
    return _score_from(_dev_thirds, "thirds", landmarks)


def score_fifths(landmarks: Landmarks) -> float:
    """Score how closely face width divides into five equal eye widths."""
    return _score_from(_dev_fifths, "fifths", landmarks)


def score_golden_ratio(landmarks: Landmarks) -> float:
    """Score how close key facial distance ratios sit to 1.618."""
    return _score_from(_dev_golden_ratio, "golden_ratio", landmarks)


def score_contour(landmarks: Landmarks) -> float:
    """Score jawline smoothness via curvature change along the jaw chain."""
    return _score_from(_dev_contour, "contour", landmarks)


def score_placement(landmarks: Landmarks) -> float:
    """Score eye spacing and the vertical placement of nose and mouth."""
    return _score_from(_dev_placement, "placement", landmarks)


DIMENSION_WEIGHTS: dict[str, float] = {
    "symmetry": 0.25,
    "thirds": 0.15,
    "fifths": 0.15,
    "golden_ratio": 0.15,
    "contour": 0.15,
    "placement": 0.15,
}

DIMENSION_LABELS: dict[str, str] = {
    "symmetry": "Symmetry",
    "thirds": "Vertical Thirds",
    "fifths": "Horizontal Fifths",
    "golden_ratio": "Golden Ratio",
    "contour": "Facial Contour",
    "placement": "Feature Placement",
}

_SCORERS = {
    "symmetry": score_symmetry,
    "thirds": score_thirds,
    "fifths": score_fifths,
    "golden_ratio": score_golden_ratio,
    "contour": score_contour,
    "placement": score_placement,
}

_BAND_THRESHOLDS: list[tuple[float, str, str]] = [
    (85.0, "very_close", "Very close to the classical proportion."),
    (70.0, "close", "Close to the classical proportion, with a slight deviation."),
    (50.0, "moderate", "A moderate deviation from the classical proportion."),
    (0.0, "noticeable", "A noticeable deviation from the classical proportion."),
]


def band_key(score: float) -> str:
    """Return the band key ("very_close"/"close"/"moderate"/"noticeable") for a score.

    Uses the same thresholds as ``_explain`` so the two cannot drift apart.
    """
    for threshold, key, _text in _BAND_THRESHOLDS:
        if score >= threshold:
            return key
    return _BAND_THRESHOLDS[-1][1]


@dataclass
class DimensionScore:
    """One scored dimension, ready to render into a report."""

    key: str
    label: str
    score: float
    explanation: str


def _explain(score: float) -> str:
    """Turn a numeric score into a short human-readable sentence."""
    for threshold, _key, text in _BAND_THRESHOLDS:
        if score >= threshold:
            return text
    return _BAND_THRESHOLDS[-1][2]


def analyze_landmarks(landmarks: Landmarks) -> tuple[float, list[DimensionScore]]:
    """Score all six dimensions and combine them into a weighted overall score.

    Dimension scores are rounded to 1 decimal place; the overall score is
    rounded to 2 decimals to avoid floating-point serialization noise.
    """
    landmarks = align_landmarks(landmarks)
    dimensions = []
    for key, scorer in _SCORERS.items():
        value = round(scorer(landmarks), 1)
        dimensions.append(
            DimensionScore(
                key=key,
                label=DIMENSION_LABELS[key],
                score=value,
                explanation=_explain(value),
            )
        )
    overall = round(sum(d.score * DIMENSION_WEIGHTS[d.key] for d in dimensions), 2)
    return overall, dimensions
