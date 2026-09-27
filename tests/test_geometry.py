import numpy as np

from app.geometry import (
    DIMENSION_WEIGHTS,
    _normalize_deviation,
    analyze_landmarks,
    band_key,
    score_contour,
    score_fifths,
    score_golden_ratio,
    score_placement,
    score_symmetry,
    score_thirds,
)


def test_perfectly_symmetric_face_scores_near_100(symmetric_landmarks):
    assert score_symmetry(symmetric_landmarks) > 95.0


def test_asymmetric_face_scores_lower_than_symmetric(
    symmetric_landmarks, asymmetric_landmarks
):
    assert score_symmetry(asymmetric_landmarks) < score_symmetry(symmetric_landmarks)


def test_score_is_always_within_bounds(asymmetric_landmarks):
    assert 0.0 <= score_symmetry(asymmetric_landmarks) <= 100.0


def test_wildly_distorted_face_clamps_at_zero(symmetric_landmarks):
    points = symmetric_landmarks.copy()
    points[points[:, 0] > 200.0, 0] += 5000.0
    assert score_symmetry(points) == 0.0


def test_normalize_deviation_zero_deviation_is_100():
    assert _normalize_deviation(0.0, tolerance=0.1) == 100.0


ALL_DIMENSION_SCORERS = [
    score_thirds,
    score_fifths,
    score_golden_ratio,
    score_contour,
    score_placement,
]


def test_every_dimension_stays_within_bounds(symmetric_landmarks):
    for scorer in ALL_DIMENSION_SCORERS:
        value = scorer(symmetric_landmarks)
        assert 0.0 <= value <= 100.0, f"{scorer.__name__} returned {value}"


def test_weights_sum_to_one():
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9


def test_analyze_returns_six_dimensions(symmetric_landmarks):
    overall, dimensions = analyze_landmarks(symmetric_landmarks)
    assert len(dimensions) == 6
    assert {d.key for d in dimensions} == set(DIMENSION_WEIGHTS)
    assert 0.0 <= overall <= 100.0


def test_overall_is_the_weighted_mean_of_dimensions(symmetric_landmarks):
    overall, dimensions = analyze_landmarks(symmetric_landmarks)
    expected = sum(d.score * DIMENSION_WEIGHTS[d.key] for d in dimensions)
    assert abs(overall - expected) < 0.01


def test_every_dimension_has_a_nonempty_explanation(symmetric_landmarks):
    _, dimensions = analyze_landmarks(symmetric_landmarks)
    for dimension in dimensions:
        assert dimension.explanation.strip()


def _stars_with(monkeypatch, symmetric_landmarks, **overrides):
    import app.geometry as scoring

    for key, mu in zip(scoring.DEVIATION_ORDER, scoring.RATING_MODEL["mu"]):
        value = overrides.get(key, mu)
        monkeypatch.setitem(scoring._DEVIATIONS, key, lambda _lm, v=value: v)
    return scoring.predict_stars(symmetric_landmarks)


def test_predict_stars_is_bounded_on_real_shaped_input(symmetric_landmarks, asymmetric_landmarks):
    from app.geometry import predict_stars

    for landmarks in (symmetric_landmarks, asymmetric_landmarks):
        assert 1.0 <= predict_stars(landmarks) <= 5.0


def test_average_measurements_score_three(monkeypatch, symmetric_landmarks):
    assert _stars_with(monkeypatch, symmetric_landmarks) == 3.0


def test_learned_signs(monkeypatch, symmetric_landmarks):
    """Signs come from the fit: contour/golden deviation hurts, placement deviation helps."""
    import app.geometry as scoring

    base = _stars_with(monkeypatch, symmetric_landmarks)
    mu = dict(zip(scoring.DEVIATION_ORDER, scoring.RATING_MODEL["mu"]))
    sd = dict(zip(scoring.DEVIATION_ORDER, scoring.RATING_MODEL["sd"]))
    worse_contour = _stars_with(monkeypatch, symmetric_landmarks, contour=mu["contour"] + 2 * sd["contour"])
    worse_golden = _stars_with(monkeypatch, symmetric_landmarks, golden_ratio=mu["golden_ratio"] + 2 * sd["golden_ratio"])
    more_placement = _stars_with(monkeypatch, symmetric_landmarks, placement=mu["placement"] + 2 * sd["placement"])
    assert worse_contour < base and worse_golden < base and more_placement > base


def test_missing_measurement_is_neutral(monkeypatch, symmetric_landmarks):
    assert _stars_with(monkeypatch, symmetric_landmarks, thirds=None) == 3.0


def test_stars_spread_is_meaningful(monkeypatch, symmetric_landmarks):
    import app.geometry as scoring

    mu = dict(zip(scoring.DEVIATION_ORDER, scoring.RATING_MODEL["mu"]))
    sd = dict(zip(scoring.DEVIATION_ORDER, scoring.RATING_MODEL["sd"]))
    good = _stars_with(monkeypatch, symmetric_landmarks, contour=mu["contour"] - 2 * sd["contour"], golden_ratio=mu["golden_ratio"] - 2 * sd["golden_ratio"], placement=mu["placement"] + 2 * sd["placement"])
    bad = _stars_with(monkeypatch, symmetric_landmarks, contour=mu["contour"] + 2 * sd["contour"], golden_ratio=mu["golden_ratio"] + 2 * sd["golden_ratio"], placement=mu["placement"] - 2 * sd["placement"])
    assert good - bad > 1.5


def test_align_removes_head_roll(symmetric_landmarks):
    from app.geometry import align_landmarks

    theta = np.deg2rad(15.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tilted = symmetric_landmarks.copy()
    tilted[:, :2] = symmetric_landmarks[:, :2] @ rot.T
    realigned = align_landmarks(tilted)
    left, right = realigned[33, :2], realigned[263, :2]
    assert abs(left[1] - right[1]) < 1e-3


def test_band_key_boundaries():
    assert band_key(100.0) == "very_close"
    assert band_key(85.0) == "very_close"
    assert band_key(84.9) == "close"
    assert band_key(70.0) == "close"
    assert band_key(69.9) == "moderate"
    assert band_key(50.0) == "moderate"
    assert band_key(49.9) == "noticeable"
    assert band_key(0.0) == "noticeable"
