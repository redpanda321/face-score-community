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


def test_star_score_is_centred_and_bounded():
    from app.geometry import COMPOSITE_MEAN, star_score

    assert star_score(COMPOSITE_MEAN) == 3.0
    assert star_score(100.0) <= 5.0
    assert star_score(0.0) >= 1.0
    assert star_score(80.0) > star_score(60.0) > star_score(40.0) > star_score(20.0)


def test_star_score_separates_faces_by_more_than_a_point():
    from app.geometry import star_score

    assert star_score(85.0) - star_score(30.0) > 2.0


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
