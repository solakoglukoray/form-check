"""Tests for form_check — angle math, benchmark scoring, rep detection, orientation."""

import math
from unittest.mock import MagicMock, patch

import pytest

from form_check.benchmarks import BENCHMARKS, score_angles
from form_check.main import find_rep_peaks
from form_check.poses import (
    KEY_JOINT,
    Landmark,
    angle_from_vertical,
    calculate_angle,
    check_orientation,
)

# ---------------------------------------------------------------------------
# calculate_angle
# ---------------------------------------------------------------------------


def test_calculate_angle_right_angle():
    a = Landmark(0.0, 1.0)
    b = Landmark(0.0, 0.0)
    c = Landmark(1.0, 0.0)
    assert abs(calculate_angle(a, b, c) - 90.0) < 0.01


def test_calculate_angle_straight_line():
    a = Landmark(0.0, 0.0)
    b = Landmark(1.0, 0.0)
    c = Landmark(2.0, 0.0)
    assert abs(calculate_angle(a, b, c) - 180.0) < 0.01


def test_calculate_angle_60_degrees():
    a = Landmark(0.0, 0.0)
    b = Landmark(1.0, 0.0)
    c = Landmark(0.5, math.sqrt(3) / 2)
    assert abs(calculate_angle(a, b, c) - 60.0) < 0.1


def test_calculate_angle_zero_magnitude_returns_zero():
    a = Landmark(0.0, 0.0)
    b = Landmark(0.0, 0.0)
    c = Landmark(1.0, 0.0)
    assert calculate_angle(a, b, c) == 0.0


# ---------------------------------------------------------------------------
# angle_from_vertical
# ---------------------------------------------------------------------------


def test_angle_from_vertical_straight_up():
    assert abs(angle_from_vertical(Landmark(0.0, 0.0), Landmark(0.0, 1.0))) < 0.01


def test_angle_from_vertical_45_degrees():
    assert abs(angle_from_vertical(Landmark(0.0, 0.0), Landmark(1.0, 1.0)) - 45.0) < 0.1


# ---------------------------------------------------------------------------
# score_angles — benchmark scoring
# ---------------------------------------------------------------------------


def test_score_perfect_squat():
    angles = {"knee": 90.0, "hip": 80.0, "back": 30.0}
    result = score_angles(angles, "squat")
    assert result["score"] == 100
    assert all("Good" in f for f in result["feedback"])


def test_score_bad_squat_knee_not_bent():
    angles = {"knee": 170.0, "hip": 80.0, "back": 30.0}
    result = score_angles(angles, "squat")
    assert result["score"] < 80
    knee_feedback = next(f for f in result["feedback"] if "Knee" in f)
    assert "Too high" in knee_feedback


def test_score_perfect_pushup():
    angles = {"elbow": 90.0, "hip": 170.0}
    result = score_angles(angles, "pushup")
    assert result["score"] == 100


def test_score_pushup_sagging_hips():
    angles = {"elbow": 90.0, "hip": 130.0}
    result = score_angles(angles, "pushup")
    assert result["score"] < 100
    hip_feedback = next(f for f in result["feedback"] if "hip" in f.lower())
    assert "Too low" in hip_feedback


def test_score_perfect_deadlift():
    angles = {"hip": 50.0, "knee": 110.0, "back": 45.0}
    result = score_angles(angles, "deadlift")
    assert result["score"] == 100


def test_score_unknown_exercise_raises():
    with pytest.raises(ValueError, match="Unknown exercise"):
        score_angles({}, "burpee")


def test_all_exercises_defined_in_benchmarks():
    for exercise in ("squat", "deadlift", "pushup"):
        assert exercise in BENCHMARKS
        assert len(BENCHMARKS[exercise]) >= 2


def test_score_partial_angles_still_scores():
    result = score_angles({"knee": 90.0}, "squat")
    assert result["score"] == 100


# ---------------------------------------------------------------------------
# find_rep_peaks — rep bottom detection
# ---------------------------------------------------------------------------


def test_find_rep_peaks_single_rep():
    # One clear dip in the middle
    angles = [170.0, 150.0, 90.0, 150.0, 170.0]
    peaks = find_rep_peaks(angles)
    assert peaks == [2]


def test_find_rep_peaks_three_reps():
    # Three dips separated by standing positions; min_distance=2 matches spacing
    angles = [170.0, 90.0, 170.0, 88.0, 170.0, 92.0, 170.0]
    peaks = find_rep_peaks(angles, min_distance=2)
    assert len(peaks) == 3
    assert 1 in peaks and 3 in peaks and 5 in peaks


def test_find_rep_peaks_too_short_returns_all():
    # Series with fewer than 3 points → return all indices
    assert find_rep_peaks([90.0]) == [0]
    assert find_rep_peaks([90.0, 100.0]) == [0, 1]


def test_find_rep_peaks_flat_series_returns_empty():
    # Flat line has no local minimum
    angles = [170.0, 170.0, 170.0, 170.0]
    peaks = find_rep_peaks(angles)
    assert peaks == []


def test_find_rep_peaks_min_distance_enforced():
    # Two close minima — only the first should survive with min_distance=3
    angles = [170.0, 90.0, 88.0, 170.0]
    peaks = find_rep_peaks(angles, min_distance=3)
    # idx 1 and 2 are both minima but only 1 frame apart; 1 is detected first
    assert len(peaks) == 1


# ---------------------------------------------------------------------------
# check_orientation — camera angle detection
# ---------------------------------------------------------------------------


class _LM:
    """Minimal landmark stub for orientation tests."""

    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def _make_landmarks(overrides: dict[int, _LM] | None = None) -> list[_LM]:
    """Build a 33-landmark list with sensible side-view defaults."""
    # Default: plausible side-view positions, all fully visible
    defaults = {i: _LM(0.5, 0.5) for i in range(33)}
    # MediaPipe indices: shoulders 11/12, hips 23/24, knees 25/26
    defaults[11] = _LM(0.3, 0.3)   # left shoulder
    defaults[12] = _LM(0.7, 0.3)   # right shoulder (far away x → side view)
    defaults[23] = _LM(0.3, 0.6)   # left hip
    defaults[24] = _LM(0.7, 0.6)   # right hip (far away x)
    defaults[25] = _LM(0.3, 0.8)   # left knee
    defaults[26] = _LM(0.7, 0.8)   # right knee
    if overrides:
        defaults.update(overrides)
    return [defaults[i] for i in range(33)]


def test_check_orientation_clean_side_view():
    lms = _make_landmarks()
    warnings = check_orientation(lms, "squat")
    assert warnings == []


def test_check_orientation_low_visibility_warns():
    # All key joints barely visible
    overrides = {
        23: _LM(0.3, 0.6, visibility=0.2),
        24: _LM(0.7, 0.6, visibility=0.2),
        25: _LM(0.3, 0.8, visibility=0.2),
        26: _LM(0.7, 0.8, visibility=0.2),
    }
    lms = _make_landmarks(overrides)
    warnings = check_orientation(lms, "squat")
    assert any("visibility" in w.lower() for w in warnings)


def test_check_orientation_front_facing_no_longer_warns():
    # With 3D angle math the front-facing check is removed — z compensates.
    # A front-facing camera should NOT trigger a warning anymore.
    overrides = {
        23: _LM(0.49, 0.6),  # left hip
        24: _LM(0.51, 0.6),  # right hip — nearly same x (would look front-facing)
    }
    lms = _make_landmarks(overrides)
    warnings = check_orientation(lms, "squat")
    assert not any("front" in w.lower() for w in warnings)


def test_check_orientation_pushup_no_front_check():
    # Front-facing detection only applies to squat/deadlift
    overrides = {
        23: _LM(0.49, 0.6),
        24: _LM(0.51, 0.6),
    }
    lms = _make_landmarks(overrides)
    warnings = check_orientation(lms, "pushup")
    assert not any("front" in w.lower() for w in warnings)


def test_key_joint_defined_for_all_exercises():
    for exercise in ("squat", "deadlift", "pushup"):
        assert exercise in KEY_JOINT


# ---------------------------------------------------------------------------
# analyze_video — mocked pipeline
# ---------------------------------------------------------------------------


@patch("form_check.main.cv2")
@patch("form_check.main.mp")
def test_analyze_video_no_pose_returns_zero(mock_mp, mock_cv2):
    """When pose detection yields no landmarks, returns zero-score result."""
    mock_cap = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cap.isOpened.side_effect = [True, True, False]
    mock_cap.read.return_value = (True, MagicMock())
    mock_cv2.cvtColor.return_value = MagicMock()

    mock_pose_ctx = MagicMock()
    mock_pose_ctx.__enter__ = MagicMock(return_value=mock_pose_ctx)
    mock_pose_ctx.__exit__ = MagicMock(return_value=False)
    mock_result = MagicMock()
    mock_result.pose_landmarks = None
    mock_pose_ctx.process.return_value = mock_result
    mock_mp.solutions.pose.Pose.return_value = mock_pose_ctx

    from form_check.main import analyze_video

    result = analyze_video("dummy.mp4", "squat")
    assert result["avg_score"] == 0
    assert result["frames_analyzed"] == 0
    assert result["rep_count"] == 0
    assert len(result["feedback"]) > 0
