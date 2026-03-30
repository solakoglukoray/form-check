"""Tests for form_check — angle math, benchmark scoring, and video pipeline."""

import math
from unittest.mock import MagicMock, patch

import pytest

from form_check.benchmarks import BENCHMARKS, score_angles
from form_check.poses import (  # noqa: E402
    Landmark,
    angle_from_vertical,
    calculate_angle,
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
    # Equilateral triangle — all angles 60 deg
    a = Landmark(0.0, 0.0)
    b = Landmark(1.0, 0.0)
    c = Landmark(0.5, math.sqrt(3) / 2)
    assert abs(calculate_angle(a, b, c) - 60.0) < 0.1


def test_calculate_angle_zero_magnitude_returns_zero():
    a = Landmark(0.0, 0.0)
    b = Landmark(0.0, 0.0)  # same as a -> zero vector
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
    # Only knee provided — should still return a score based on that joint
    result = score_angles({"knee": 90.0}, "squat")
    assert result["score"] == 100


# ---------------------------------------------------------------------------
# analyze_video — mocked pipeline
# ---------------------------------------------------------------------------


@patch("form_check.main.cv2")
@patch("form_check.main.mp")
def test_analyze_video_no_pose_returns_zero(mock_mp, mock_cv2):
    """When pose detection yields no landmarks, returns zero-score result."""
    mock_cap = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    # isOpened: True (initial check), True (while iteration), False (exit loop)
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
    assert len(result["feedback"]) > 0
