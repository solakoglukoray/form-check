"""Pose landmark extraction and joint angle calculations."""

import math
from typing import NamedTuple


class Landmark(NamedTuple):
    x: float
    y: float


def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Return the angle at point b formed by points a-b-c, in degrees."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def angle_from_vertical(a: Landmark, b: Landmark) -> float:
    """Return the angle of segment a->b relative to vertical (0 = straight up)."""
    dx = b.x - a.x
    dy = b.y - a.y
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def extract_angles(landmarks: list, exercise: str) -> dict[str, float]:
    """
    Extract relevant joint angles from MediaPipe pose landmarks.

    Args:
        landmarks: MediaPipe pose_landmarks.landmark list (33 points).
        exercise: One of "squat", "deadlift", "pushup".

    Returns:
        Dict mapping joint names to angles in degrees.
    """
    # MediaPipe Pose landmark indices
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16

    def lm_avg(*indices: int) -> Landmark:
        xs = [landmarks[i].x for i in indices]
        ys = [landmarks[i].y for i in indices]
        return Landmark(sum(xs) / len(xs), sum(ys) / len(ys))

    angles: dict[str, float] = {}

    if exercise in ("squat", "deadlift"):
        shoulder = lm_avg(LEFT_SHOULDER, RIGHT_SHOULDER)
        hip = lm_avg(LEFT_HIP, RIGHT_HIP)
        knee = lm_avg(LEFT_KNEE, RIGHT_KNEE)
        ankle = lm_avg(LEFT_ANKLE, RIGHT_ANKLE)

        angles["knee"] = calculate_angle(hip, knee, ankle)
        angles["hip"] = calculate_angle(shoulder, hip, knee)
        angles["back"] = angle_from_vertical(hip, shoulder)

    elif exercise == "pushup":
        shoulder = lm_avg(LEFT_SHOULDER, RIGHT_SHOULDER)
        elbow = lm_avg(LEFT_ELBOW, RIGHT_ELBOW)
        wrist = lm_avg(LEFT_WRIST, RIGHT_WRIST)
        hip = lm_avg(LEFT_HIP, RIGHT_HIP)
        knee = lm_avg(LEFT_KNEE, RIGHT_KNEE)

        angles["elbow"] = calculate_angle(shoulder, elbow, wrist)
        angles["hip"] = calculate_angle(shoulder, hip, knee)

    return angles
