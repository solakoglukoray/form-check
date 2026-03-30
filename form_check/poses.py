"""Pose landmark extraction, joint angle calculations, and orientation checks."""

import math
from typing import NamedTuple

# The joint whose angle is used to detect rep bottom positions (local minima).
KEY_JOINT: dict[str, str] = {
    "squat": "knee",
    "deadlift": "hip",
    "pushup": "elbow",
}


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


def check_orientation(landmarks: list, exercise: str) -> list[str]:
    """
    Inspect landmark visibility to detect camera angle problems.

    Returns a list of warning strings (empty = no issues detected).
    """
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26

    warnings: list[str] = []

    if exercise in ("squat", "deadlift"):
        vis_indices = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]
    else:
        vis_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    vis_scores = [getattr(landmarks[i], "visibility", 1.0) for i in vis_indices]
    avg_vis = sum(vis_scores) / len(vis_scores)

    if avg_vis < 0.5:
        warnings.append(
            f"Low joint visibility ({avg_vis * 100:.0f}%) — "
            "ensure full body is visible in frame."
        )

    if exercise in ("squat", "deadlift"):
        hip_x_diff = abs(landmarks[LEFT_HIP].x - landmarks[RIGHT_HIP].x)
        if hip_x_diff < 0.08:
            warnings.append(
                "Camera appears front-facing — rotate 90 degrees to a side view "
                "for accurate joint angle measurement."
            )

    return warnings


def extract_angles(landmarks: list, exercise: str) -> dict[str, float]:
    """
    Extract relevant joint angles from MediaPipe pose landmarks.

    Args:
        landmarks: NormalizedLandmark list (33 points, x/y in [0,1]).
        exercise: One of "squat", "deadlift", "pushup".

    Returns:
        Dict mapping joint names to angles in degrees.
    """
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
