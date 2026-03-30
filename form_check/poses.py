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
    z: float = 0.0  # MediaPipe depth estimate relative to hip


def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """
    Return the angle at point b formed by vectors b->a and b->c, in degrees.

    Uses all three dimensions (x, y, z) so the result is correct regardless
    of camera orientation — a camera at 45° or 60° from the side produces the
    same anatomical angle as a perfect side view.
    """
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def angle_from_vertical(a: Landmark, b: Landmark) -> float:
    """
    Return the angle of segment a->b relative to the world vertical axis.

    Uses x and z to measure lateral lean and combines with y (gravity axis),
    so the result is independent of which direction the camera faces.

    A perfectly upright torso returns 0°; forward lean increases the angle.
    """
    # Horizontal displacement in the x-z plane (camera-direction agnostic)
    dx = b.x - a.x
    dz = b.z - a.z
    dy = b.y - a.y  # vertical (MediaPipe y grows downward)

    horizontal = math.sqrt(dx ** 2 + dz ** 2)
    return math.degrees(math.atan2(horizontal, abs(dy)))


def check_orientation(landmarks: list, exercise: str) -> list[str]:
    """
    Inspect landmark visibility to detect camera angle problems.

    With 3D angles the front-facing warning is no longer raised — the z
    coordinate compensates for off-axis cameras. Only very low visibility
    (body not fully in frame) is flagged.

    Returns a list of warning strings (empty = no issues).
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

    if avg_vis < 0.4:
        warnings.append(
            f"Low joint visibility ({avg_vis * 100:.0f}%) — "
            "ensure full body is visible in frame and lighting is good."
        )

    return warnings


def extract_angles(landmarks: list, exercise: str) -> dict[str, float]:
    """
    Extract relevant joint angles from MediaPipe pose landmarks.

    Args:
        landmarks: NormalizedLandmark list (33 points, x/y/z in normalised space).
        exercise: One of "squat", "deadlift", "pushup".

    Returns:
        Dict mapping joint names to 3D angles in degrees.
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
        zs = [getattr(landmarks[i], "z", 0.0) for i in indices]
        return Landmark(
            sum(xs) / len(xs),
            sum(ys) / len(ys),
            sum(zs) / len(zs),
        )

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
